from typing import Dict, List, Any, Tuple, Optional


def get_seq_length_from_request(request_inputs: Dict[str, Any]) -> int:
    """
    Extract the sequence length from a request's inputs.
    
    Args:
        request_inputs: The inputs dictionary of a request
        
    Returns:
        The sequence length value (defaults to 4096 if not found)
    """
    # Try to get mask_seq_length directly from inputs
    seq_length = request_inputs.get('mask_seq_length')
    if seq_length is not None:
        return seq_length
    
    # If not available, use a default value
    return 4096
def cal_compute_time(request_seq_length):
    flops = cal_flops_flux(request_seq_length)/1e9
    # compute_time = 2.037649e-05 * flops + 0.206068
    # ootd: 3.577125e-04 * flops + 0.478224
    compute_time = 2.235931e-05 * flops + 0.057528
    return compute_time
def cal_load_time(batch_size):
    load_time_dict = {1:0.791073,2: 1.662352,3: 2.367406,4: 2.810369,5: 3.797935,6: 4.000628,7: 4.927391,8: 7.043778,}
    return load_time_dict[batch_size]
def cal_real_time(seqlen_list,batch_size):
    compute_time = 0
    for seqlen in seqlen_list:
        compute_time += cal_compute_time(seqlen)
    load_time = cal_load_time(batch_size)
    return max(compute_time,load_time)
def cal_real_time_steps(seqlen_list, batch_size, remaining_steps):
    """
    计算考虑不同剩余步数的总处理时间
    
    Args:
        seqlen_list: 每个请求的序列长度列表
        batch_size: 当前批次大小
        remaining_steps: 每个请求剩余的步数列表
    
    Returns:
        估计的总处理时间
    """
    if not seqlen_list or not remaining_steps or len(seqlen_list) != len(remaining_steps):
        return 0
    
    # 排序剩余步数，从小到大
    sorted_data = sorted(zip(remaining_steps, seqlen_list), key=lambda x: x[0])
    sorted_remaining_steps = [item[0] for item in sorted_data]
    sorted_seqlen_list = [item[1] for item in sorted_data]
    
    total_time = 0
    prev_step = 0
    current_batch_size = len(sorted_remaining_steps)
    
    # 处理每个步数区间
    for i, steps in enumerate(sorted_remaining_steps):
        if steps > prev_step:
            # 当前步数区间内的序列长度
            current_seqlens = sorted_seqlen_list[i:]
            # 当前区间内的步数
            step_count = steps - prev_step
            # 计算这个区间的总时间
            per_step_compute_time = sum(cal_compute_time(seqlen) for seqlen in current_seqlens)
            per_step_load_time = cal_load_time(current_batch_size)
            per_step_time = max(per_step_compute_time, per_step_load_time)
            # 将时间乘以步数加到总时间上
            total_time += per_step_time * step_count
            
            prev_step = steps
            current_batch_size = len(sorted_remaining_steps) - (i + 1)
    
    return total_time
class Scheduler:
    """
    Base abstract scheduler class that defines the interface for different scheduling strategies.
    """
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     req_id: str) -> str:
        """
        Select a worker to assign the request to.
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The selected worker ID
        """
        raise NotImplementedError("Subclasses must implement select_worker")
    
    def get_request_log_info(self, request_seq_length: int, pipeline_name) -> str:
        """
        Generate additional log information for a request.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with additional log information
        """
        # Default implementation returns empty string
        return ""
    
    def update_worker_info(self,
                          worker_id: str,
                          req_id: str,
                          request_inputs: Dict[str, Any],
                          worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Update worker batch information after assigning a request.
        
        Args:
            worker_id: The worker ID the request was assigned to
            req_id: The request ID
            request_inputs: The request inputs
            worker_batch_info: Current batch information for each worker
            
        Returns:
            Updated worker batch information
        """
        # Default implementation to update worker_batch_info
        if worker_id not in worker_batch_info:
            worker_batch_info[worker_id] = []
            
        worker_batch_info[worker_id].append({
            'request_id': req_id,
            'mask_seq_length': get_seq_length_from_request(request_inputs),
            'inputs': request_inputs,
        })
        
        return worker_batch_info
    
    def remove_completed_request(self,
                                worker_id: str,
                                completed_req_id: str,
                                worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Update worker batch information when a worker completes processing a request.
        
        Args:
            worker_id: ID of the worker that completed the request
            completed_req_id: ID of the completed request
            worker_batch_info: Current batch information for each worker
            
        Returns:
            Updated worker batch information
        """
        if worker_id not in worker_batch_info:
            print(f"Warning: Worker {worker_id} not found in batch info")
            return worker_batch_info
        
        # Find and remove the completed request from the worker's batch list
        worker_batches = worker_batch_info[worker_id]
        updated_batches = [
            batch for batch in worker_batches 
            if batch.get('request_id') != completed_req_id
        ]
        
        # Update the worker's batch info
        worker_batch_info[worker_id] = updated_batches
        
        return worker_batch_info
    
    def recalculate_worker_load(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Recalculate the load metrics for workers based on latest step information.
        This is called when step updates are received from workers.
        
        Args:
            worker_batch_info: Updated batch information for each worker
        """
        # Base implementation does nothing - subclasses can implement specific metrics
        worker_stats = {}
        
        # Calculate some basic statistics for logging
        for worker_id, batches in worker_batch_info.items():
            total_remaining_steps = sum(
                batch.get('num_inference_steps', 0) - batch.get('scheduler_steps', 0) 
                for batch in batches
            )
            
            total_seq_length = sum(batch.get('mask_seq_length', 0) for batch in batches)
            
            worker_stats[worker_id] = {
                'batch_count': len(batches),
                'total_seq_length': total_seq_length,
                'total_remaining_steps': total_remaining_steps
            }
        
        # Log the recalculated statistics
        if worker_stats:
            print("\n--- Recalculated Worker Load After Step Update ---")
            for worker_id, stats in worker_stats.items():
                print(f"Worker {worker_id}: {stats['batch_count']} batches, "
                      f"remaining steps: {stats['total_remaining_steps']}, "
                      f"seq length: {stats['total_seq_length']}")
            print("--------------------------------------------------\n")
    
    # def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
    #     """
    #     Log statistics about the current load distribution across workers.
        
    #     Args:
    #         worker_batch_info: Current batch information for each worker
    #     """
    #     # Default implementation for logging statistics
    #     pass


class SeqLengthBalanceScheduler(Scheduler):
    """
    Scheduler that balances requests based on sequence length.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name: str) -> str:
        """
        Generate additional log information for a request.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with sequence length information
        """
        return ""  # No additional info needed as seq_length is already logged
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     model_type: str,
                     req_id: str) -> str:
        """
        Select a worker based on sequence length balance.
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID with the lowest total sequence length
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Extract sequence length from request inputs
        request_seq_length = get_seq_length_from_request(request_inputs)
        
        # Calculate total sequence length for each idle worker
        worker_total_seq_lengths = {}
        for worker_id in idle_workers:
            worker_batches = worker_batch_info.get(worker_id, [])
            total_seq_length = sum(batch.get('mask_seq_length', 0) for batch in worker_batches)
            worker_total_seq_lengths[worker_id] = total_seq_length
        
        # Find the worker with the lowest total sequence length
        min_seq_length = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id, total_seq_length in worker_total_seq_lengths.items():
            # Calculate what the new total would be if we assign to this worker
            new_total = total_seq_length + request_seq_length
            if new_total < min_seq_length:
                min_seq_length = new_total
                selected_worker = worker_id
        
        return selected_worker
    
    # def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
    #     """
    #     Log statistics about the current load distribution across workers.
        
    #     Args:
    #         worker_batch_info: Current batch information for each worker
    #     """
    #     if not worker_batch_info:
    #         print("No workers with active batches.")
    #         return
        
    #     print("\n----- Worker Load Distribution -----")
    #     total_loads = {}
    #     for worker_id, batches in worker_batch_info.items():
    #         total_seq_length = sum(batch.get('mask_seq_length', 0) for batch in batches)
    #         batch_count = len(batches)
    #         total_loads[worker_id] = {
    #             'total_seq_length': total_seq_length,
    #             'batch_count': batch_count
    #         }
    #         print(f"Worker {worker_id}: {batch_count} batches, total seq length: {total_seq_length}")
        
    #     # Calculate load balance metrics
    #     seq_lengths = [info['total_seq_length'] for info in total_loads.values()]
    #     if seq_lengths:
    #         avg_seq_length = sum(seq_lengths) / len(seq_lengths)
    #         max_seq_length = max(seq_lengths) if seq_lengths else 0
    #         min_seq_length = min(seq_lengths) if seq_lengths else 0
    #         imbalance_ratio = max_seq_length / avg_seq_length if avg_seq_length > 0 else 0
            
    #         print(f"\nLoad Balance Statistics:")
    #         print(f"  Average sequence length per worker: {avg_seq_length:.2f}")
    #         print(f"  Min-Max sequence length: {min_seq_length} - {max_seq_length}")
    #         print(f"  Imbalance ratio (max/avg): {imbalance_ratio:.2f}")
    #     print("------------------------------------\n")


class BatchSizeBalanceScheduler(Scheduler):
    """
    Scheduler that balances requests based on the number of batches (batch size) on each worker,
    without considering sequence length.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name: str) -> str:
        """
        Generate additional log information for a request.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with additional log information
        """
        return ""  # Batch size scheduler doesn't need additional logging
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     pipeline_name: str,
                     req_id: str) -> str:
        """
        Select a worker with the fewest number of batches.
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID with the smallest batch count
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Find the worker with the fewest batches
        min_batch_count = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id in idle_workers:
            batch_count = len(worker_batch_info.get(worker_id, []))
            if batch_count < min_batch_count:
                min_batch_count = batch_count
                selected_worker = worker_id
        
        return selected_worker
    
    def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Log statistics about the current batch distribution across workers.
        
        Args:
            worker_batch_info: Current batch information for each worker
        """
        if not worker_batch_info:
            print("No workers with active batches.")
            return
        
        print("\n----- Worker Batch Distribution -----")
        batch_counts = {}
        for worker_id, batches in worker_batch_info.items():
            batch_count = len(batches)
            batch_counts[worker_id] = batch_count
            print(f"Worker {worker_id}: {batch_count} batches")
        
        # Calculate batch balance metrics
        counts = list(batch_counts.values())
        if counts:
            avg_count = sum(counts) / len(counts)
            max_count = max(counts) if counts else 0
            min_count = min(counts) if counts else 0
            imbalance_ratio = max_count / avg_count if avg_count > 0 else 0
            
            print(f"\nBatch Balance Statistics:")
            print(f"  Average batches per worker: {avg_count:.2f}")
            print(f"  Min-Max batches: {min_count} - {max_count}")
            print(f"  Imbalance ratio (max/avg): {imbalance_ratio:.2f}")
        print("------------------------------------\n")


# For backward compatibility (these will be deprecated)
def select_worker_by_seq_length_balance(
    idle_workers: List[str], 
    worker_batch_info: Dict[str, List[Dict[str, Any]]], 
    request_seq_length: int
) -> str:
    """
    Legacy function for backward compatibility.
    Use SeqLengthBalanceScheduler().select_worker() instead.
    """
    scheduler = SeqLengthBalanceScheduler()
    # Create dummy request inputs with the sequence length
    request_inputs = {'mask_seq_length': request_seq_length}
    return scheduler.select_worker(idle_workers, worker_batch_info, request_inputs, "dummy_req_id")


def update_worker_batch_info_on_completion(
    worker_id: str,
    completed_request_id: str,
    worker_batch_info: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Legacy function for backward compatibility.
    Use Scheduler().remove_completed_request() instead.
    """
    scheduler = Scheduler()
    return scheduler.remove_completed_request(worker_id, completed_request_id, worker_batch_info)


def log_worker_load_statistics(worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Legacy function for backward compatibility.
    Use SeqLengthBalanceScheduler().log_statistics() instead.
    """
    scheduler = SeqLengthBalanceScheduler()
    scheduler.log_statistics(worker_batch_info)
def cal_flops(request_length, pipeline_name):
    if pipeline_name == "Flux_inpaint":
        return cal_flops_flux(request_length)
    elif pipeline_name == "OOTD_HD":
        return cal_flops_ootd(request_length)
def cal_flops_flux(request_length):
    standard_len = 4608
    h = 3072

    def cal_attn_flops(request_length, standard_len, b, h):
        # b is the batch size, h is the hidden size
        # Q [b, request_length, h]
        # K[b, standard_len, h]
        # V[b, standard_len, h]
        # attn_score = QK^T / sqrt(h)
        # attn_prob = softmax(attn_score)
        # attn_output = attn_prob @ V
        request_length = request_length + 512
        attn_flops = (
            2 * b * (request_length) * standard_len * h
            + 2 * b * request_length * standard_len * h
            + 2 * b * request_length * h * h
        )
        to_q_flops = 2 * b * request_length * h * h
        to_k_flops = 2 * b * standard_len * h * h
        to_v_flops = 2 * b * standard_len * h * h
        # text flops
        to_out_flops = 2 * b * request_length * h * h
        return attn_flops + to_q_flops + to_k_flops + to_v_flops + to_out_flops

    def cal_linear_flops(s, b, h, h1):
        # mlp_flops = 4*b*request_length*h*4*h
        # return mlp_flops
        linear_flops = 2 * b * s * h * h1
        return linear_flops

    def cal_mmdit_flops(request_length, standard_len, b, h):
        attn_flops = cal_attn_flops(request_length, standard_len, b, h)
        mlp_flops = 2 * cal_linear_flops(request_length + 512, b, h, 4 * h)
        return attn_flops + mlp_flops

    def cal_mmdit_flops_reduced(request_length, standard_len, h):
        a = 4 * standard_len * h + 22 * h * h
        b = 512 * (4 * standard_len * h + 22 * h * h) + 4 * standard_len * h * h
        return a * request_length + b

    def cal_single_flops(request_length, standard_len, b, h):
        linear_flops = cal_linear_flops(request_length + 512, b, h, 4 * h)
        attn_flops = cal_attn_flops(request_length, standard_len, b, h)
        linear_flops1 = cal_linear_flops(request_length + 512, b, 5 * h, h)
        return attn_flops + linear_flops + linear_flops1

    # test = cal_mmdit_flops_reduced(request_length, standard_len,h)
    mmdit_flops = cal_mmdit_flops(request_length, standard_len, 1, h)
    # return mmdit_flops
    single_flops = cal_single_flops(request_length, standard_len, 1, h)
    total_flops = mmdit_flops * 19 + single_flops * 38
    return total_flops
def cal_flops_ootd(request_length):
    def cal_self_attn_flops(request_length, standard_len, b, h):
        request_length = request_length
        attn_flops = (
            2 * b * (request_length) * standard_len * h
            + 2 * b * request_length * standard_len * h
            + 2 * b * request_length * h * h
        )
        to_q_flops = 2 * b * request_length * h * h
        to_k_flops = 2 * b * standard_len * h * h
        to_v_flops = 2 * b * standard_len * h * h
        # text flops
        to_out_flops = 2 * b * request_length * h * h
        return attn_flops + to_q_flops + to_k_flops + to_v_flops + to_out_flops
    def cal_cross_attn_flops(hidden_len, encoder_len, b, hidden_h,encoder_h):
        to_q_flops = 2 * b * hidden_len * hidden_h * hidden_h
        # (b, encoder_len , e_h) (eh, hh)
        to_k_flops = 2 * b * encoder_len * encoder_h * hidden_h
        to_v_flops = 2 * b * encoder_len * encoder_h * hidden_h
        # text flops
        # q*k (b, hidden_len, hh) (b, hh, encoder_len) --> b, hidden_len, encoder_len
        # *v (b, hidden_len, encoder_len) (b, encoder_len, hidden_h) --> b, hidden_len, hidden_h
        # *out (b, hidden_len, hidden_h) (hidden_h, hidden_h) --> b, hidden_len, hidden_h
        attn_flops = (
            2 * b * (hidden_len) * hidden_h * encoder_h
            + 2 * b * hidden_len * encoder_len * hidden_h
            + 2 * b * hidden_len * hidden_h * hidden_h
        )
        to_out_flops = 2 * b * hidden_len * encoder_h * encoder_h
        return to_q_flops + to_k_flops + to_v_flops + to_out_flops + attn_flops

    def cal_linear_flops(s, b, h, h1):
        # mlp_flops = 4*b*request_length*h*4*h
        # return mlp_flops
        linear_flops = 2 * b * s * h * h1
        return linear_flops
    self_attn_shape = [(24576, 320),(24576, 320),[6144, 640],[6144, 640],[1536, 1280],[1536, 1280],[384, 1280],[1536, 1280],[1536, 1280],[6144, 640],[6144, 640],(24576, 320),(24576, 320)]
    cross_attn_shape = [(12288, 320),(12288, 320),[3077, 640],[3077, 640],[1536//2, 1280],[1536//2, 1280],[384//2, 1280],[1536//2, 1280],[1536//2, 1280],[3077, 640],[3077, 640],(12288, 320),(12288, 320)]
    cross_attn_shape_encoder = (2, 768)
    total_flops = 0
    for i in range(len(self_attn_shape)):
        masked_len = request_length // (24576//self_attn_shape[i][0] )
        self_attn_flops = cal_self_attn_flops(masked_len, self_attn_shape[i][0], 1, self_attn_shape[i][1])
  
        cross_attn_flops = cal_cross_attn_flops(cross_attn_shape[i][0], cross_attn_shape_encoder[0], 1, cross_attn_shape[i][1], cross_attn_shape_encoder[1])
        ff_flops = cal_linear_flops(cross_attn_shape[i][0], 1, cross_attn_shape[i][1], 4 * cross_attn_shape[i][1])
        flops = self_attn_flops + cross_attn_flops + ff_flops
        total_flops += flops
    return total_flops
class  NewFlopsBalanceScheduler(Scheduler):
    """
    Scheduler that balances requests based on estimated FLOPS (computational load).
    Uses cal_flops_flux function to calculate the computational requirements of each request.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name: str) -> str:
        """
        Generate additional log information for a request, including FLOPS estimates.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with FLOPS information
        """
        if pipeline_name == "Flux_inpaint":
            request_flops = cal_flops_flux(request_seq_length)
        elif pipeline_name == "OOTD_HD":
            request_flops = cal_flops_ootd(request_seq_length)
        flops_in_gigaflops = request_flops / 1e9
        return f", FLOPS: {flops_in_gigaflops:.2f}G"
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     model_type: str,
                     req_id: str) -> str:
        """
        Select a worker with the lowest total computational load (FLOPS).
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID with the lowest total FLOPS
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Extract sequence length from request inputs
        request_seq_length = get_seq_length_from_request(request_inputs)
        
        # Calculate FLOPS for the new request
        if model_type == "OOTD_HD":
            request_flops = cal_flops_ootd(request_seq_length)
        elif model_type == "Flux_inpaint":
            request_flops = cal_flops_flux(request_seq_length)
        # select the worker with min batch size
        min_batch_size = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        for worker_id in idle_workers:
            batch_size = len(worker_batch_info.get(worker_id, []))
            if batch_size < min_batch_size:
                min_batch_size = batch_size
                selected_worker = worker_id
        # if min_batch_size is less than other batch_size by more than 2, return the selected_worker
        for worker_id in idle_workers:
            if worker_id != selected_worker:
                if len(worker_batch_info.get(worker_id, [])) - min_batch_size >= 2:
                    return selected_worker
        # Calculate total FLOPS for each idle worker
        worker_total_flops = {}
        for worker_id in idle_workers:
            worker_batches = worker_batch_info.get(worker_id, [])
            if model_type == "OOTD_HD":
                total_flops = sum(cal_flops_ootd(batch.get('mask_seq_length', 0)) for batch in worker_batches)
            elif model_type == "Flux_inpaint":
                total_flops = sum(cal_flops_flux(batch.get('mask_seq_length', 0)) for batch in worker_batches)
            worker_total_flops[worker_id] = total_flops
        
        # Find the worker with the lowest total FLOPS
        min_flops = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id, total_flops in worker_total_flops.items():
            # Calculate what the new total would be if we assign to this worker
            new_total = total_flops + request_flops
            if new_total < min_flops:
                min_flops = new_total
                selected_worker = worker_id
        
        return selected_worker
class FlopsBalanceScheduler(Scheduler):
    """
    Scheduler that balances requests based on estimated FLOPS (computational load).
    Uses cal_flops_flux function to calculate the computational requirements of each request.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name: str) -> str:
        """
        Generate additional log information for a request, including FLOPS estimates.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with FLOPS information
        """
        if pipeline_name == "OOTD_HD":
            request_flops = cal_flops_ootd(request_seq_length)
        elif pipeline_name == "Flux_inpaint":
            request_flops = cal_flops_flux(request_seq_length)
        flops_in_gigaflops = request_flops / 1e9
        return f", FLOPS: {flops_in_gigaflops:.2f}G"
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     model_type: str,
                     req_id: str) -> str:
        """
        Select a worker with the lowest total computational load (FLOPS).
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID with the lowest total FLOPS
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Extract sequence length from request inputs
        request_seq_length = get_seq_length_from_request(request_inputs)
        
        # Calculate FLOPS for the new request
        if model_type == "ootd":
            request_flops = cal_flops_ootd(request_seq_length)
        else:
            request_flops = cal_flops_flux(request_seq_length)
        
        # Calculate total FLOPS for each idle worker
        worker_total_flops = {}
        for worker_id in idle_workers:
            worker_batches = worker_batch_info.get(worker_id, [])
            if model_type == "ootd":
                total_flops = sum(cal_flops_ootd(batch.get('mask_seq_length', 0)) for batch in worker_batches)
            else:
                total_flops = sum(cal_flops_flux(batch.get('mask_seq_length', 0)) for batch in worker_batches)
            worker_total_flops[worker_id] = total_flops
        
        # Find the worker with the lowest total FLOPS
        min_flops = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id, total_flops in worker_total_flops.items():
            # Calculate what the new total would be if we assign to this worker
            new_total = total_flops + request_flops
            if new_total < min_flops:
                min_flops = new_total
                selected_worker = worker_id
        
        return selected_worker
    
    def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]], model_type: str) -> None:
        """
        Log statistics about the current computational load distribution across workers.
        
        Args:
            worker_batch_info: Current batch information for each worker
        """
        if not worker_batch_info:
            print("No workers with active batches.")
            return
        
        print("\n----- Worker Computational Load Distribution -----")
        total_loads = {}
        for worker_id, batches in worker_batch_info.items():
            batch_count = len(batches)
            # Calculate total FLOPS for each worker
            if model_type == "ootd":
                total_flops = sum(cal_flops_ootd(batch.get('mask_seq_length', 0)) for batch in batches)
            else:
                total_flops = sum(cal_flops_flux(batch.get('mask_seq_length', 0)) for batch in batches)
            flops_in_gigaflops = total_flops / 1e9  # Convert to GFLOPS for readability
            
            total_loads[worker_id] = {
                'total_flops': total_flops,
                'batch_count': batch_count
            }
            print(f"Worker {worker_id}: {batch_count} batches, total compute: {flops_in_gigaflops:.2f} GFLOPS")
        
        # Calculate load balance metrics
        flops_values = [info['total_flops'] for info in total_loads.values()]
        if flops_values:
            avg_flops = sum(flops_values) / len(flops_values)
            max_flops = max(flops_values) if flops_values else 0
            min_flops = min(flops_values) if flops_values else 0
            imbalance_ratio = max_flops / avg_flops if avg_flops > 0 else 0
            
            avg_gflops = avg_flops / 1e9
            max_gflops = max_flops / 1e9
            min_gflops = min_flops / 1e9
            
            print(f"\nComputational Load Balance Statistics:")
            print(f"  Average compute per worker: {avg_gflops:.2f} GFLOPS")
            print(f"  Min-Max compute: {min_gflops:.2f} - {max_gflops:.2f} GFLOPS")
            print(f"  Imbalance ratio (max/avg): {imbalance_ratio:.2f}")
        print("------------------------------------\n")
    def get_flops(self, seqlen, model_type: str):
        if model_type == "ootd":
            return cal_flops_ootd(seqlen)/1e9
        else:
            return cal_flops_flux(seqlen)/1e9
# Factory function to create scheduler based on name
def create_scheduler(scheduler_name: str) -> Scheduler:
    """
    Create a scheduler instance based on the provided name.
    
    Args:
        scheduler_name: Name of the scheduler strategy
        
    Returns:
        An instance of the requested scheduler
    """
    if scheduler_name == "seq_length_balance":
        return SeqLengthBalanceScheduler()
    elif scheduler_name == "batch_size_balance":
        return BatchSizeBalanceScheduler()
    elif scheduler_name == "flops_balance":
        return FlopsBalanceScheduler()
    elif scheduler_name == "new_flops_balance":
        return NewFlopsBalanceScheduler()
    elif scheduler_name == "flops_batch_balance":
        return FlopsBatchBalanceScheduler()
    elif scheduler_name =="step_flops_batch_balance":
        return StepFlpsBatchBalanceScheduler()
    # Add more scheduler implementations as needed
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def schedule_request_balanced_seq_length(
    request_id: str,
    request_inputs: Dict[str, Any],
    idle_workers: List[str],
    worker_batch_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Schedule a request to an idle worker considering the sequence length balance.
    
    This function selects the worker with the lowest total sequence length workload
    to ensure balanced distribution of computational load across workers.
    
    Args:
        request_id: Unique identifier for the request
        request_inputs: The inputs for the request
        idle_workers: List of available worker IDs
        worker_batch_info: Current batch information for each worker
        
    Returns:
        Tuple of (selected_worker_id, updated_worker_batch_info)
    """
    # Extract sequence length from request inputs
    request_seq_length = get_seq_length_from_request(request_inputs)
    
    # Select worker based on sequence length balance
    selected_worker = select_worker_by_seq_length_balance(
        idle_workers, worker_batch_info, request_seq_length
    )
    
    # Update worker batch info with the new request
    if selected_worker not in worker_batch_info:
        worker_batch_info[selected_worker] = []
    
    # Add this request to the selected worker's batch info
    worker_batch_info[selected_worker].append({
        'request_id': request_id,
        'mask_seq_length': request_seq_length,
        'num_inference_steps': request_inputs.get('num_inference_steps', 28),  # Default to 28 if not specified
        'inputs': request_inputs,
    })
    
    return selected_worker, worker_batch_info

class FlopsBatchBalanceScheduler(Scheduler):
    """
    Scheduler that balances requests based on estimated FLOPS (computational load).
    Uses cal_flops_flux function to calculate the computational requirements of each request.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name: str) -> str:
        """
        Generate additional log information for a request, including real-time and compute estimates.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with real-time and compute information
        """
        # Calculate FLOPS for reference
        request_flops = cal_flops_flux(request_seq_length)
        flops_in_gigaflops = request_flops / 1e9
        
        # Calculate estimated compute time
        compute_time = cal_compute_time(request_seq_length)
        
        # Return both metrics for comprehensive logging
        return f", FLOPS: {flops_in_gigaflops:.2f}G, Est. compute time: {compute_time:.2f}ms"
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     pipeline_name: str,
                     req_id: str) -> str:
        """
        Select a worker with the most balanced real time after adding the new request.
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID that will result in the most balanced real time distribution
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Extract sequence length from request inputs
        request_seq_length = get_seq_length_from_request(request_inputs)
        
        # Calculate current real time for each idle worker before adding new request
        worker_current_real_time = {}
        for worker_id in idle_workers:
            worker_batches = worker_batch_info.get(worker_id, [])
            if not worker_batches:
                # No existing batches, so real time is 0
                worker_current_real_time[worker_id] = 0
                continue
                
            # Get sequence lengths of all batches for this worker
            seq_lengths = [batch.get('mask_seq_length', 0) for batch in worker_batches]
            batch_size = len(seq_lengths)
            
            # Calculate real time using the cal_real_time function
            worker_current_real_time[worker_id] = cal_real_time(seq_lengths, batch_size)
        
        # Find the worker that would result in the most balanced real time distribution after adding the new request
        min_max_real_time = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id in idle_workers:
            # Get the current worker's batches
            worker_batches = worker_batch_info.get(worker_id, [])
            
            # Create a new sequence length list including the new request
            new_seq_lengths = [batch.get('mask_seq_length', 0) for batch in worker_batches] + [request_seq_length]
            new_batch_size = len(new_seq_lengths)
            
            # Calculate what the real time would be after adding the new request
            new_real_time = cal_real_time(new_seq_lengths, new_batch_size)
            
            # Compute the maximum real time across all workers if we assign to this worker
            max_real_time = new_real_time
            for other_worker_id in idle_workers:
                if other_worker_id != worker_id:
                    max_real_time = max(max_real_time, worker_current_real_time[other_worker_id])
            
            # Choose the worker that minimizes the maximum real time
            if max_real_time < min_max_real_time:
                min_max_real_time = max_real_time
                selected_worker = worker_id
                
        return selected_worker
    
    def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Log statistics about the current real-time load distribution across workers.
        
        Args:
            worker_batch_info: Current batch information for each worker
        """
        if not worker_batch_info:
            print("No workers with active batches.")
            return
        
        print("\n----- Worker Real-Time Load Distribution -----")
        worker_real_times = {}
        
        for worker_id, batches in worker_batch_info.items():
            batch_count = len(batches)
            
            if batch_count == 0:
                worker_real_times[worker_id] = 0
                print(f"Worker {worker_id}: 0 batches, real-time: 0.00 ms")
                continue
            
            # Extract sequence lengths from batches
            seq_lengths = [batch.get('mask_seq_length', 0) for batch in batches]
            
            # Calculate real time for this worker
            real_time = cal_real_time(seq_lengths, batch_count)
            worker_real_times[worker_id] = real_time
            
            print(f"Worker {worker_id}: {batch_count} batches, real-time: {real_time:.2f} ms")
        
        # Calculate load balance metrics for real-time
        real_time_values = list(worker_real_times.values())
        if real_time_values:
            avg_real_time = sum(real_time_values) / len(real_time_values)
            max_real_time = max(real_time_values) if real_time_values else 0
            min_real_time = min(real_time_values) if real_time_values else 0
            
            # Only calculate imbalance ratio if average is greater than zero to avoid division by zero
            imbalance_ratio = max_real_time / avg_real_time if avg_real_time > 0 else 0
            
            print(f"\nReal-Time Load Balance Statistics:")
            print(f"  Average real-time per worker: {avg_real_time:.2f} ms")
            print(f"  Min-Max real-time: {min_real_time:.2f} - {max_real_time:.2f} ms")
            print(f"  Imbalance ratio (max/avg): {imbalance_ratio:.2f}")
        
        # Also show compute time and load time breakdown
        print("\nBreakdown by Worker:")
        for worker_id, batches in worker_batch_info.items():
            if not batches:
                continue
                
            batch_count = len(batches)
            seq_lengths = [batch.get('mask_seq_length', 0) for batch in batches]
            
            # Calculate compute time for each sequence
            compute_times = [cal_compute_time(seq_len) for seq_len in seq_lengths]
            total_compute_time = sum(compute_times)
            
            # Get load time based on batch size
            load_time = cal_load_time(batch_count) if batch_count <= 8 else "N/A (batch size > 8)"
            if isinstance(load_time, (int, float)):
                time_bottleneck = "Compute" if total_compute_time > load_time else "Load"
            else:
                time_bottleneck = "Unknown"
                
            print(f"  Worker {worker_id}: Compute time: {total_compute_time:.2f} ms, Load time: {load_time}, Bottleneck: {time_bottleneck}")
        
        print("------------------------------------\n")
    def get_flops(self, seqlen):
        return cal_flops_flux(seqlen)/1e9
    
class StepFlpsBatchBalanceScheduler(Scheduler):

    """
    Scheduler that balances requests based on estimated FLOPS (computational load).
    Uses cal_flops_flux function to calculate the computational requirements of each request.
    """
    def get_request_log_info(self, request_seq_length: int, pipeline_name) -> str:
        """
        Generate additional log information for a request, including real-time and compute estimates.
        
        Args:
            request_seq_length: The sequence length of the request
            
        Returns:
            A string with real-time and compute information
        """
        # Calculate FLOPS for reference
        request_flops = cal_flops_flux(request_seq_length)
        flops_in_gigaflops = request_flops / 1e9
        
        # Calculate estimated compute time
        compute_time = cal_compute_time(request_seq_length)
        
        # Return both metrics for comprehensive logging
        return f", FLOPS: {flops_in_gigaflops:.2f}G, Est. compute time: {compute_time:.2f}ms"
    
    def select_worker(self, 
                     idle_workers: List[str],
                     worker_batch_info: Dict[str, List[Dict[str, Any]]],
                     request_inputs: Dict[str, Any],
                     pipeline_name,
                     req_id: str) -> str:
        """
        Select a worker with the most balanced real time after adding the new request.
        
        Args:
            idle_workers: List of available worker IDs
            worker_batch_info: Dictionary of current batch information for each worker
            request_inputs: The request inputs
            req_id: The request ID
            
        Returns:
            The worker ID that will result in the most balanced real time distribution
        """
        if not idle_workers:
            raise ValueError("No idle workers available for scheduling")
        
        # Extract sequence length from request inputs
        request_seq_length = get_seq_length_from_request(request_inputs)
        
        # Calculate current real time for each idle worker before adding new request
        worker_current_real_time = {}
        for worker_id in idle_workers:
            worker_batches = worker_batch_info.get(worker_id, [])
            if not worker_batches:
                # No existing batches, so real time is 0
                worker_current_real_time[worker_id] = 0
                continue
                
            # Get sequence lengths of all batches for this worker
            remaining_steps = [batch.get('remaining_steps', 0) for batch in worker_batches]
            print("remaining_steps", remaining_steps)
            seq_lengths = [batch.get('mask_seq_length', 0) for batch in worker_batches]
            batch_size = len(seq_lengths)
            
            # Calculate real time using the cal_real_time function
            worker_current_real_time[worker_id] = cal_real_time_steps(seq_lengths, batch_size, remaining_steps)
        
        # Find the worker that would result in the most balanced real time distribution after adding the new request
        min_max_real_time = float('inf')
        selected_worker = idle_workers[0]  # Default to first worker
        
        for worker_id in idle_workers:
            # Get the current worker's batches
            worker_batches = worker_batch_info.get(worker_id, [])
            
            # Create a new sequence length list including the new request
            new_seq_lengths = [batch.get('mask_seq_length', 0) for batch in worker_batches] + [request_seq_length]
            remaining_steps = [batch.get('remaining_steps', 0) for batch in worker_batches]
            new_remaining_steps = remaining_steps + [request_inputs.get('num_inference_steps', 20)]
            new_batch_size = len(new_seq_lengths)
            
            # Calculate what the real time would be after adding the new request
            new_real_time = cal_real_time_steps(new_seq_lengths, new_batch_size, new_remaining_steps)
            
            # Compute the maximum real time across all workers if we assign to this worker
            max_real_time = new_real_time
            for other_worker_id in idle_workers:
                if other_worker_id != worker_id:
                    max_real_time = max(max_real_time, worker_current_real_time[other_worker_id])
            
            # Choose the worker that minimizes the maximum real time
            if max_real_time < min_max_real_time:
                min_max_real_time = max_real_time
                selected_worker = worker_id
                
        return selected_worker
    
    def log_statistics(self, worker_batch_info: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Log statistics about the current real-time load distribution across workers.
        
        Args:
            worker_batch_info: Current batch information for each worker
        """
        if not worker_batch_info:
            print("No workers with active batches.")
            return
        
        print("\n----- Worker Real-Time Load Distribution (With Step-Aware Calculation) -----")
        worker_real_times = {}
        
        for worker_id, batches in worker_batch_info.items():
            batch_count = len(batches)
            
            if batch_count == 0:
                worker_real_times[worker_id] = 0
                print(f"Worker {worker_id}: 0 batches, real-time: 0.00 ms")
                continue
            
            # Extract sequence lengths from batches
            seq_lengths = [batch.get('mask_seq_length', 0) for batch in batches]
            remaining_steps = [batch.get('remaining_steps', batch.get('num_inference_steps', 20)) for batch in batches]
            
            # Calculate real time for this worker using updated step-aware function
            real_time = cal_real_time_steps(seq_lengths, batch_count, remaining_steps)
            worker_real_times[worker_id] = real_time
            
            # 计算步数完成百分比
        
        # Calculate load balance metrics for real-time
        real_time_values = list(worker_real_times.values())
        if real_time_values:
            avg_real_time = sum(real_time_values) / len(real_time_values)
            max_real_time = max(real_time_values) if real_time_values else 0
            min_real_time = min(real_time_values) if real_time_values else 0
            
            # Only calculate imbalance ratio if average is greater than zero to avoid division by zero
            imbalance_ratio = max_real_time / avg_real_time if avg_real_time > 0 else 0
            
            print(f"\nReal-Time Load Balance Statistics:")
            print(f"  Average real-time per worker: {avg_real_time:.2f} ms")
            print(f"  Min-Max real-time: {min_real_time:.2f} - {max_real_time:.2f} ms")
            print(f"  Imbalance ratio (max/avg): {imbalance_ratio:.2f}")
        
        # 展示每个worker上不同请求的详细信息
        print("\nDetailed Request Info by Worker:")
        for worker_id, batches in worker_batch_info.items():
            if not batches:
                continue
            
            print(f"  Worker {worker_id} requests:")
            for i, batch in enumerate(batches):
                req_id = batch.get('request_id', batch.get('req_id', f'unknown-{i}'))
                seq_len = batch.get('mask_seq_length', 0)
                total_steps = batch.get('num_inference_steps', 20)
                remaining = batch.get('remaining_steps', total_steps)
            
                
                # print(f"    - Req {req_id}: {progress:.1f}% complete, "
                #       f"steps: {total_steps-remaining}/{total_steps}, "
                #       f"seq_len: {seq_len}")
        
        print("------------------------------------\n")
    def get_flops(self, seqlen):
        return cal_flops_flux(seqlen)/1e9
