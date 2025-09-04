import numpy as np
import os
import json

def genereate_seqlen(n, data):
    # 从*.npy文件中读取n个数据
    # 确保n不超过数据长度
    standard_len = 12288
    n = min(n, len(data))
    # 随机选择n个索引
    indices = np.random.choice(len(data), n, replace=False)
    # 返回随机选择的n个值作为列表
    mask_list = data[indices].tolist()
    seqlen_list = [int(standard_len * mask) for mask in mask_list]
    return seqlen_list

def save_seqlen_to_file(seqlen_list, n, test_id, output_dir="seqlen_data_ootd"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 创建文件名
    filename = os.path.join(output_dir, f"spring_seqlen_n{n}_test{test_id}.json")
    # 将数据保存为JSON文件
    with open(filename, "w") as f:
        json.dump(seqlen_list, f)
    return filename
def load_all_seqlen_files_experiment(n=400,base_dir='/home/xjiangbp/image-inpainting/scheduler/seqlen_data_flux_experiment'):
    file_name = f"spring_seqlen_n{n}_test0.json"
    file_path = os.path.join(base_dir, file_name)
    results = []
    with open(file_path, "r") as f:
        seqlen_list = json.load(f)
    results.append(seqlen_list)
    return results
def load_all_seqlen_files(base_dir="/home/xjiangbp/image-inpainting/scheduler/seqlen_data"):
    """
    遍历指定目录中的所有seqlen文件并读取它们的内容
    
    Args:
        base_dir: 存储seqlen文件的基础目录
        
    Returns:
        字典，格式为 {文件名: seqlen列表}
    """
    if not os.path.exists(base_dir):
        print(f"目录 {base_dir} 不存在")
        return {}
    
    results = []
    for filename in os.listdir(base_dir):
        if filename.startswith("katz_seqlen_") and filename.endswith(".json"):
            file_path = os.path.join(base_dir, filename)
            try:
                with open(file_path, "r") as f:
                    seqlen_list = json.load(f)
                results.append(seqlen_list)
                # print(f"成功加载文件 {filename}: {seqlen_list}")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    return results
def load_all_seqlen_files_ootd_experiment():
    pass
def load_all_seqlen_files_ootd(base_dir="/home/xjiangbp/image-inpainting/scheduler/seqlen_data_ootd"):
    """
    遍历指定目录中的所有seqlen文件并读取它们的内容
    
    Args:
        base_dir: 存储seqlen文件的基础目录
        
    Returns:
        字典，格式为 {文件名: seqlen列表}
    """
    if not os.path.exists(base_dir):
        print(f"目录 {base_dir} 不存在")
        return {}
    
    results = []
    for filename in os.listdir(base_dir):
        if filename.startswith("spring_seqlen_") and filename.endswith(".json"):
            file_path = os.path.join(base_dir, filename)
            try:
                with open(file_path, "r") as f:
                    seqlen_list = json.load(f)
                results.append(seqlen_list)
                # print(f"成功加载文件 {filename}: {seqlen_list}")
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")
    
    return results

if __name__ == "__main__":
    data1 = np.load("/home/xjiangbp/image-inpainting/datas/inpaint_mask_ratios_1001.npy")
    data2 = np.load("/home/xjiangbp/image-inpainting/datas/controlnet_inpaint_mask_ratios_1918.npy")
    data3 = np.load("/home/xjiangbp/image-inpainting/datas/spring_festival_mask_ratio_original.npy")
    data4 = np.load("/home/xjiangbp/image-inpainting/datas/adetailor_trace.npy")

    # cat data and data1
    data = np.concatenate([data1, data2, data3, data4], axis=0)
    n_list = [400]
    test_time = 1
    output_dir = "seqlen_data_flux_experiment"
    
    for n in n_list:
        for i in range(test_time):
            seqlen_list = genereate_seqlen(n, data)
            # 保存到文件
            filename = save_seqlen_to_file(seqlen_list, n, i, output_dir)
            print(f"Saved seqlen to {filename}: {seqlen_list}")
    # results = load_all_seqlen_files_ootd()
    # print(results)
    
    # 示例：如何加载所有保存的seqlen文件
    # all_seqlen_data = load_all_seqlen_files(output_dir)
    # print(f"加载了 {len(all_seqlen_data)} 个seqlen文件")
