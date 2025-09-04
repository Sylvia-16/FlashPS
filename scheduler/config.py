from dataclasses import dataclass
from typing import List


@dataclass
class NodeConfig:
    rank: int
    address: str
    gpu_count: int


@dataclass
class DistributedConfig:
    nodes: List[NodeConfig]
    port: int = 29500

    @property
    def world_size(self) -> int:
        return sum(node.gpu_count for node in self.nodes)

    @property
    def master_addr(self) -> str:
        """Get master node (rank 0) address"""
        return self.get_node_by_rank(0).address

    def get_node_by_rank(self, rank: int) -> NodeConfig:
        for node in self.nodes:
            if node.rank == rank:
                return node
        raise ValueError(f"No node found with rank {rank}")

@dataclass
class CacheConfig:
    cached_kv_folder: str = ""
    cached_latents_folder: str = ""
    cached_o_folder: str = ""
