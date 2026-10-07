import time
import numpy as np
import torch.distributed as dist
from torch.utils.data import Sampler

class BucketSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.

    Args:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, cfg, spec_header,
                 shuffle=True, drop_last=True) -> Sampler:
        super().__init__(data_source=None)
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.spec_header = spec_header
        self.epoch = 0
        self.step = 0

        self.node_size_bucket = [0]+sorted(self.cfg.data.node_size_bucket)
        self.node_size = (self.spec_header['Peaks Count']*2+3).to_numpy()
        index_list = np.arange(len(self.spec_header))
        self.buckets_ori = [index_list[np.logical_and(self.node_size>self.node_size_bucket[i],self.node_size<=self.node_size_bucket[i+1])] for i in range(len(self.node_size_bucket)-1)]

        
    def __iter__(self):
        self.rng = np.random.default_rng(seed=self.epoch)
        self.generate_bins()
        self.readed_index = np.zeros(len(self.buckets),dtype=int)
        self.buckets_len = np.array([len(bucket) for bucket in self.buckets])
        for _ in range(self.step): self.__next__()
        self.epoch += 1
        return self

    def __next__(self):
        # 如果所有的桶都是空的，那么抛出 StopIteration。
        if ((self.buckets_len-self.readed_index)==0).all(): raise StopIteration
        
        # 如果需要随机化，随机选择一个非空桶。
        if self.shuffle: chosen_bucket_idx = self.rng.choice(len(self.buckets), p=(self.buckets_len-self.readed_index)/np.sum(self.buckets_len-self.readed_index)) 
        # 否则，按照桶的顺序选择一个非空桶。
        else: chosen_bucket_idx = 0

        readed_index = self.readed_index[chosen_bucket_idx]
        self.readed_index[chosen_bucket_idx] += self.batch_size
        chosen_bucket = self.buckets[chosen_bucket_idx]
        batch = chosen_bucket[readed_index:(readed_index+self.batch_size)]
        return batch
        
    def __len__(self):
        return len(self.spec_header)

    def generate_bins(self):
        if self.shuffle: 
            for bucket in self.buckets_ori:
                self.rng.shuffle(bucket)
        
        if dist.is_initialized():
            if self.drop_last:
                self.buckets = [bucket[:-(len(bucket)%dist.get_world_size())] if len(bucket)%dist.get_world_size()>0 else bucket for bucket in self.buckets_ori if len(bucket)//dist.get_world_size()>0]
                self.buckets = [bucket[dist.get_rank()::dist.get_world_size()] for bucket in self.buckets]
                self.buckets = [bucket[:-(len(bucket)%self.batch_size)] if len(bucket)%self.batch_size>0 else bucket for bucket in self.buckets if len(bucket)//self.batch_size>0]
            else:
                self.buckets = [bucket[dist.get_rank()::dist.get_world_size()] for bucket in self.buckets_ori]
        else:
            if self.drop_last:
                self.buckets = [bucket[:-(len(bucket)%self.batch_size)] if len(bucket)%self.batch_size>0 else bucket for bucket in self.buckets_ori if len(bucket)//self.batch_size>0]
            else:
                self.buckets = self.buckets_ori.copy()