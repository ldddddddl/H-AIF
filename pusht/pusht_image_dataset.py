from typing import Dict
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            stride=horizon,  # 关键修改：步长=序列长度
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            stride=self.horizon,  # 关键修改：步长=序列长度
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_val_indices(self):
        return self.sampler.get_indices()
    
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample, index = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)

        return torch_data, index


def test_is_repeat(dataset):
    # 修改后的测试代码
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for batch, index in dataloader:
        # 检查相邻批次的首尾样本是否不同
        print(batch['obs']['image'][0, -1].sum(), batch['obs']['image'][1, 0].sum())

def test():
    import os
    zarr_path = os.path.expanduser(r'D:/diffusion_policy-main/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=10)
    # test_is_repeat(dataset)
    
    # 在 __main__ 里或者你的测试函数里加入

    indices = dataset.sampler.indices  # shape=(N, 6)

    # # 按 buffer_start_idx 排序，然后检查相邻窗口
    # sorted_idx = np.argsort(indices[:, 0])
    # for i0, i1 in zip(sorted_idx[:-1], sorted_idx[1:]):
    #     end_of_prev = indices[i0, 1]
    #     start_of_next = indices[i1, 0]
    #     if end_of_prev > start_of_next:
    #         print("重叠！", i0, i1, end_of_prev, start_of_next)

    
    
    
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # from matplotlib import pyplot as plt
    # for x, index in dataloader:
    #     print(x['obs']['image'][0, -1].sum(), x['obs']['image'][1, 0].sum())
        
    #     for i in range(x['obs']['image'].shape[0]):
    #         for j in range(x['obs']['image'].shape[1]):
    #             plt.imshow(x['obs']['image'][i][j].transpose(0, -1))
    #             plt.title(f'{i}_{j}')
    #             plt.show()
    #             pass
    
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == '__main__':
    test()
