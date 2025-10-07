import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from itertools import islice
# from train import read_config
import torchvision.transforms as transforms



class GetDataset(Dataset):
    def __init__(self, episode_path, batchsize=8, config=None, transform_v=None, phase=None):
        super().__init__()
        self.config = config
        
        self.batchsize = config.batchsize
        self.file_cnt = 0
        self.getitem_cnt = 0
        self.data = None
        self.data_epis = {}
        self.transform_v = transform_v
        self.data_list = []
        self.phase = phase
        # 加载npz文件中的数据
        for p in episode_path:
            file_index, ep_path = p
            data = np.load(ep_path, allow_pickle=True)
            data_dict = {key: data[key] for key in data}
            datas = self.nested_dict_from_dot_keys(data_dict)
            # self.data_epis = self.datas['datas']
            data_epis = list(self.split_dict(datas['datas'], self.config.past_img_num + self.config.future_img_num))
            self.data_list += data_epis

        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        return:
        timestamp_motors_sucker: [S, H]  
        gripper_frame_seq: [S, W, H, C]
        side_frame_seq: [S, W, H, C]
        '''
        batch_raw = self.data_list[idx]
        timestamp_list = list(batch_raw.keys())  # get time seq
        if len(timestamp_list) >= (self.config.past_img_num + self.config.future_img_num):
            
            timestamp_datas = timestamp_list[:self.config.past_img_num]   # 前n张为过去图像
            timestamp_labels = timestamp_list[self.config.past_img_num:]  # 后n张为未来图像
        else:
            timestamp_datas = timestamp_list[:len(timestamp_list) // 2]   # 前半为训练
            timestamp_labels = timestamp_list[len(timestamp_list) // 2:]  # 后半为标签

        # timestamp_tensor = torch.tensor(timestamp_datas).unsqueeze(-1)
        # get sucker actions
        sucker = np.concatenate([batch_raw[de]["action"]["sucker_actions"] for de in timestamp_datas], axis=0)
        # sucker_bool = [s[:, 1:] for s in sucker]
        sucker_bool = sucker[:, 1:]
        # sucker_bool_np_temp = np.array(sucker_bool)
        sucker_tensor = torch.tensor(sucker_bool, dtype=torch.float32)
        # get position signal
        # when convert to motors
        # motor_temp = [self.data_epis[idx][de]["action"]["motor_angle"] for de in timestamp_datas]
        # motor_list_temp = [[m["motor_1"].item(), m["motor_2"].item(), m["motor_3"].item()] for m in motor_temp]
        # joints
        # [timestamp, joint1, ···, joint9]   
        joints_temp = np.concatenate([batch_raw[de]["action"]["joints_position"] for de in timestamp_datas], axis=0)
        # from matplotlib import pyplot as plt
        # temps = joints_temp[:, 1:]
        # for i in range(temps.shape[1]):
        #     plt.plot(temps[:, i])
        # # plt.title(phase)
        # plt.show()
        
        # np_temp = np.array(joints_temp)
        joints_tensor = torch.tensor(joints_temp, dtype=torch.float32)
        # cat [[motors, sucker]]
        timestamp_joints_sucker = torch.cat([joints_tensor, sucker_tensor], dim=-1)  #
        # get image frames
        gripper_frame_list = [batch_raw[de]["observation"]["gripper_video"] for de in timestamp_datas]
        transform_gripper_frame_list = [self.transform_v(img.transpose(0, -1)).transpose(0, -1).unsqueeze(0) for img in gripper_frame_list]
        side_frame_list = [batch_raw[de]["observation"]["side_video"] for de in timestamp_datas]
        transform_side_frame_list = [self.transform_v(img.transpose(0, -1)).transpose(0, -1).unsqueeze(0) for img in side_frame_list]
        # add timestamp to image
        # timestamp_tensor = [torch.full((1, transform_gripper_frame_list[0].shape[1], 3), t) for t in timestamp_datas]  #
        # gripper_frame_timestamp = [torch.cat(z, dim=0).unsqueeze(0) for z in zip(transform_gripper_frame_list, timestamp_tensor)]
        # side_frame_timestamp = [torch.cat(z, dim=0).unsqueeze(0) for z in zip(transform_side_frame_list, timestamp_tensor)]
        # cat to seq
        gripper_frame_seq = torch.cat(transform_gripper_frame_list, dim=0)
        side_frame_seq = torch.cat(transform_side_frame_list, dim=0)
        
        # get labels
        sucker_ = np.concatenate([batch_raw[de]["action"]["sucker_actions"] for de in timestamp_labels], axis=0)
        # np_temp = np.array(sucker_)
        # sucker_bool_ = [s[:, 1:] for s in sucker_]
        sucker_bool_ = sucker_[:, 1:]
        # sucker_bool_np_temp_ = np.array(sucker_bool_)
        sucker_labels = torch.tensor(sucker_bool_, dtype=torch.long).squeeze(-1)

        joints_temp_ = np.concatenate([batch_raw[de]["action"]["joints_position"] for de in timestamp_labels], axis=0)
        # joints_without_timest = [j[:, 1:] for j in joints_temp_] 
        if self.config.data_format == 'joints':
            joints_without_timest = joints_temp_[:, 1:]
        else: # == rpy
            joints_without_timest = joints_temp_

        # np_temp = np.array(joints_without_timest)
        joints_pos_labels = torch.tensor(joints_without_timest, dtype=torch.float32)

        # np_temp_ = np.array(joints_temp_)
        joints_pos_labels_with_timestamp = torch.tensor(joints_temp_, dtype=torch.float32)


        gripper_frame_list_ = [batch_raw[de]["observation"]["gripper_video"].unsqueeze(0) for de in timestamp_labels]
        # add timestamp to image
        # timestamp_tensor_label = [torch.full((1, gripper_frame_list_[0].shape[1], 3), t) for t in timestamp_labels] 
        # gripper_timestamp_label = [torch.cat(z, dim=0).unsqueeze(0) for z in zip(gripper_frame_list_, timestamp_tensor_label)]
        gripper_frame_labels = torch.cat(gripper_frame_list_, dim=0)
        
        side_frame_list_ = [batch_raw[de]["observation"]["side_video"].unsqueeze(0) for de in timestamp_labels]
        # add timestamp to image
        # side_timestamp_label = [torch.cat(z, dim=0).unsqueeze(0) for z in zip(side_frame_list_, timestamp_tensor_label)]
        side_frame_labels = torch.cat(side_frame_list_, dim=0)
        
        labels = [sucker_labels, joints_pos_labels, gripper_frame_labels, side_frame_labels, joints_pos_labels_with_timestamp]


        ## test 
        # varid_dataloader(gripper_frame_seq, side_frame_seq, labels)
        
        return timestamp_joints_sucker, gripper_frame_seq, side_frame_seq, labels

    def nested_dict_from_dot_keys(self, data):
        """将以点分隔的键名转换为嵌套字典"""
        result = {}
        for key, value in data.items():
            keys = key.split('.')  # 将点分隔的键名拆分
            d = result
            for k in keys[:-1]:
                d = d.setdefault(k, {})  # 嵌套字典逐层添加
                
            if isinstance(value, np.ndarray):
                d[keys[-1]] = torch.tensor(value, dtype=torch.float32) if value.dtype in [np.int64, np.float32] else value
            else:
                d[keys[-1]] = torch.tensor(value) if isinstance(value, (int, float)) else value
            # d[keys[-1]] = torch.tensor(value) if value.dtype in ["int64", "float32", np.float32] else value # to tensor 
            # print(value.dtype)
        return result

    def split_dict(self, big_dict, chunk_size):
        it = iter(big_dict)
        for _ in range(0, len(big_dict), chunk_size):
            yield {int(k) / int(next(reversed(big_dict))): big_dict[k] for k in islice(it, chunk_size)}

def varid_dataloader(gripper_frame_seq, side_frame_seq, labels):
    import matplotlib.pyplot as plt
    import torch

    # 假设 dataloader 返回的 tensor 形状如下：
    # image_data: [seq, h, w, c]
    # signal_data: [seq, 12, 9]

    # 模拟加载的图像数据和电信号数据
    seq_length = 5  # 序列长度
    # h, w, c = gripper_frame_seq.shape  # 图像尺寸
    # image_data = torch.randint(0, 255, (seq_length, h, w, c), dtype=torch.uint8)  # 假设的图像数据
    # signal_data = torch.rand((seq_length, 12, 9))  # 假设的电信号数据

    # 1. 绘制图像数据
    fig1, axes1 = plt.subplots(1, seq_length, figsize=(15, 3))
    for i in range(seq_length):
        img_np = gripper_frame_seq[i].numpy()  # 转换为 numpy 数组
        axes1[i].imshow(img_np)
        axes1[i].axis('off')
        axes1[i].set_title(f'Grepper_input {i+1}')


    fig2, axes2 = plt.subplots(1, seq_length, figsize=(15, 3))
    for i in range(seq_length):
        img_np = side_frame_seq[i].numpy()  # 转换为 numpy 数组
        axes2[i].imshow(img_np)
        axes2[i].axis('off')
        axes2[i].set_title(f'Side_input {i+1}')

    
    fig3, axes3 = plt.subplots(1, seq_length, figsize=(15, 3))
    for i in range(seq_length):
        img_np = labels[2][i].numpy()  # 转换为 numpy 数组
        axes3[i].imshow(img_np)
        axes3[i].axis('off')
        axes3[i].set_title(f'Gripper_label {i+1}')

    
    fig4, axes4 = plt.subplots(1, seq_length, figsize=(15, 3))
    for i in range(seq_length):
        img_np = labels[3][i].numpy()  # 转换为 numpy 数组
        axes4[i].imshow(img_np)
        axes4[i].axis('off')
        axes4[i].set_title(f'Side_label {i+1}')

    

    # 绘制电信号数据为折线图
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps = range(12)  # 时间步设置为12

    for j in range(9):
        ax.plot(time_steps, labels[1][0][:, j].numpy(), label=f'Channel {j+1}')

    ax.set_xlabel('Time Step')
    ax.set_ylabel('Signal Value')
    ax.set_title('Signal Data Over Time Steps')
    ax.legend(loc="upper right", fontsize="small")
    plt.show()


def pad_to_batch_size(tensor_list, batch_size):
    """
        对于传入的多个张量，如果它们的数量不足 batch_size，则通过复制数据的方式填充。
        
        参数：
        - tensor_list: 传入的张量列表，要求所有张量的 batch 维度大小相同
        - batch_size: 目标 batch 大小

        返回：
        - 填充后的张量列表，每个张量在 batch 维度上满足 batch_size
    """
    padded_tensors = []
    
    for tensor in tensor_list:
        current_size = tensor.shape[0]
        if current_size < batch_size:
            # 计算需要复制的次数
            repeats = (batch_size + current_size - 1) // current_size
            # 复制张量并截取到目标 batch 大小
            padded_tensor = tensor.repeat(repeats, *([1] * (tensor.dim() - 1)))[:batch_size]
        else:
            padded_tensor = tensor[:batch_size]  # 如果当前大小已经大于等于 batch_size，直接截取
        padded_tensors.append(padded_tensor)
    
    return padded_tensors

def sort_dataset(npz_file:str):
    # data_list = []
    file_list = os.listdir(npz_file)
    file_path_list = [[i_count, os.path.join(npz_file, i)] for i_count, i in enumerate(file_list)]
    return file_path_list

# 使用DataLoader加载数据
class CreateDatasets():
    def __init__(self, xlstm_cfg) -> None:
        self.transform_v = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.config = xlstm_cfg
    
    def create_dataloader(self, npz_file_list:list, batch_size=None,  
                          shuffle:bool=False, num_workers:int=0, phase:str=''):
        dataset = GetDataset(npz_file_list, config=self.config, transform_v=self.transform_v, phase=phase)
        if batch_size is None:
            batch_size = self.config.batchsize
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader

if __name__ == "__main__":
    # ParamConfig = read_config()

    # 示例：使用自定义的 DataLoader
    npz_file = 'datasets/datas_npz'  # 替换为你的npz文件路径
    file_path_list = sort_dataset(npz_file)
    for ep_cnt, ep_path in enumerate(file_path_list):
        dataloader = create_dataloader(ep_path)

        # 遍历DataLoader
        for batch_idx, (timestamp_motors_sucker, gripper_frame_timestamp, side_frame_timestamp) in enumerate(dataloader):
            print(f"Batch {timestamp_motors_sucker.shape, gripper_frame_timestamp.shape, side_frame_timestamp.shape}:")

            if timestamp_motors_sucker.size(0)  < ParamConfig.batchsize:
                padded_tensors = pad_to_batch_size([timestamp_motors_sucker, gripper_frame_timestamp, side_frame_timestamp], ParamConfig.batchsize)
                timestamp_motors_sucker, gripper_frame_timestamp, side_frame_timestamp = padded_tensors
