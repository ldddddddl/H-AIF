import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0):
        super(LIFNeuron, self).__init__()
        self.threshold = threshold
        self.mem = None

    def forward(self, input_):
        if self.mem is None:
            self.mem = torch.zeros(input_.shape).to(input_.device)  # 初始化膜电位

        self.mem += input_  # 累积输入
        spikes = (self.mem > self.threshold).float()  # 发放脉冲
        self.mem -= self.threshold * spikes  # 重置膜电位
        return spikes, self.mem


def process_images(image_folder, spike_image_folder, threshold=0.5):
    # 创建保存脉冲图像的文件夹
    if not os.path.exists(spike_image_folder):
        os.makedirs(spike_image_folder)

    # 获取文件夹中所有图片文件名，并按顺序排序
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 假设文件名是数字命名的，按数字顺序排序

    # 创建LIF神经元实例
    lif_neuron = LIFNeuron(threshold=threshold)

    # 遍历连续的每两张图片
    for i in range(len(image_files) - 1):
        image_path1 = os.path.join(image_folder, image_files[i])
        image_path2 = os.path.join(image_folder, image_files[i + 1])

        # 读取并转换图片为灰度图和PyTorch张量
        image1 = Image.open(image_path1).convert('L')
        image2 = Image.open(image_path2).convert('L')
        transform = lambda x: torch.tensor(np.array(x), dtype=torch.float32).unsqueeze(0) / 255.0
        image_tensor1 = transform(image1)
        image_tensor2 = transform(image2)

        # 计算两张图像的差异
        difference = torch.abs(image_tensor1 - image_tensor2)

        # 将差异转换为脉冲序列
        spikes, _ = lif_neuron(difference)

        # 将脉冲图像保存到指定文件夹
        spike_image_path = os.path.join(spike_image_folder, f'spike_image_{i}.jpg')
        spike_image = Image.fromarray((spikes.squeeze(0).numpy() * 255).astype(np.uint8))  # 将脉冲张量转换为PIL图像
        spike_image.save(spike_image_path)


def process_all_folders(root_folder, threshold=0.5):
    # 遍历根文件夹下的所有子文件夹
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # 检查是否是文件夹
        if os.path.isdir(folder_path):
            visual_folder = os.path.join(folder_path, 'visual')
            spike_image_folder = os.path.join(folder_path, 'spike_image')

            # 检查是否存在visual文件夹
            if os.path.exists(visual_folder):
                print(f"正在处理文件夹: {folder_name}")
                process_images(visual_folder, spike_image_folder, threshold=threshold)
            else:
                print(f"文件夹 {folder_name} 中没有 visual 文件夹，跳过处理。")


# 指定根文件夹路径
root_folder = r'C:\Users\甄海乾\Desktop\Grasping_state_assessment_master (2)\Grasping_state_assessment_master\sorted data\traindata'

# 调用函数处理所有文件夹
process_all_folders(root_folder, threshold=0.5)
