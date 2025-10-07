import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from spikingjelly.activation_based import ann2snn, functional, neuron
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

# 设置 matplotlib 字体以支持中文
plt.rcParams['font.family'] = 'SimHei'  # 可根据系统替换为其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Config:
    data_root = r"C:\Users\10931\Desktop\Grasping_state_assessment_master(2)\Grasping_state_assessment_master(2)\sorted data\traindata"  # 修改为实际路径
    batch_size = 8
    num_epochs = 100
    max_seq_len = 50
    ann_lr = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = 20
    calibration_mode = 'max'


class VoltageScaler:
    def __init__(self):
        self.global_min = None
        self.global_max = None

    def fit(self, train_groups):
        all_voltages = []
        print("\n计算训练集电压标准化:")
        for group in tqdm(train_groups):
            tactile_file = os.path.join(Config.data_root, group, "flex", f"resampled_{group}.xlsx")
            try:
                df = pd.read_excel(tactile_file, engine='openpyxl')
                all_voltages.extend(df["Amplitude - Plot 0"].values)
            except Exception as e:
                continue
        self.global_min = np.min(all_voltages)
        self.global_max = np.max(all_voltages)
        print(f"电压范围: [{self.global_min:.4f}, {self.global_max:.4f}]")

    def transform(self, voltages):
        if self.global_max == self.global_min:
            return np.full_like(voltages, 0.5)
        return (voltages - self.global_min) / (self.global_max - self.global_min)


class TimeSeriesDataset(Dataset):
    def __init__(self, groups, scaler):
        super().__init__()
        self.samples = []
        self.image_cache = {}
        print("\n加载数据集中:")

        for group in tqdm(groups):
            group_path = os.path.join(Config.data_root, group)
            visual_dir = os.path.join(group_path, "spike_image")

            img_files = sorted([f for f in os.listdir(visual_dir) if f.endswith('.jpg')],
                               key=lambda x: int(x.split('_')[-1].split('.')[0]))
            img_paths = [os.path.join(visual_dir, f) for f in img_files]

            tactile_file = os.path.join(group_path, "flex", f"resampled_{group}.xlsx")
            try:
                df = pd.read_excel(tactile_file, engine='openpyxl')
                voltages = df["Amplitude - Plot 0"].values
            except Exception as e:
                continue

            min_len = min(len(img_paths), len(voltages))
            self.samples.append({
                'image_paths': img_paths[:min_len],
                'voltages': scaler.transform(voltages[:min_len])
            })

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        images = []
        for p in sample['image_paths']:
            if p in self.image_cache:
                img = self.image_cache[p]
            else:
                img = self.transform(Image.open(p).convert("RGB"))
                self.image_cache[p] = img
            images.append(img)

        return (
            torch.stack(images),  # [seq_len, 3, 64, 64]
            torch.FloatTensor(sample['voltages']),
            len(images)
        )


def train_collate_fn(batch):
    images, volts, lengths = zip(*batch)
    max_len = min(Config.max_seq_len, max(lengths))

    padded_images = []
    padded_volts = []
    for imgs, v in zip(images, volts):
        seq_len = imgs.size(0)
        if seq_len < max_len:
            padding = torch.zeros(max_len - seq_len, 3, 64, 64)
            padded_img = torch.cat([imgs, padding], dim=0)
            padded_volt = torch.cat([v, torch.zeros(max_len - seq_len)], dim=0)
        else:
            padded_img = imgs[:max_len]
            padded_volt = v[:max_len]
        padded_images.append(padded_img)
        padded_volts.append(padded_volt)

    return (
        torch.stack(padded_images),  # [batch, seq_len, C, H, W]
        torch.stack(padded_volts),  # [batch, seq_len]
        torch.LongTensor([min(l, max_len) for l in lengths])
    )


def calib_collate_fn(batch):
    images, _, _ = zip(*batch)
    padded_images = []
    for imgs in images:
        seq_len = imgs.size(0)
        if seq_len < Config.max_seq_len:
            padding = torch.zeros(Config.max_seq_len - seq_len, 3, 64, 64)
            padded_img = torch.cat([imgs, padding], dim=0)
        else:
            padded_img = imgs[:Config.max_seq_len]
        padded_images.append(padded_img)
    # 添加一个虚拟的第二个返回值
    dummy_value = torch.zeros(len(padded_images))
    return torch.stack(padded_images), dummy_value


class ANNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            self.cnn_features = self.cnn(dummy).shape[-1]

        self.gru = nn.GRU(self.cnn_features, 128, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.to(Config.device)

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        x = x.reshape(-1, 3, 64, 64)  # [batch*seq_len, 3, 64, 64]
        cnn_out = self.cnn(x)  # [batch*seq_len, cnn_features]
        cnn_out = cnn_out.view(batch_size, seq_len, -1)  # [batch, seq_len, features]
        gru_out, _ = self.gru(cnn_out)  # [batch, seq_len, 128]
        return self.fc(gru_out).squeeze(-1)  # [batch, seq_len]


class SNNConverter:
    def __init__(self, ann_model, train_dataset):
        self.ann_model = ann_model
        self.train_dataset = train_dataset
        self.snn_model = self._convert()
        self._validate()

    def _convert(self):
        print("\n开始转换ANN到SNN...")
        calib_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.batch_size,
            collate_fn=calib_collate_fn,
            shuffle=False
        )

        converter = ann2snn.Converter(
            mode=Config.calibration_mode,
            dataloader=calib_loader,
            device=Config.device
        )
        return converter(self.ann_model)

    def _validate(self):
        input_data = self.train_dataset[0][0].unsqueeze(0).to(Config.device)
        output = self.snn_model(input_data)
        print("实际输出维度:", output.shape)
        expected_shape = (1, min(Config.max_seq_len, len(self.train_dataset[0][0])))
        assert output.shape == expected_shape, f"输出维度错误，期望维度 {expected_shape}，实际维度 {output.shape}"


def visualize_results(images, true_voltages, pred_voltages, epoch, model_type):
    images = images.cpu()
    true_voltages = true_voltages.cpu().numpy()
    pred_voltages = pred_voltages.detach().cpu().numpy()

    # 裁剪图像数据到有效范围
    images = torch.clamp(images, 0, 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(true_voltages, label='真实值')
    plt.plot(pred_voltages, label='预测值')
    plt.title("电压预测对比")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(np.hstack([images[i].permute(1, 2, 0) for i in range(0, len(images), 5)]))
    plt.title("样本图像")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{model_type}_epoch_{epoch}.png")
    plt.close()


def main():
    # 数据准备
    all_groups = [d for d in os.listdir(Config.data_root)
                  if os.path.isdir(os.path.join(Config.data_root, d))]
    train_groups, test_groups = train_test_split(all_groups, test_size=0.2, random_state=42)

    scaler = VoltageScaler()
    scaler.fit(train_groups)

    train_set = TimeSeriesDataset(train_groups, scaler)
    test_set = TimeSeriesDataset(test_groups, scaler)

    # 模型训练
    model = ANNModel()
    optimizer = optim.Adam(model.parameters(), lr=Config.ann_lr)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        train_set,
        batch_size=Config.batch_size,
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True
    )

    print("\n开始训练ANN...")
    for epoch in range(Config.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, volts, _ in pbar:
            images = images.to(Config.device, non_blocking=True)
            volts = volts.to(Config.device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, volts[:, :outputs.size(1)])
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        # 可视化
        sample_idx = random.randint(0, len(train_set) - 1)
        sample_images, sample_volts, _ = train_set[sample_idx]
        with torch.no_grad():
            preds = model(sample_images.unsqueeze(0).to(Config.device))
        visualize_results(sample_images, sample_volts, preds[0], epoch, "ann")

    # 转换SNN
    print("\n转换SNN...")
    print("\n转换SNN...")
    converter = SNNConverter(model, train_set)
    torch.save(converter.snn_model.state_dict(), "snn_model.pth")

    # SNN推理
    test_loader = DataLoader(test_set, batch_size=1, collate_fn=train_collate_fn)
    converter.snn_model.eval()
    with torch.no_grad():
        for images, volts, _ in test_loader:
            images = images.to(Config.device)
            snn_input = images.permute(1, 0, 2, 3, 4)

            outputs = []
            for _ in range(Config.T):
                functional.reset_net(converter.snn_model)
                outputs.append(converter.snn_model(snn_input))

            avg_output = torch.stack(outputs).mean(dim=0)
            visualize_results(images[0], volts[0], avg_output[0], 0, "snn")


if __name__ == "__main__":
    main()
