import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns  # 仅用于配色
import matplotlib as mpl

FONT_SIZE = 72

# 全局统一字体大小（可根据需要调整数字）
mpl.rcParams['axes.titlesize']   = FONT_SIZE   # 所有标题默认 18pt
mpl.rcParams['axes.labelsize']   = FONT_SIZE   # 坐标轴标签默认 16pt
mpl.rcParams['xtick.labelsize']  = 48   # x 轴刻度数字默认 14pt
mpl.rcParams['ytick.labelsize']  = 48   # y 轴刻度数字默认 14pt
mpl.rcParams['legend.fontsize']  = FONT_SIZE   # 图例文字默认 14pt
mpl.rcParams['figure.titlesize'] = FONT_SIZE   # 整体 figure 标题默认 20pt

def visualize_latent_sequence(z: torch.Tensor,
                              method: str = 'pca',
                              n_components: int = 5,
                              per_batch: bool = False):
    """
    将形状为 (B, T, D) 的隐变量 z 可视化到 2D 或 3D。
    
    参数:
      - z: torch.Tensor, shape=(B, T, D)
      - method: 'pca' | 'tsne' | 'umap'
      - n_components: 降维目标维度 (2 或 3)
      - per_batch: 若 True，则对每个 batch 分别绘制子图；否则合并所有样本绘制一个图。
    """
    z0 = z[0]  # shape = (T, D)
    T, D = z0.shape
    z_flat = z0.detach().cpu().numpy()

    # 2. 降维
    m = method.lower()
    if m == 'pca':
        reducer = PCA(n_components=n_components)
    elif m == 'tsne':
        reducer = TSNE(n_components=n_components, init='pca', random_state=42)
    elif m == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unsupported method: {method}")

    z_proj = reducer.fit_transform(z_flat)  # shape = (T, n_components)

    # 3. 绘图
    fig = plt.figure(figsize=(15, 10))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(z_proj[:, 0], z_proj[:, 1], z_proj[:, 2],
                        c=np.arange(T), cmap='viridis', s=200)
        ax.set_zlabel('Dim 3')
    else:
        ax = fig.add_subplot(111)
        sc = ax.scatter(z_proj[:, 0], z_proj[:, 1],
                        c=np.arange(T), cmap='viridis', s=200)
    ax.set_xlabel('Dim 1', fontsize=FONT_SIZE)
    ax.set_ylabel('Dim 2', fontsize=FONT_SIZE)
    plt.colorbar(sc, label='Time step')
    plt.title(f'{method.upper()} of first sequence', fontsize=FONT_SIZE)
    plt.tight_layout()
    plt.show()
    
    
def visualize_heatmap(z: torch.Tensor, batch_idx: int = 0):
    """
    绘制单个样本 (batch_idx) 的 (T, D) 热图。
    """

    data = z[0].detach().cpu().numpy().T  # (D, T)
    plt.figure(figsize=(18, 12))
    plt.imshow(data, aspect='auto', origin='lower', cmap='coolwarm')
    plt.xlabel('Time step', fontsize=FONT_SIZE)
    plt.ylabel('Latent dim', fontsize=FONT_SIZE)
    plt.title(f'Heatmap', fontsize=FONT_SIZE)
    plt.colorbar(label='Activation')
    plt.tight_layout()
    plt.show()


def visualize_fn(zt, n_components=2):
    visualize_latent_sequence(zt, method='pca', n_components=n_components, per_batch=False)
    # 分批 tsne 可视化
    visualize_latent_sequence(zt, method='tsne', n_components=n_components, per_batch=False)
    visualize_latent_sequence(zt, method='umap', n_components=n_components, per_batch=False)
    # visualize_latent_sequence(zt, method='umap', n_components=n_components, per_batch=False)
    # 查看第 0 个样本的热图
    visualize_heatmap(zt, batch_idx=0)

# —— 使用示例 —— #
if __name__ == '__main__':
    # 假设 zt 已在 GPU 上
    zt = torch.randn(8, 120, 256).cuda()
    # 整体 PCA 可视化
    visualize_latent_sequence(zt, method='pca', n_components=5, per_batch=False)
    # 分批 tsne 可视化
    visualize_latent_sequence(zt, method='tsne', n_components=5, per_batch=False)
    # 查看第 0 个样本的热图
    visualize_heatmap(zt, batch_idx=0)
