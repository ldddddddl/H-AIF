import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

import torch
from torch.utils.data import DataLoader
import numpy as np


def _try_import_lerobot():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset
    except Exception as exc:
        raise ImportError("未找到 lerobot 库，请先安装：pip install lerobot") from exc


def _snapshot_download_dataset(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None,
) -> Path:
    """
    使用 Hugging Face Hub 将数据集下载到 local_dir。
    若 local_dir 已存在且不为空，则跳过下载。
    """
    local_dir = local_dir.resolve()
    if local_dir.exists():
        # 目录存在且非空视为已下载
        has_any = any(local_dir.iterdir())
        if has_any:
            return local_dir

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise ImportError(
            "缺少 huggingface_hub，请先安装：pip install huggingface_hub"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return local_dir


def ensure_dataset_local(
    repo_id: str = "lddddl/jetmax_dataset_v4",
    local_dir: Optional[str] = None,
) -> Path:
    """
    确保数据集已在本地可用；若不存在则从 Hub 下载。
    返回本地根目录路径。
    """
    repo_name = repo_id.split("/")[-1]
    # 默认放置到项目内 datasets/jetmax_dataset_v4（与转换/上传脚本保持一致）
    default_dir = Path(os.getcwd()) / "datasets" / repo_name
    target_dir = Path(local_dir) if local_dir else default_dir
    token = os.environ.get("HF_TOKEN")
    return _snapshot_download_dataset(
        repo_id=repo_id, local_dir=target_dir, token=token
    )


def _instantiate_lerobot_dataset(
    local_root: Path,
    repo_id: Optional[str] = None,
    split: Optional[str] = None,
):
    """
    兼容不同版本 lerobot 的数据集实例化方式。
    优先从本地 root 加载，失败时尝试从 Hub 加载（若实现）。
    """
    LeRobotDataset = _try_import_lerobot()

    # 退化为直接构造
    try:
        return LeRobotDataset(root=str(local_root), repo_id=repo_id)
    except Exception:
        # 若本地读取失败且存在 from_hub 接口，则尝试从 Hub 直接实例化
        if repo_id and hasattr(LeRobotDataset, "from_hub"):
            return getattr(LeRobotDataset, "from_hub")(repo_id=repo_id)
        raise


def _split_episodes(
    total_episodes: int, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[int], List[int]]:
    import numpy as np

    assert 0.0 < train_ratio < 1.0, "train_ratio 必须在 (0,1) 之间"
    ep_indices = np.arange(total_episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(ep_indices)
    n_train = int(round(train_ratio * total_episodes))
    train_eps = ep_indices[:n_train].tolist()
    val_eps = ep_indices[n_train:].tolist()
    return train_eps, val_eps


def _build_delta_timestamps(fps: int, horizon: int) -> Dict[str, List[float]]:
    if horizon <= 0:
        return {}
    step = 1.0 / float(fps)
    # 为 action 构造 t..t+h-1 的时间偏移
    return {"action": [i * step for i in range(horizon)]}


class _NormalizeWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        norm_stats: Dict[str, dict],
        keys: Optional[List[str]] = None,
    ):
        self.base = base_dataset
        self.keys = keys or ["state", "action"]
        # 预转换为 torch 张量，减少每次索引的开销
        self.stats: Dict[str, Dict[str, torch.Tensor]] = {}
        for k in self.keys:
            if k in norm_stats:
                st = norm_stats[k]
                self.stats[k] = {
                    "mean": torch.as_tensor(np.array(st.mean)),
                    "std": torch.as_tensor(np.array(st.std)).clamp(min=1e-6),
                    "q01": torch.as_tensor(np.array(st.q01))
                    if getattr(st, "q01", None) is not None
                    else None,
                    "q99": torch.as_tensor(np.array(st.q99))
                    if getattr(st, "q99", None) is not None
                    else None,
                }
        # 对于缺失统计项的键，跳过归一化

    def __len__(self):
        return len(self.base)

    def _apply_norm(self, x: torch.Tensor, st: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 广播到 x 的最后一维
        device = x.device
        mean = st["mean"].to(device)
        std = st["std"].to(device)
        q01 = st.get("q01")
        q99 = st.get("q99")
        if q01 is not None and q99 is not None:
            x = torch.clamp(x, min=q01.to(device), max=q99.to(device))
        return (x - mean) / std

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        for k, st in self.stats.items():
            if k not in item:
                continue
            x = item[k]
            # 支持 action chunk: (H, D) 或 (D,)
            if isinstance(x, torch.Tensor):
                if x.ndim == 1:
                    item[k] = self._apply_norm(x, st)
                elif x.ndim >= 2:
                    # 将最后一维视为特征维做归一化
                    shape = x.shape
                    x2 = x.reshape(-1, shape[-1])
                    x2 = self._apply_norm(x2, st)
                    item[k] = x2.reshape(*shape)
        return item


def load_lerobot_dataloader(
    repo_id: str = "lddddl/jetmax_dataset_v4",
    local_dir: Optional[str] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    train_ratio: float = 0.8,
    seed: int = 42,
    horizon: int = 0,
    normalize: bool = True,
    norm_stats_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, object, object]:
    """
    - 检查/下载数据集（Hugging Face Hub）
    - 按 episode 比例切分训练/验证
    - 依据 horizon 生成 action chunk（t..t+h-1）
    - 使用 norm_stats.json 执行归一化
    - 返回 (train_loader, val_loader, train_dataset, val_dataset)
    """
    from normalize import load as load_norm

    local_root = ensure_dataset_local(repo_id=repo_id, local_dir=local_dir)

    # 先实例化一次以获取 fps 与 episode 总数
    base_ds = _instantiate_lerobot_dataset(
        local_root=local_root, repo_id=repo_id, split=None
    )
    total_eps = base_ds.meta.total_episodes
    fps = base_ds.fps

    train_eps, val_eps = _split_episodes(total_eps, train_ratio=train_ratio, seed=seed)

    # 根据 horizon 构造 delta_timestamps（仅对 action）
    delta_ts = _build_delta_timestamps(fps=fps, horizon=horizon)

    # 重新构建训练/验证数据集
    LeRobot = type(base_ds)
    train_ds = LeRobot(
        repo_id=repo_id,
        root=str(local_root),
        episodes=train_eps,
        delta_timestamps=delta_ts,
    )
    val_ds = LeRobot(
        repo_id=repo_id,
        root=str(local_root),
        episodes=val_eps,
        delta_timestamps=delta_ts,
    )

    # 归一化
    if normalize:
        # 默认在 datasets/<repo_name>/norm_stats.json
        repo_name = repo_id.split("/")[-1]
        default_norm_dir = Path.cwd() / "datasets" / repo_name
        norm_dir = Path(norm_stats_dir) if norm_stats_dir else default_norm_dir
        norm_stats = load_norm(norm_dir)
        train_ds = _NormalizeWrapper(train_ds, norm_stats)
        val_ds = _NormalizeWrapper(val_ds, norm_stats)

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_ds, val_ds


def denormalize_action(
    normalized_action: torch.Tensor,
    norm_stats_or_dir: Union[Dict[str, dict], str, Path],
    key: str = "action",
    clamp: bool = True,
) -> torch.Tensor:
    """
    将归一化后的动作反变换回原始尺度。

    支持两种输入：
    - norm_stats_or_dir 为目录路径（包含 norm_stats.json），将使用 normalize.load() 读取
    - norm_stats_or_dir 为已加载的 norm_stats 字典（来自 normalize.load 的返回，或 _NormalizeWrapper.stats 的等价结构）

    形状兼容：(D,) 或 (H, D)；会对最后一维执行广播。
    """
    from normalize import load as load_norm

    # 读取统计
    if isinstance(norm_stats_or_dir, (str, Path)):
        stats_dict = load_norm(norm_stats_or_dir)
        st = stats_dict.get(key)
        if st is None:
            raise KeyError(f"norm_stats 中缺少键: {key}")
        mean = torch.as_tensor(np.array(st.mean), device=normalized_action.device)
        std = torch.as_tensor(np.array(st.std), device=normalized_action.device).clamp(
            min=1e-6
        )
        q01 = (
            torch.as_tensor(np.array(st.q01), device=normalized_action.device)
            if getattr(st, "q01", None) is not None
            else None
        )
        q99 = (
            torch.as_tensor(np.array(st.q99), device=normalized_action.device)
            if getattr(st, "q99", None) is not None
            else None
        )
    else:
        st = norm_stats_or_dir.get(key)
        if st is None:
            raise KeyError(f"norm_stats 中缺少键: {key}")
        # 支持两种结构：来自 _NormalizeWrapper.stats 的字典 或 normalize.load 的 dataclass
        if isinstance(st, dict) and "mean" in st:
            mean = st["mean"].to(normalized_action.device)
            std = st["std"].to(normalized_action.device).clamp(min=1e-6)
            q01 = st.get("q01")
            q99 = st.get("q99")
            if q01 is not None:
                q01 = q01.to(normalized_action.device)
            if q99 is not None:
                q99 = q99.to(normalized_action.device)
        else:
            mean = torch.as_tensor(np.array(st.mean), device=normalized_action.device)
            std = torch.as_tensor(
                np.array(st.std), device=normalized_action.device
            ).clamp(min=1e-6)
            q01 = (
                torch.as_tensor(np.array(st.q01), device=normalized_action.device)
                if getattr(st, "q01", None) is not None
                else None
            )
            q99 = (
                torch.as_tensor(np.array(st.q99), device=normalized_action.device)
                if getattr(st, "q99", None) is not None
                else None
            )

    # 反归一化
    orig = normalized_action * std + mean
    if clamp and (q01 is not None) and (q99 is not None):
        orig = torch.clamp(orig, min=q01, max=q99)
    return orig


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="JetMax LeRobot 数据集下载与加载（支持split/horizon/normalize）"
    )
    parser.add_argument(
        "--repo-id", default="lddddl/jetmax_dataset_v3", help="HF 数据集仓库 ID"
    )
    parser.add_argument(
        "--local-dir", default=None, help="本地数据集目录（默认 datasets/<repo_name>）"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-shuffle", action="store_true", help="禁用打乱")
    parser.add_argument("--no-pin-memory", action="store_true", help="禁用 pin_memory")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="训练集比例(0,1)"
    )
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument(
        "--horizon", type=int, default=0, help="action horizon 大小，0 表示不拼接"
    )
    parser.add_argument("--no-normalize", action="store_true", help="禁用归一化")
    parser.add_argument(
        "--norm-stats-dir", default=None, help="norm_stats.json 所在目录（包含该文件）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    try:
        train_loader, val_loader, train_ds, val_ds = load_lerobot_dataloader(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            batch_size=args.batch_size,
            shuffle=not args.no_shuffle,
            num_workers=args.num_workers,
            pin_memory=not args.no_pin_memory,
            train_ratio=args.train_ratio,
            seed=args.seed,
            horizon=args.horizon,
            normalize=not args.no_normalize,
            norm_stats_dir=args.norm_stats_dir,
        )
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
