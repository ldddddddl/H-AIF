import os
import sys
import hashlib
import json
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
    repo_id: str = "",
    local_dir: Optional[str] = None,
) -> Path:
    """
    确保数据集已在本地可用；优先本地目录；必要时从 Hub 下载。
    返回本地根目录路径。
    """
    # 1) 显式本地目录优先
    if local_dir:
        p = Path(local_dir).resolve()
        if p.exists():
            return p

    # 2) repo_id 也可能是本地目录
    if repo_id:
        p_repo = Path(repo_id).resolve()
        if p_repo.exists():
            return p_repo

    # 3) 需要从 Hub 下载（要求 repo_id 为有效的 Hub 路径）
    if not repo_id:
        raise ValueError("请提供可用的 repo_id 或 local_dir")
    repo_name = repo_id.split("/")[-1]
    default_dir = Path(os.getcwd()) / "datasets" / repo_name
    token = os.environ.get("HF_TOKEN")
    return _snapshot_download_dataset(repo_id=repo_id, local_dir=default_dir, token=token)


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


def _build_delta_timestamps(fps: int, horizon: int) -> Optional[Dict[str, List[float]]]:
    # 当 horizon<=0 时返回 None，避免触发 LeRobotDataset 的 delta_indices 路径
    # 在 episodes 子集场景下，delta_indices 会在 _get_query_indices 中用全局 episode_index 作为索引
    # 而 episode_data_index 仅有子集长度，可能产生越界。
    if horizon <= 0:
        return None
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


class _DecodeVideoRefs(torch.utils.data.Dataset):
    """
    将样本中的视频路径 + 帧索引，在线解码为图像张量并返回，移除原先的引用字段。

    - 期望输入键：
      - 'top_video_path' (str or 形如(1,)的ndarray[str])
      - 'top_frame_idx' (int or 形如(1,)的ndarray[int])
      - 'wrist_video_path'
      - 'wrist_frame_idx'
    - 输出键：
      - 'top_image': torch.FloatTensor, 形状 (3, H, W)，数值范围 [0, 1]
      - 'wrist_image': torch.FloatTensor, 形状 (3, H, W)，数值范围 [0, 1]
    """

    def __init__(self, base_dataset: torch.utils.data.Dataset, local_root:Path, image_size: int = 0,
                 decoder_cache_max: int = 32, frame_cache_max: int = 512,
                 past_img_num: int = 0, future_img_num: int = 0):
        self.base = base_dataset
        self.image_size = image_size  # 0 表示不强制缩放
        self.local_root = local_root
        # 图像时序窗口参数（以当前帧 t 为中心，采样 [t-past, ..., t+future]）
        self.past_img_num = int(max(0, past_img_num))
        self.future_img_num = int(max(0, future_img_num))
        # 解码器缓存（按视频路径复用 VideoReader/VideoCapture）
        from collections import OrderedDict as _OD
        self._decoder_cache = _OD()
        self._decoder_cache_max = int(decoder_cache_max)
        # 帧级别缓存：键为 (path, idx)，值为张量 (3,H,W)
        self._frame_cache = _OD()
        self._frame_cache_max = int(frame_cache_max)
        # 后端探测（每个进程/worker 各自一次）
        self._backend = None  # 'decord' 或 'opencv'

    def __len__(self):
        return len(self.base)

    def _as_py(self, v):
        # 将可能的 numpy 标量/数组包装转换为 Python 基本类型
        import numpy as _np

        if isinstance(v, (list, tuple)) and len(v) == 1:
            v = v[0]
        if isinstance(v, _np.ndarray):
            if v.shape == ():
                return v.item()
            if v.size == 1:
                return v.reshape(()).item()
            # 对于形如 (1,) 的数组，取第 0 个
            if v.ndim == 1 and v.shape[0] == 1:
                return v[0].item() if _np.isscalar(v[0]) else v[0]
        return v

    def _get_reader(self, video_path: str):
        # LRU 获取/创建解码器
        video_path = os.path.join(self.local_root, video_path)
        dec = self._decoder_cache.get(video_path)
        if dec is not None:
            self._decoder_cache.move_to_end(video_path)
            return dec

        # 探测后端
        if self._backend is None:
            try:
                self._backend = 'decord'
            except Exception:
                self._backend = 'opencv'

        if self._backend == 'decord':
            from decord import VideoReader  # type: ignore
            dec = VideoReader(video_path)
        else:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")
            dec = cap

        self._decoder_cache[video_path] = dec
        if len(self._decoder_cache) > self._decoder_cache_max:
            # 逐出最久未使用的，并做必要的释放
            old_path, old_dec = self._decoder_cache.popitem(last=False)
            try:
                import cv2  # type: ignore
                if hasattr(old_dec, 'release'):
                    old_dec.release()
            except Exception:
                pass
        return dec

    def _read_frame(self, video_path: str, frame_idx: int) -> torch.Tensor:
        # 帧级别 LRU 缓存
        key = (video_path, int(frame_idx))
        hit = self._frame_cache.get(key)
        if hit is not None:
            self._frame_cache.move_to_end(key)
            return hit

        # 使用缓存的解码器读取帧
        dec = self._get_reader(video_path)
        img = None
        if self._backend == 'decord':
            img_np = dec[frame_idx].asnumpy()
            img = img_np
        else:
            import cv2  # type: ignore
            dec.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = dec.read()
            if not ok:
                raise RuntimeError(f"读取视频帧失败: {video_path} @ {frame_idx}")
            img = frame_bgr[:, :, ::-1]

        # 到此 img 为 HWC, uint8/uint类型
        import numpy as _np
        if img.dtype != _np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).astype(_np.uint8)
            else:
                img = img.astype(_np.uint8)

        # 可选缩放到固定尺寸
        if self.image_size and self.image_size > 0:
            from PIL import Image

            pil = Image.fromarray(img)
            pil = pil.resize((self.image_size, self.image_size), Image.BICUBIC)
            img = _np.array(pil)

        # HWC(uint8) -> CHW(float32 in [0,1])
        chw = _np.transpose(img, (2, 0, 1)).astype(_np.float32) / 255.0
        tens = torch.from_numpy(chw)

        # 写入帧缓存
        self._frame_cache[key] = tens
        if len(self._frame_cache) > self._frame_cache_max:
            self._frame_cache.popitem(last=False)
        return tens

    def _get_frame_count(self, video_path: str) -> int:
        # 获取视频总帧数，用于越界裁剪/填充
        dec = self._get_reader(video_path)
        try:
            if self._backend == 'decord':
                return len(dec)
            else:
                import cv2  # type: ignore
                cnt = int(dec.get(cv2.CAP_PROP_FRAME_COUNT))
                return max(0, cnt)
        except Exception:
            # 兜底：无法获取帧数时返回一个较大的上界，后续仍会经由读取失败抛错
            return 0

    def __del__(self):
        # 释放 opencv 解码器
        try:
            if getattr(self, '_decoder_cache', None):
                for _, dec in list(self._decoder_cache.items()):
                    try:
                        if hasattr(dec, 'release'):
                            dec.release()
                    except Exception:
                        pass
        except Exception:
            pass

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        item["observation"] = {}
        # 读取与转换键值
        top_path = self._as_py(item.get("top_video_path")) if "top_video_path" in item else None
        top_idx = self._as_py(item.get("top_frame_idx")) if "top_frame_idx" in item else None
        wrist_path = self._as_py(item.get("wrist_video_path")) if "wrist_video_path" in item else None
        wrist_idx = self._as_py(item.get("wrist_frame_idx")) if "wrist_frame_idx" in item else None

        # 若存在引用字段，则在线解码
        if top_path is not None and top_idx is not None:
            # 当前帧 t
            t_top = int(top_idx)
            # 读取中心帧（保持兼容：单帧键）
            item["observation"]["top_image"] = self._read_frame(str(top_path), t_top)

            # 构造窗口序列 [t-past, ..., t+future]
            if (self.past_img_num + self.future_img_num) > 0:
                total = self._get_frame_count(str(top_path))
                win = []
                if total <= 0:
                    # 无法获得帧数时，保守仅返回中心帧重复
                    for _ in range(self.past_img_num + self.future_img_num + 1):
                        win.append(item["observation"]["top_image"]) 
                else:
                    start = t_top - self.past_img_num
                    end = t_top + self.future_img_num
                    for t in range(start, end + 1):
                        tt = min(max(0, t), total - 1)
                        win.append(self._read_frame(str(top_path), tt))
                item["observation"]["top_image_seq"] = torch.stack(win, dim=0)  # [S, C, H, W]

            # 删除原始字符串/索引，避免 DataLoader 默认 collate 产生字符串批次
            item.pop("top_video_path", None)
            item.pop("top_frame_idx", None)
        if wrist_path is not None and wrist_idx is not None:
            t_wrist = int(wrist_idx)
            item["observation"]["wrist_image"] = self._read_frame(str(wrist_path), t_wrist)

            if (self.past_img_num + self.future_img_num) > 0:
                total = self._get_frame_count(str(wrist_path))
                win = []
                if total <= 0:
                    for _ in range(self.past_img_num + self.future_img_num + 1):
                        win.append(item["observation"]["wrist_image"]) 
                else:
                    start = t_wrist - self.past_img_num
                    end = t_wrist + self.future_img_num
                    for t in range(start, end + 1):
                        tt = min(max(0, t), total - 1)
                        win.append(self._read_frame(str(wrist_path), tt))
                item["observation"]["wrist_image_seq"] = torch.stack(win, dim=0)  # [S, C, H, W]

            item.pop("wrist_video_path", None)
            item.pop("wrist_frame_idx", None)
        
        item["observation"]["state"] = torch.tensor(item["state"], dtype=torch.float32)
        item.pop("state", None)

        # batch = {
        #     'observation': {
        #         'top_image': item["top_image"],
        #         'wrist_image': item["wrist_image"],
        #         'state':item['state'],
        #     },
        #     'actions':item['action'],
        #     'prompt':item['prompt'],
        #     'velocities':item['velocities'],
        #     'efforts':item['efforts'],
        #     'sucker':item['sucker'],
        #     ...
        # }

        return item


def load_lerobot_dataloader(
    repo_id: str = "",
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
    require_train_only_norm: bool = False,
    cache_train_norm_by_split: bool = True,
    past_img_num: int = 0,
    future_img_num: int = 0,
    image_size: int = 0
) -> Tuple[DataLoader, DataLoader, object, object]:
    """
    - 检查/下载数据集（Hugging Face Hub）
    - 按 episode 比例切分训练/验证
    - 依据 horizon 生成 action chunk（t..t+h-1）
    - 基于 past_img_num/future_img_num 提取图像序列窗口（t-past..t+future）
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

    # 使用全量数据集 + Subset 基于帧范围按 episode 划分，确保 horizon>0 时不越界
    LeRobot = type(base_ds)
    full_ds = LeRobot(
        repo_id=repo_id,
        root=str(local_root),
        episodes=None,  # 全量 episodes，episode_index 为全局索引
        delta_timestamps=delta_ts,
    )

    # 基于 episode 的帧范围构造索引
    ep_from = full_ds.episode_data_index["from"].tolist()
    ep_to = full_ds.episode_data_index["to"].tolist()

    def _build_indices(selected_eps):
        idxs = []
        for ep in selected_eps:
            start, end = ep_from[ep], ep_to[ep]
            if end > start:
                idxs.extend(range(start, end))
        return idxs

    from torch.utils.data import Subset
    train_indices = _build_indices(train_eps)
    val_indices = _build_indices(val_eps)
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices)

    # 归一化
    if normalize:
        # 默认在 datasets/<repo_name>/train_only/norm_stats.json（仅用训练集统计，避免泄漏）
        from normalize import RunningStats, save as save_norm_stats

        repo_name = repo_id.split("/")[-1]
        default_norm_dir = Path.cwd() / "datasets" / repo_name
        base_norm_dir = Path(norm_stats_dir) if norm_stats_dir else default_norm_dir
        # 基于训练 episode 列表生成可复用的划分缓存目录（避免不同划分重复计算）
        if cache_train_norm_by_split:
            split_key = json.dumps(sorted(train_eps), separators=(",", ":"))
            split_hash = hashlib.sha1(split_key.encode("utf-8")).hexdigest()[:8]
            train_norm_dir = base_norm_dir / "train_only" / f"split-{split_hash}"
        else:
            train_norm_dir = base_norm_dir / "train_only"

        def _compute_train_norm_stats(ds) -> Dict[str, object]:
            stats_acc: Dict[str, RunningStats] = {"state": RunningStats(), "action": RunningStats()}
            total = len(ds)
            for i in range(total):
                sample = ds[i]
                for k, acc in stats_acc.items():
                    if k not in sample:
                        continue
                    arr = np.asarray(sample[k])
                    if arr.ndim == 0:
                        arr = arr.reshape(1, 1)
                    elif arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    else:
                        arr = arr.reshape(-1, arr.shape[-1])
                    acc.update(arr)
            # 转为可保存/可加载的结构
            return {k: acc.get_statistics() for k, acc in stats_acc.items()}

        # 若已存在训练集统计则直接加载；若要求必须存在但缺失，则报错；否则计算并缓存
        try:
            norm_stats = load_norm(train_norm_dir)
        except FileNotFoundError:
            if require_train_only_norm:
                raise
            train_norm_dir.mkdir(parents=True, exist_ok=True)
            computed = _compute_train_norm_stats(train_ds)
            save_norm_stats(train_norm_dir, computed)
            norm_stats = load_norm(train_norm_dir)

        train_ds = _NormalizeWrapper(train_ds, norm_stats)
        val_ds = _NormalizeWrapper(val_ds, norm_stats)

    # 将视频引用解码为图像张量/序列，供下游直接消费
    # 如果样本本身已是 'top_image'/'wrist_image'，该包装不会产生多余开销
    try:
        # 尺寸不强制缩放（沿用原始尺寸）
        train_ds = _DecodeVideoRefs(
            train_ds,
            local_dir,
            image_size=image_size,
            decoder_cache_max=32,
            frame_cache_max=512,
            past_img_num=past_img_num,
            future_img_num=future_img_num,
        )
        val_ds = _DecodeVideoRefs(
            val_ds,
            local_dir,
            image_size=image_size,
            decoder_cache_max=32,
            frame_cache_max=512,
            past_img_num=past_img_num,
            future_img_num=future_img_num,
        )
    except Exception:
        # 出错时回退为原数据集（不影响训练的其余部分）
        pass

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
        "--repo-id", default="", help="HF 数据集仓库 ID"
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
    parser.add_argument(
        "--require-train-only-norm",
        action="store_true",
        help="若缺少训练集统计则报错（不在线计算）",
    )
    parser.add_argument(
        "--no-cache-train-norm-by-split",
        action="store_true",
        help="不按当前训练划分建立独立缓存目录（始终写入 train_only/）",
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
            require_train_only_norm=args.require_train_only_norm,
            cache_train_norm_by_split=not args.no_cache_train_norm_by_split,
        )
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
