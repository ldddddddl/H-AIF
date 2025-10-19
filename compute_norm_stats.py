import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from tqdm import tqdm

from normalize import RunningStats, save as save_norm_stats


def _iter_dataset(dataset, max_samples: Optional[int] = None) -> Iterable[dict]:
    """
    逐条读取数据集样本，避免对 collate_fn 的依赖，保证兼容性。
    """
    total = len(dataset)
    limit = min(total, max_samples) if (max_samples is not None and max_samples > 0) else total
    for idx in range(limit):
        yield dataset[idx]


def compute_norm_stats(
    repo_id: str = "lddddl/jetmax_dataset_v4",
    local_dir: Optional[str] = None,
    keys: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    out_dir: Optional[str] = None,
) -> Path:
    """
    - 从本地/Hub 获取 LeRobot 数据集
    - 遍历样本，针对指定键（默认 state, action）计算归一化统计
    - 将结果写入 out_dir/repo_id/norm_stats.json
    返回保存目录路径
    """
    if keys is None:
        keys = ["state", "action"]

    # 延迟导入，避免无依赖环境报错
    try:
        import load_jetmax_dataset as loader
    except Exception as exc:
        raise ImportError("请确保同目录存在 load_jetmax_dataset.py 并可导入") from exc

    # 获取数据集（直接取 dataset，避免 DataLoader 的混合类型 batch）
    local_root = loader.ensure_dataset_local(repo_id=repo_id, local_dir=local_dir)
    dataset = loader._instantiate_lerobot_dataset(local_root=local_root, repo_id=repo_id, split=None)

    stats = {k: RunningStats() for k in keys}

    with tqdm(total=min(len(dataset), max_samples) if max_samples else len(dataset), desc="Computing norm stats") as pbar:
        for sample in _iter_dataset(dataset, max_samples=max_samples):
            for k in keys:
                if k not in sample:
                    continue
                arr = np.asarray(sample[k])
                if arr.ndim == 0:
                    arr = arr.reshape(1, 1)
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                else:
                    # 展平成 (N, C)
                    arr = arr.reshape(-1, arr.shape[-1])
                stats[k].update(arr)
            pbar.update(1)

    norm_stats = {k: s.get_statistics() for k, s in stats.items()}

    # 保存路径：assets/<repo_id>/norm_stats.json（支持层级目录）
    if out_dir is None:
        out_path = Path.cwd() / "datasets" / repo_id.split("/")[-1]
    else:
        out_path = Path(out_dir) / repo_id.split("/")[-1]
    save_norm_stats(out_path, norm_stats)
    return out_path


def _parse_cli_args():
    import argparse

    parser = argparse.ArgumentParser(description="计算 LeRobot 数据集的归一化统计（mean/std/q01/q99）")
    parser.add_argument("--repo-id", default="lddddl/jetmax_dataset_v4", help="HF 数据集仓库 ID")
    parser.add_argument("--local-dir", default=None, help="本地数据集根目录（默认 datasets/<repo_id>）")
    parser.add_argument("--keys", default="state,action", help="要统计的键，逗号分隔，例如 state,action,velocities")
    parser.add_argument("--max-samples", type=int, default=None, help="最多使用多少条样本进行统计（默认全部）")
    parser.add_argument("--out-dir", default=None, help="norm_stats.json 输出目录（最终为 <out-dir>/<repo_id>/norm_stats.json）")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    try:
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        out_dir = compute_norm_stats(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            keys=keys,
            max_samples=args.max_samples,
            out_dir=args.out_dir,
        )
        print(f"归一化统计已保存到: {out_dir / 'norm_stats.json'}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


