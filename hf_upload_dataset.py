import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login

os.environ["HF_TOKEN"] = ""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local dataset folder to Hugging Face Hub (dataset repo)."
    )
    parser.add_argument(
        "--repo-id",
        default="lddddl/jetmax_dataset_v4",
        help="Target dataset repo id on Hugging Face, e.g., 'username/dataset-name'.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="./datasets/v4_lerobot",
        help="Local dataset folder to upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the dataset repo as private. Omit to make it public.",
        default=True,
    )
    parser.add_argument(
        "--path-in-repo",
        default=".",
        help="Path inside the repo where files will be placed. Default is repo root.",
    )
    parser.add_argument(
        "--commit-message",
        default="Initial dataset upload",
        help="Commit message for this upload.",
    )
    parser.add_argument(
        "--ignore-patterns",
        nargs="*",
        default=["*.tmp", "*.log", "__pycache__/", ".DS_Store", "Thumbs.db"],
        help="Glob patterns to ignore during upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_dir).resolve()
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"[ERROR] Dataset directory not found or not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] Missing HF_TOKEN environment variable. Please set it with your Hugging Face token.", file=sys.stderr)
        sys.exit(1)

    # Non-interactive auth
    login(token=token, add_to_git_credential=False)

    api = HfApi()

    # Ensure repo exists (idempotent)
    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Upload the whole folder (recursively)
    print(f"[INFO] Uploading '{dataset_path}' to 'datasets/{args.repo_id}' (path_in_repo='{args.path_in_repo}') ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(dataset_path),
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns if args.ignore_patterns else None,
    )
    print("[INFO] Upload completed successfully.")


if __name__ == "__main__":
    main()