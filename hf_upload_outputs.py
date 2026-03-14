#!/usr/bin/env python
"""
HF Hub uploader for VLMEvalKit/outputs.

사용법:
1) HF 토큰 준비: https://huggingface.co/settings/tokens 에서 write 권한 토큰 생성
2) 터미널에서:
   export HF_TOKEN=your_write_token
3) VLMEvalKit 루트에서:
   python hf_upload.py --repo ch-min/vlmevalkit-outputs --private  # 또는 --public
"""

import argparse
import os
from pathlib import Path

from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi, create_repo, upload_folder, upload_large_folder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HF dataset repo id, e.g. 'ch-min/vlmevalkit-outputs'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (default: public).",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="outputs",
        help="Local folder to upload (default: 'outputs').",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help="Use upload_large_folder (more resilient for many/large files).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 토큰 확인
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set.")

    root = Path(__file__).resolve().parent
    folder_path = root / args.path
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    print(f"[INFO] Using HF repo: {args.repo}")
    print(f"[INFO] Local folder: {folder_path}")
    print(f"[INFO] Visibility  : {'private' if args.private else 'public'}")

    api = HfApi()

    # 2. repo 없으면 생성 (dataset 타입)
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
            token=hf_token,
        )
        print("[INFO] Repo created or already exists.")
    except HfHubHTTPError as e:
        print(f"[WARN] create_repo failed (maybe already exists with different type?): {e}")

    if args.large:
        print("[INFO] Using upload_large_folder(...)")
        upload_large_folder(
            folder_path=str(folder_path),
            repo_id=args.repo,
            repo_type="dataset",
            # token=hf_token,  # <-- 이 줄 삭제
            # num_workers=4,
            # max_workers=8,
        )
    else:
        print("[INFO] Using upload_folder(...)")
        upload_folder(
            folder_path=str(folder_path),
            repo_id=args.repo,
            repo_type="dataset",
            token=hf_token,  # upload_folder는 token 인자 허용
        )

    print("[INFO] Upload completed.")


if __name__ == "__main__":
    main()
