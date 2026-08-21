#!/usr/bin/env python3
"""Push jobs+qrels to HF dataset repo. Usage: python scripts/push_hf_dataset.py --repo ahmedarfaoui/signalrank-jobs"""
import argparse
import os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=str, default="ahmedarfaoui99/signalrank-jobs")
    args = ap.parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping HF push (dry run)")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=args.repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo, repo_type="dataset",
            folder_path="data",
            path_in_repo="data",
            ignore_patterns=["*.db","__pycache__"],
        )
        # also upload dataset card
        if Path("hf/dataset/README.md").exists():
            api.upload_file(path_or_fileobj="hf/dataset/README.md", path_in_repo="README.md", repo_id=args.repo, repo_type="dataset")
        print(f"Pushed to https://huggingface.co/datasets/{args.repo}")
    except Exception as e:
        print(f"HF push failed (non-fatal, check token permissions): {e}")
        # do not raise — allow CI to continue (Kaggle may have succeeded)
        return

if __name__ == "__main__":
    main()
