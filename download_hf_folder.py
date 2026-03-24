import re
import sys
from pathlib import Path
from huggingface_hub import snapshot_download


def parse_hf_dataset_tree_url(url: str):
    """
    Parse URL like:
    https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026/tree/main/skel_2d
    """
    pattern = re.compile(
        r"^https?://huggingface\.co/datasets/([^/]+/[^/]+)/tree/([^/]+)(?:/(.*))?$"
    )
    m = pattern.match(url.strip())
    if not m:
        raise ValueError(f"Unsupported Hugging Face dataset tree URL: {url}")
    repo_id = m.group(1)
    revision = m.group(2)
    subfolder = m.group(3) or ""
    return repo_id, revision, subfolder


def download_dataset_subfolder(url: str, local_root: str):
    repo_id, revision, subfolder = parse_hf_dataset_tree_url(url)

    local_root = Path(local_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = f"{subfolder}/**" if subfolder else None

    print(f"Repo ID    : {repo_id}")
    print(f"Revision   : {revision}")
    print(f"Subfolder  : {subfolder or '[root]'}")
    print(f"Local root : {local_root}")
    print("Downloading...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_root),
        local_dir_use_symlinks=False,
    )

    print("Done.")
    if subfolder:
        print(f"Files should be under: {local_root / subfolder}")
    else:
        print(f"Files should be under: {local_root}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python download_hf_folder.py <hf_dataset_tree_url> <local_root>")
        print("")
        print("Example:")
        print(
            "  python download_hf_folder.py "
            "https://huggingface.co/datasets/tijiang13/FIFA-Skeletal-Tracking-Light-2026/tree/main/skel_2d "
            "./data"
        )
        sys.exit(1)

    url = sys.argv[1]
    local_root = sys.argv[2]
    download_dataset_subfolder(url, local_root)