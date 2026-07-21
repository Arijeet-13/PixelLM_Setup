from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="liuhaotian/LLaVA-Lightning-7B-v1-1",
    local_dir="./llava-7b",
    local_dir_use_symlinks=False  # Downloads actual files instead of symlinks
)