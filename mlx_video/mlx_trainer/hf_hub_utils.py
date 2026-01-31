from __future__ import annotations

from pathlib import Path
from typing import Optional


def push_to_hub(model_dir: Path, repo_id: str, token: Optional[str] = None, message: str = "Upload") -> None:
    """Upload a folder (e.g., checkpoints) to the Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to push to hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        path_in_repo=".",
        commit_message=message,
    )
