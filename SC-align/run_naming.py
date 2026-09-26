"""How run directories are named, shared by the mutation and eval pipelines."""

import re
from datetime import datetime
from typing import Optional

_TIMESTAMP_FORMAT = "%Y-%m-%d__%H-%M-%S"


def run_dir_name(tag: Optional[str] = None) -> str:
    """Build a run directory name: a timestamp, plus the tag if one was given."""
    stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
    if not tag:
        return stamp
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-")
    return f"{stamp}__{safe_tag}" if safe_tag else stamp
