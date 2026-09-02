from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_output_directory(
    output: Union[str, Path], readonly_roots: Iterable[Union[str, Path]]
) -> Path:
    """Create output only after proving it is outside every source-data root."""
    output_path = Path(output).resolve()
    for root in readonly_roots:
        root_path = Path(root).resolve()
        if _is_within(output_path, root_path):
            raise ValueError(
                "Refusing to write output inside read-only data root: "
                f"{output_path} is within {root_path}"
            )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
