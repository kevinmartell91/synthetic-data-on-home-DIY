import json
from pathlib import Path
import string
from typing import Dict, TypeVar, List, Any, Type
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def load_dataset(path: Path) -> List[Dict[str, Any]] | None:
    """Load a JSON dataset into a list of pydantic models"""
    try:
        with open(path, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        print(f"❌ Loading dataset {path} - ERROR: File not found")
        return None

    except json.JSONDecodeError as e:
        print(f"❌ Loading dataset {path} - ERROR: Invalid JSON format - {e}")
        return None

    except Exception as e:
        print(f"❌ Loading dataset {path} - ERROR: {type(e).__name__}: {e}")
        return None


def save_report(data: Any, path: Path | str, filename: str):
    """Save a report under ``path`` (output directory); creates dirs if needed."""
    payload = data.model_dump() if isinstance(data, BaseModel) else data
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / filename
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Saved failure statistics to {out_file}")


def _to_jsonable(data: Any) -> Any:
    """Recursively convert supported payloads into JSON-serializable values."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, dict):
        return {k: _to_jsonable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_to_jsonable(i) for i in data]
    if isinstance(data, tuple):
        return [_to_jsonable(i) for i in data]
    return data


def save_dataset(data: Any, path: Path):
    """Save dataset payloads that can include nested models/lists/dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(data)

    with open(path, "w", encoding="utf-8") as f:
        # json.dump([i.model_dump() for i in data], f, indent=2, ensure_ascii=False)
        json.dump(payload, f, indent=2, ensure_ascii=False)
