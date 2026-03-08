import json
from pathlib import Path
from typing import TypeVar, Type, List, Any
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


def load_dataset(path: Path):
    """Load a JSON dataset into a list of pydantic models"""
    with open(path, "r") as f:
        return json.load(f)


def save_dataset(data: List[Type[T]], path: Path):
    """Save a list of pydantic models to a JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump([i.model_dump() for i in data], f, indent=2)
