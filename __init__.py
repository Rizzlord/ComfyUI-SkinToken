from __future__ import annotations

import os

import folder_paths

SKINTOKEN_MODEL_DIR = os.path.join(folder_paths.models_dir, "skintoken")
os.makedirs(SKINTOKEN_MODEL_DIR, exist_ok=True)
os.environ["SKINTOKEN_MODEL_DIR"] = SKINTOKEN_MODEL_DIR

existing = folder_paths.folder_names_and_paths.get("skintoken")
if existing is None:
    folder_paths.folder_names_and_paths["skintoken"] = (
        [SKINTOKEN_MODEL_DIR],
        set(folder_paths.supported_pt_extensions),
    )
else:
    folder_paths.add_model_folder_path("skintoken", SKINTOKEN_MODEL_DIR, is_default=True)
    paths = folder_paths.get_folder_paths("skintoken")
    folder_paths.folder_names_and_paths["skintoken"] = (
        paths,
        set(folder_paths.supported_pt_extensions),
    )

folder_paths.filename_list_cache.pop("skintoken", None)

from .sktn_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
