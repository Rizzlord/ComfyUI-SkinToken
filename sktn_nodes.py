from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
import trimesh as Trimesh

DEFAULT_TOKENRIG_CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
DEFAULT_VAE_CKPT = "experiments/skin_vae_2_10_32768/last.ckpt"
DEFAULT_QWEN_DIR = "Qwen3-0.6B"
HF_REPO_ID = "VAST-AI/SkinTokens"
HF_LLM_REPO = "Qwen/Qwen3-0.6B"

SKELETON_TEMPLATE_KEEP = "original"
SKELETON_TEMPLATE_LABELS = {
    "original": "Keep model names",
    "mixamo": "Mixamo",
    "ue5": "Unreal Engine 5",
}
SKELETON_TEMPLATE_LABEL_CHOICES = list(SKELETON_TEMPLATE_LABELS.values())

MODEL_CACHE_LOCK = threading.Lock()
MODEL_CACHE: dict[str, Any] = {"key": None, "model": None}


def _models_dir() -> Path:
    model_dir = Path(folder_paths.models_dir) / "skintoken"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SKINTOKEN_MODEL_DIR"] = str(model_dir)
    return model_dir


def _refresh_model_cache() -> None:
    folder_paths.filename_list_cache.pop("skintoken", None)


def _asset_to_payload(asset: Any) -> dict[str, Any]:
    return dict(asset.__dict__)


def _download_model_file(relative_path: str) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=relative_path,
        local_dir=str(_models_dir()),
    )
    _refresh_model_cache()
    return Path(local_path)


def _download_qwen_config() -> Path:
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=HF_LLM_REPO,
        local_dir=str(_models_dir() / DEFAULT_QWEN_DIR),
        ignore_patterns=["*.bin", "*.safetensors"],
    )
    return Path(local_path)


def _ensure_file(relative_path: str) -> Path:
    target = _models_dir() / relative_path
    if target.exists():
        return target
    return _download_model_file(relative_path)


def _ensure_qwen() -> Path:
    target = _models_dir() / DEFAULT_QWEN_DIR
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    if target.is_dir() and all((target / name).exists() for name in required_files):
        return target
    return _download_qwen_config()


def _ensure_required_models(ckpt_name: str) -> Path:
    ckpt_path = _ensure_file(ckpt_name)
    vae_path = _ensure_file(DEFAULT_VAE_CKPT)
    qwen_dir = _ensure_qwen()

    missing = []
    if not ckpt_path.exists():
        missing.append(str(ckpt_path))
    if not vae_path.exists():
        missing.append(str(vae_path))
    if not qwen_dir.exists():
        missing.append(str(qwen_dir))
    if missing:
        raise RuntimeError(
            "Missing SkinToken model assets after automatic download attempt under "
            f"{_models_dir()}: {', '.join(missing)}"
        )
    return ckpt_path


def _available_checkpoints() -> list[str]:
    try:
        available = folder_paths.get_filename_list("skintoken")
    except Exception:
        available = []
    ckpts = sorted(path for path in available if path.lower().endswith(".ckpt"))
    if DEFAULT_TOKENRIG_CKPT not in ckpts:
        ckpts.insert(0, DEFAULT_TOKENRIG_CKPT)
    return ckpts or [DEFAULT_TOKENRIG_CKPT]


def _resolve_device(device_name: str) -> str:
    if device_name == "auto":
        try:
            return str(mm.get_torch_device())
        except Exception:
            return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _collate_processed_batch(processed_batch: list[dict[str, Any]]) -> dict[str, Any]:
    tensors_stack: dict[str, list[torch.Tensor]] = {}
    tensors_cat: dict[str, list[torch.Tensor]] = {}
    non_tensors: dict[str, list[Any]] = {}
    seen: set[str] = set()

    def _to_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            return value
        raise TypeError(f"Expected tensor-like value, found {type(value)!r}")

    for key, value in processed_batch[0].items():
        if key == "cat":
            for nested_key in value.keys():
                if nested_key in seen:
                    raise ValueError(f"duplicate batch key: {nested_key}")
                seen.add(nested_key)
                tensors_cat[nested_key] = []
                for item in processed_batch:
                    tensors_cat[nested_key].append(_to_tensor(item["cat"][nested_key]))
        elif key == "non":
            for nested_key in value.keys():
                if nested_key in seen:
                    raise ValueError(f"duplicate batch key: {nested_key}")
                seen.add(nested_key)
                non_tensors[nested_key] = []
                for item in processed_batch:
                    nested_value = item["non"][nested_key]
                    if isinstance(nested_value, np.ndarray):
                        nested_value = torch.from_numpy(nested_value)
                    non_tensors[nested_key].append(nested_value)
        else:
            if key in seen:
                raise ValueError(f"duplicate batch key: {key}")
            seen.add(key)
            tensors_stack[key] = []
            for item in processed_batch:
                tensors_stack[key].append(_to_tensor(item[key]))

    collated_stack = {key: torch.stack(values) for key, values in tensors_stack.items()}
    collated_cat = {key: torch.concat(values, dim=1) for key, values in tensors_cat.items()}
    return {**collated_stack, **collated_cat, **non_tensors}


def _as_single_trimesh(mesh: Any) -> Trimesh.Trimesh:
    if isinstance(mesh, Trimesh.Trimesh):
        return mesh.copy()
    if isinstance(mesh, Trimesh.Scene):
        dumped = mesh.dump(concatenate=True)
        if isinstance(dumped, list):
            meshes = [item for item in dumped if isinstance(item, Trimesh.Trimesh)]
            if not meshes:
                raise ValueError("Scene does not contain any mesh geometry.")
            dumped = Trimesh.util.concatenate(meshes)
        if not isinstance(dumped, Trimesh.Trimesh):
            raise ValueError("Could not convert scene to a single trimesh.")
        return dumped
    raise TypeError(f"Expected TRIMESH input, got {type(mesh)!r}")


def _make_temp_dir() -> Path:
    base_dir = Path(folder_paths.get_temp_directory()) / "skintoken"
    base_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="skintoken_", dir=str(base_dir)))


def _export_input_trimesh(mesh: Trimesh.Trimesh) -> Path:
    temp_dir = _make_temp_dir()
    source_path = temp_dir / "input_mesh.glb"
    mesh.export(source_path, file_type="glb")
    return source_path


def _load_output_trimesh(path: Path) -> Trimesh.Trimesh:
    return _as_single_trimesh(Trimesh.load(path, force="mesh"))


def _make_output_path(filename_prefix: str, file_format: str, save_file: bool) -> Path:
    if save_file:
        full_output_folder, filename, counter, _subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
        )
        output_path = Path(full_output_folder) / f"{filename}_{counter:05}_.{file_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    temp_dir = _make_temp_dir()
    return temp_dir / f"skintoken_result.{file_format}"


def _select_blender_backend() -> str:
    force_headless = os.environ.get("SKINTOKEN_FORCE_HEADLESS", "").lower() in {"1", "true", "yes"}
    if force_headless or sys.version_info[:2] == (3, 12):
        return "blender_headless"
    try:
        import bpy  # type: ignore  # noqa: F401
    except Exception:
        return "blender_headless"
    return "bpy"


def _resolve_blender_binary() -> str:
    configured = os.environ.get("SKINTOKEN_BLENDER_BIN")
    if configured:
        return configured
    discovered = shutil.which("blender")
    if discovered:
        return discovered
    raise RuntimeError("Blender binary was not found. Set SKINTOKEN_BLENDER_BIN to override it.")


def _run_headless_blender(op: str, payload: dict[str, Any]) -> Any:
    temp_dir = _make_temp_dir()
    payload_in = temp_dir / "payload_in.pkl"
    payload_out = temp_dir / "payload_out.pkl"
    with payload_in.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    command = [
        _resolve_blender_binary(),
        "--background",
        "--factory-startup",
        "--python",
        str(Path(__file__).with_name("blender_headless_bridge.py")),
        "--",
        "--op",
        op,
        "--payload-in",
        str(payload_in),
        "--payload-out",
        str(payload_out),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    response = None
    if payload_out.exists():
        with payload_out.open("rb") as handle:
            response = pickle.load(handle)

    if result.returncode != 0:
        error = ""
        if isinstance(response, dict) and response.get("status") == "error":
            error = response.get("error", "")
        raise RuntimeError(
            "Headless Blender bridge failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
            f"{error}"
        )

    if isinstance(response, dict) and response.get("status") == "error":
        raise RuntimeError(response.get("error", "Unknown Blender bridge error."))
    return response


def _export_asset(asset: Any, output_path: Path, use_transfer: bool, group_per_vertex: int, bottom_center_origin: bool = False) -> str:
    backend = _select_blender_backend()
    kwargs = {
        "group_per_vertex": group_per_vertex,
        "bottom_center_origin": bottom_center_origin,
        "export_yup": use_transfer,
    }

    if backend == "bpy":
        from .vendor.skintokens.rig_package.parser.bpy import BpyParser, transfer_rigging

        if use_transfer:
            if not asset.path:
                raise RuntimeError("Transfer export requires the source mesh path on the generated asset.")
            transfer_rigging(
                source_asset=asset,
                target_path=asset.path,
                export_path=str(output_path),
                **kwargs,
            )
        else:
            BpyParser.export(asset, str(output_path), **kwargs)
        return backend

    payload = {"kwargs": kwargs}
    if use_transfer:
        if not asset.path:
            raise RuntimeError("Transfer export requires the source mesh path on the generated asset.")
        payload.update(
            {
                "source_asset": _asset_to_payload(asset),
                "target_path": asset.path,
                "export_path": str(output_path),
            }
        )
        _run_headless_blender("transfer", payload)
    else:
        payload.update({"asset": _asset_to_payload(asset), "filepath": str(output_path)})
        _run_headless_blender("export", payload)
    return backend


def _build_asset(mesh: Trimesh.Trimesh, source_path: Path) -> Any:
    from .vendor.skintokens.rig_package.info.asset import Asset

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("SkinToken requires a mesh with vertices and faces.")

    vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float32)

    return Asset(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
        vertex_bias=np.array([vertices.shape[0]], dtype=np.int32),
        face_bias=np.array([faces.shape[0]], dtype=np.int32),
        mesh_names=["mesh_0"],
        matrix_world=np.eye(4, dtype=np.float32),
        cls="articulation",
        path=str(source_path),
    )


def _load_model(ckpt_name: str, device_name: str) -> Any:
    ckpt_path = _ensure_required_models(ckpt_name)
    device = _resolve_device(device_name)
    if not str(device).startswith("cuda"):
        raise RuntimeError("SkinToken inference currently expects a CUDA device.")

    cache_key = f"{ckpt_path.resolve()}::{device}"
    previous_model = None
    with MODEL_CACHE_LOCK:
        if MODEL_CACHE["key"] == cache_key and MODEL_CACHE["model"] is not None:
            return MODEL_CACHE["model"]
        previous_model = MODEL_CACHE["model"]
        MODEL_CACHE["key"] = None
        MODEL_CACHE["model"] = None

    try:
        from .vendor.skintokens.model.tokenrig import TokenRig
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python dependency `{exc.name}` in the ComfyUI environment. "
            "The node code is wired up, but the upstream SkinTokens runtime dependencies still need to exist in this env."
        ) from exc

    if previous_model is not None:
        del previous_model
    mm.soft_empty_cache()

    model = TokenRig.load_from_system_checkpoint(str(ckpt_path))
    model = model.to(device)
    model.eval()

    with MODEL_CACHE_LOCK:
        MODEL_CACHE["key"] = cache_key
        MODEL_CACHE["model"] = model
    return model


def _prepare_batch(model: Any, asset: Any) -> dict[str, Any]:
    from .vendor.skintokens.model.spec import ModelInput

    predict_transform = model.get_predict_transform()
    if predict_transform is None:
        raise RuntimeError("SkinToken checkpoint does not provide a predict transform.")

    working_asset = asset.copy()
    predict_transform.apply(working_asset)
    processed_batch = model._process_fn([ModelInput(asset=working_asset, tokens=None)])
    return _collate_processed_batch(processed_batch)


def _apply_postprocess(asset: Any) -> None:
    from .vendor.skintokens.data.vertex_group import voxel_skin

    voxel = asset.voxel(resolution=196)
    asset.skin *= voxel_skin(
        grid=0,
        grid_coords=voxel.coords,
        joints=asset.joints,
        vertices=asset.vertices,
        faces=asset.faces,
        mode="square",
        voxel_size=voxel.voxel_size,
    )
    asset.normalize_skin()


def _rename_skeleton(asset: Any, skeleton_template: str) -> None:
    from .vendor.skintokens.rig_package.skeleton_template import (
        apply_asset_joint_name_template,
        normalize_skeleton_template,
    )

    template_key = normalize_skeleton_template(skeleton_template)
    asset.joint_names = apply_asset_joint_name_template(
        joint_names=asset.joint_names,
        joints=asset.joints,
        parents=asset.parents,
        template=template_key,
    )


def _run_skin_token(
    mesh: Trimesh.Trimesh,
    ckpt_name: str,
    device_name: str,
    top_k: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    num_beams: int,
    use_postprocess: bool,
    skeleton_template: str,
) -> Any:
    model = _load_model(ckpt_name, device_name)
    model_device = next(model.parameters()).device
    source_path = _export_input_trimesh(mesh)
    asset = _build_asset(mesh, source_path)
    batch = _prepare_batch(model, asset)
    batch["generate_kwargs"] = {
        "max_length": 2048,
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "repetition_penalty": float(repetition_penalty),
        "num_return_sequences": 1,
        "num_beams": int(num_beams),
        "do_sample": True,
    }
    batch = _move_to_device(batch, model_device)

    with torch.inference_mode():
        results = model.predict_step(batch, make_asset=True)["results"]

    result = results[0]
    if result.asset is None:
        raise RuntimeError("SkinToken did not return a generated rig asset.")

    generated_asset = result.asset
    _rename_skeleton(generated_asset, skeleton_template)
    if use_postprocess:
        _apply_postprocess(generated_asset)
    return generated_asset


class SkinTokenDownloadModels:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "download_tokenrig": ("BOOLEAN", {"default": True}),
                "download_vae": ("BOOLEAN", {"default": True}),
                "download_qwen_config": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "models_dir")
    FUNCTION = "download"
    CATEGORY = "3D/SkinToken"
    DESCRIPTION = "Downloads SkinTokens assets into ComfyUI/models/skintoken."

    def download(self, download_tokenrig: bool, download_vae: bool, download_qwen_config: bool):
        completed = []
        if download_tokenrig:
            path = _ensure_file(DEFAULT_TOKENRIG_CKPT)
            completed.append(str(path))
        if download_vae:
            path = _ensure_file(DEFAULT_VAE_CKPT)
            completed.append(str(path))
        if download_qwen_config:
            path = _ensure_qwen()
            completed.append(str(path))

        if not completed:
            status = "No model assets were selected."
        else:
            status = "Ready:\n" + "\n".join(completed)
        return (status, str(_models_dir()))


class SkinTokenRigTrimesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "ckpt_name": (_available_checkpoints(), {"default": DEFAULT_TOKENRIG_CKPT}),
                "device": (["auto", "cuda"], {"default": "auto"}),
                "save_file": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "3D/SkinToken_"}),
                "file_format": (["glb", "fbx"], {"default": "glb"}),
                "use_transfer": ("BOOLEAN", {"default": False}),
                "use_postprocess": ("BOOLEAN", {"default": False}),
                "group_per_vertex": ("INT", {"default": 4, "min": 1, "max": 32}),
                "bottom_center_origin": ("BOOLEAN", {"default": False}),
                "skeleton_template": (SKELETON_TEMPLATE_LABEL_CHOICES, {"default": SKELETON_TEMPLATE_LABELS[SKELETON_TEMPLATE_KEEP]}),
                "top_k": ("INT", {"default": 5, "min": 1, "max": 200}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 10, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING", "SKINTOKEN_ASSET", "STRING")
    RETURN_NAMES = ("trimesh", "rigged_path", "asset", "backend")
    FUNCTION = "rig"
    CATEGORY = "3D/SkinToken"
    DESCRIPTION = "Runs SkinTokens rigging on a TRIMESH input and exports the result through bpy or headless Blender."

    def rig(
        self,
        trimesh: Any,
        ckpt_name: str,
        device: str,
        save_file: bool,
        filename_prefix: str,
        file_format: str,
        use_transfer: bool,
        use_postprocess: bool,
        group_per_vertex: int,
        bottom_center_origin: bool,
        skeleton_template: str,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        num_beams: int,
    ):
        _ensure_required_models(ckpt_name)

        mesh = _as_single_trimesh(trimesh)
        generated_asset = _run_skin_token(
            mesh=mesh,
            ckpt_name=ckpt_name,
            device_name=device,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
            use_postprocess=use_postprocess,
            skeleton_template=skeleton_template,
        )

        output_path = _make_output_path(filename_prefix, file_format, save_file)
        backend = _export_asset(
            asset=generated_asset,
            output_path=output_path,
            use_transfer=use_transfer,
            group_per_vertex=group_per_vertex,
            bottom_center_origin=bottom_center_origin,
        )
        output_mesh = _load_output_trimesh(output_path)
        return (output_mesh, str(output_path), generated_asset, backend)


NODE_CLASS_MAPPINGS = {
    "SkinTokenDownloadModels": SkinTokenDownloadModels,
    "SkinTokenRigTrimesh": SkinTokenRigTrimesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SkinTokenDownloadModels": "SkinToken Download Models",
    "SkinTokenRigTrimesh": "SkinToken Rig TRIMESH",
}
