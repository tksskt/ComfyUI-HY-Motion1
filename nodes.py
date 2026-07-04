import os
import sys
import uuid
import time
import numpy as np
import model_management
import torch
from typing import Optional, List, Dict, Any

import comfy.model_management as model_management

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "output")

HYMOTION_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "HY-Motion")
try:
    folder_paths.add_model_folder_path("hymotion_gguf",
        os.path.join(HYMOTION_MODELS_DIR, "ckpts", "GGUF"))
    folder_paths.add_model_folder_path("hymotion_gguf",
        os.path.join(COMFY_MODELS_DIR, "llm", "GGUF"))
    folder_paths.add_model_folder_path("hymotion_gguf",
        os.path.join(COMFY_MODELS_DIR, "LLM", "GGUF"))

    if "hymotion_gguf" in folder_paths.folder_names_and_paths:
        paths = folder_paths.folder_names_and_paths["hymotion_gguf"][0]
        folder_paths.folder_names_and_paths["hymotion_gguf"] = (paths, {".gguf"})
except Exception:
    pass

_BUILTIN_CONFIGS = {
    "HY-Motion-1.0": {
        "network_module": "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT",
        "network_module_args": {
            "apply_rope_to_single_branch": False, "ctxt_input_dim": 4096,
            "dropout": 0.0, "feat_dim": 1280, "input_dim": 201,
            "mask_mode": "narrowband", "mlp_ratio": 4.0,
            "num_heads": 20, "num_layers": 27,
            "time_factor": 1000.0, "vtxt_input_dim": 768,
        },
    },
    "HY-Motion-1.0-Lite": {
        "network_module": "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT",
        "network_module_args": {
            "apply_rope_to_single_branch": False, "ctxt_input_dim": 4096,
            "dropout": 0.0, "feat_dim": 1024, "input_dim": 201,
            "mask_mode": "narrowband", "mlp_ratio": 4.0,
            "num_heads": 16, "num_layers": 18,
            "time_factor": 1000.0, "vtxt_input_dim": 768,
        },
    },
}

_CLIP_VARIANTS = [
    "clip-vit-large-patch14",
    "clip-vit-large-patch 14",
    "CLIP-ViT-L-14",
    "clip-vit-large-patch14-v2",
]


def _find_clip_dir():
    for variant in _CLIP_VARIANTS:
        test_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", variant)
        if os.path.exists(test_path):
            return test_path

    try:
        for folder in folder_paths.get_folder_paths("text_encoders"):
            if not os.path.exists(folder):
                continue
            for variant in _CLIP_VARIANTS:
                test_path = os.path.join(folder, variant)
                if os.path.exists(test_path):
                    return test_path
            for sub in os.listdir(folder):
                sub_path = os.path.join(folder, sub)
                if os.path.isdir(sub_path) and sub in _CLIP_VARIANTS:
                    return sub_path
    except Exception:
        pass

    try:
        for folder in folder_paths.get_folder_paths("clip_vision"):
            if not os.path.exists(folder):
                continue
            for variant in _CLIP_VARIANTS:
                test_path = os.path.join(folder, variant)
                if os.path.exists(test_path):
                    return test_path
    except Exception:
        pass

    return None


def _scan_llm_dirs():
    """Scan for Qwen3 LLM directories in multiple locations."""
    found = []
    seen = set()

    def _add(name):
        if name not in seen:
            seen.add(name)
            found.append(name)

    qwen_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts")
    if os.path.exists(qwen_dir):
        for f in os.listdir(qwen_dir):
            full_path = os.path.join(qwen_dir, f)
            if os.path.isdir(full_path):
                fl = f.lower()
                if "qwen3" in fl or "qwen-3" in fl or "bnb-4bit" in fl or "awq" in fl:
                    _add(f)

    try:
        llm_dir = os.path.join(COMFY_MODELS_DIR, "LLM")
        if os.path.exists(llm_dir):
            for f in os.listdir(llm_dir):
                full_path = os.path.join(llm_dir, f)
                if os.path.isdir(full_path):
                    fl = f.lower()
                    if "qwen3" in fl or "qwen-3" in fl:
                        _add(f)
    except Exception:
        pass

    return found


def _scan_gguf_files():
    """Scan for GGUF files using folder_paths (supports UI refresh + cache invalidation)."""
    try:
        return folder_paths.get_filename_list("hymotion_gguf")
    except Exception:
        found = []
        for d in [
            os.path.join(HYMOTION_MODELS_DIR, "ckpts", "GGUF"),
            os.path.join(COMFY_MODELS_DIR, "llm", "GGUF"),
            os.path.join(COMFY_MODELS_DIR, "LLM", "GGUF"),
        ]:
            if os.path.exists(d):
                for root, _dirs, files in os.walk(d):
                    for f in files:
                        if f.lower().endswith(".gguf") and f not in found:
                            found.append(f)
        return found


_GGUF_QWEN35_ARCHITECTURES = {"qwen35", "qwen3_5"}


def _resolve_gguf_path(gguf_file):
    if gguf_file == "(select file)":
        raise ValueError("Please select a GGUF file")

    gguf_file = str(gguf_file)
    if os.path.isabs(gguf_file) and os.path.exists(gguf_file):
        return gguf_file

    gguf_path = None
    try:
        gguf_path = folder_paths.get_full_path("hymotion_gguf", gguf_file)
    except Exception:
        pass

    if gguf_path and os.path.exists(gguf_path):
        return gguf_path

    raise FileNotFoundError(
        f"GGUF file not found: {gguf_file}\n"
        f"Searched locations:\n"
        f"  1. {os.path.join(HYMOTION_MODELS_DIR, 'ckpts', 'GGUF')}\n"
        f"  2. {os.path.join(COMFY_MODELS_DIR, 'llm', 'GGUF')}\n"
        f"  3. {os.path.join(COMFY_MODELS_DIR, 'LLM', 'GGUF')}"
    )


def _read_gguf_architecture(gguf_path):
    """Return GGUF general.architecture when it can be read without extra deps."""
    import struct

    scalar_sizes = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 4,
        5: 4,
        6: 4,
        7: 1,
        10: 8,
        11: 8,
        12: 8,
    }

    def _read_exact(f, size):
        data = f.read(size)
        if len(data) != size:
            raise EOFError("Unexpected end of GGUF metadata")
        return data

    def _read_u32(f):
        return struct.unpack("<I", _read_exact(f, 4))[0]

    def _read_u64(f):
        return struct.unpack("<Q", _read_exact(f, 8))[0]

    def _read_string(f):
        length = _read_u64(f)
        if length > 16 * 1024 * 1024:
            raise ValueError(f"Unreasonable GGUF string length: {length}")
        return _read_exact(f, length).decode("utf-8", errors="replace")

    def _skip_value(f, value_type):
        if value_type in scalar_sizes:
            f.seek(scalar_sizes[value_type], os.SEEK_CUR)
            return
        if value_type == 8:
            length = _read_u64(f)
            f.seek(length, os.SEEK_CUR)
            return
        if value_type == 9:
            item_type = _read_u32(f)
            count = _read_u64(f)
            if item_type in scalar_sizes:
                f.seek(scalar_sizes[item_type] * count, os.SEEK_CUR)
                return
            if item_type == 8:
                for _ in range(count):
                    length = _read_u64(f)
                    f.seek(length, os.SEEK_CUR)
                return
            for _ in range(count):
                _skip_value(f, item_type)
            return
        raise ValueError(f"Unsupported GGUF metadata value type: {value_type}")

    try:
        with open(gguf_path, "rb") as f:
            if _read_exact(f, 4) != b"GGUF":
                return None
            _read_u32(f)
            _read_u64(f)
            metadata_kv_count = _read_u64(f)
            for _ in range(metadata_kv_count):
                key = _read_string(f)
                value_type = _read_u32(f)
                if key == "general.architecture":
                    if value_type == 8:
                        return _read_string(f)
                    _skip_value(f, value_type)
                    return None
                _skip_value(f, value_type)
    except Exception as e:
        print(f"[HY-Motion] Could not read GGUF architecture metadata from {gguf_path}: {e}")
    return None


def _guard_transformers_gguf_architecture(gguf_path):
    architecture = _read_gguf_architecture(gguf_path)
    if architecture:
        normalized = architecture.strip().lower().replace("-", "_").replace(".", "_")
        if normalized in _GGUF_QWEN35_ARCHITECTURES:
            raise ValueError(
                f"Qwen3.5 GGUF detected (architecture={architecture}). "
                "Transformers GGUF loader does not support qwen35/qwen3_5. "
                "This is not a generation_config.json issue. Use the experimental llama.cpp backend "
                "or use Qwen3-8B AWQ/bnb-4bit."
            )
    return architecture

def _scan_hymotion_networks():
    """Scan for HY-Motion network models in multiple locations.
    Returns list of model names."""
    found = []
    seen = set()

    tencent_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "tencent")
    if os.path.exists(tencent_dir):
        for name in os.listdir(tencent_dir):
            model_dir = os.path.join(tencent_dir, name)
            config_path = os.path.join(model_dir, "config.yml")
            if os.path.isdir(model_dir) and os.path.exists(config_path):
                if name not in seen:
                    seen.add(name)
                    found.append(name)

    for folder_type in ("checkpoints", "diffusion_models"):
        try:
            for filename in folder_paths.get_filename_list(folder_type):
                base = os.path.splitext(os.path.basename(filename))[0]
                if base in _BUILTIN_CONFIGS and base not in seen:
                    seen.add(base)
                    found.append(base)
        except Exception:
            pass

    return found


def _resolve_network_model(model_name):
    """Resolve a model name to (config_dict, ckpt_path, stats_dir).
    Searches tencent directory first, then ComfyUI native folders."""
    import yaml

    model_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "tencent", model_name)
    config_path = os.path.join(model_dir, "config.yml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        ckpt_path = os.path.join(model_dir, "latest.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        stats_dir = os.path.join(model_dir, "stats")
        if not os.path.exists(stats_dir):
            stats_dir = None
        return config, ckpt_path, stats_dir

    if model_name in _BUILTIN_CONFIGS:
        config = _BUILTIN_CONFIGS[model_name]
        for folder_type in ("checkpoints", "diffusion_models"):
            try:
                for filename in folder_paths.get_filename_list(folder_type):
                    base = os.path.splitext(os.path.basename(filename))[0]
                    if base == model_name:
                        ckpt_path = folder_paths.get_full_path(folder_type, filename)
                        if ckpt_path:
                            return config, ckpt_path, None
            except Exception:
                pass

    return None, None, None


def get_timestamp():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"

def _get_device_choices(include_cpu=True):
    choices = ["default"]
    if include_cpu:
        choices.append("cpu")
    if torch.cuda.is_available():
        try:
            choices.extend(f"cuda:{i}" for i in range(torch.cuda.device_count()))
        except Exception:
            pass
    return choices


def _resolve_device(device_name="default"):
    if isinstance(device_name, torch.device):
        return device_name

    if device_name is None or device_name == "" or device_name == "default":
        return model_management.get_torch_device()

    device_name = str(device_name)
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"[HY-Motion] CUDA device requested but CUDA is not available: {device_name}")
        try:
            device_index = int(device_name.split(":", 1)[1])
        except (IndexError, ValueError):
            raise ValueError(f"[HY-Motion] Invalid CUDA device format: {device_name}. Use cuda:N.")
        device_count = torch.cuda.device_count()
        if device_index < 0 or device_index >= device_count:
            raise ValueError(
                f"[HY-Motion] Requested {device_name}, but only {device_count} CUDA device(s) are visible. "
                "Check CUDA_VISIBLE_DEVICES or ComfyUI --cuda-device settings."
            )
        return torch.device(device_name)

    raise ValueError(
        f"[HY-Motion] Unsupported device '{device_name}'. "
        f"Available choices: {', '.join(_get_device_choices())}"
    )


def _cuda_device_index(device):
    if device.type != "cuda":
        return None
    if device.index is not None:
        return device.index
    return torch.cuda.current_device()


def _get_module_device(module, fallback=None):
    try:
        for param in module.parameters():
            if param.device.type != "meta":
                return param.device
    except Exception:
        pass
    return fallback


def _format_gib(value):
    return f"{value / (1024 ** 3):.1f}GiB"


def _log_memory_status(label, device=None):
    parts = []
    try:
        import psutil
        mem = psutil.virtual_memory()
        proc_mem = psutil.Process(os.getpid()).memory_full_info()
        parts.append(f"system_total={_format_gib(mem.total)}")
        parts.append(f"available_ram={_format_gib(mem.available)}")
        parts.append(f"RSS={_format_gib(proc_mem.rss)}")
        if hasattr(proc_mem, "uss"):
            parts.append(f"USS={_format_gib(proc_mem.uss)}")
        if hasattr(proc_mem, "pss"):
            parts.append(f"PSS={_format_gib(proc_mem.pss)}")
    except Exception as e:
        parts.append(f"memory_info_unavailable={e}")

    try:
        if torch.cuda.is_available():
            cuda_device = device if isinstance(device, torch.device) and device.type == "cuda" else torch.device(f"cuda:{torch.cuda.current_device()}")
            parts.append(f"gpu_allocated={_format_gib(torch.cuda.memory_allocated(cuda_device))}")
            parts.append(f"gpu_reserved={_format_gib(torch.cuda.memory_reserved(cuda_device))}")
    except Exception as e:
        parts.append(f"gpu_memory_unavailable={e}")

    print(f"[HY-Motion] {label}: " + ", ".join(parts))


def _aggressive_cpu_cleanup(reason=""):
    import gc
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass

    if reason:
        print(f"[HY-Motion] CPU cleanup completed: {reason}")


def _summarize_module_devices(module):
    param_counts = {}
    buffer_counts = {}
    try:
        for param in module.parameters(recurse=True):
            key = str(param.device)
            param_counts[key] = param_counts.get(key, 0) + 1
    except Exception as e:
        param_counts[f"error:{e}"] = 1

    try:
        for buffer in module.buffers(recurse=True):
            key = str(buffer.device)
            buffer_counts[key] = buffer_counts.get(key, 0) + 1
    except Exception as e:
        buffer_counts[f"error:{e}"] = 1

    return param_counts, buffer_counts


def _device_map_has_cpu_or_disk(device_map):
    if device_map is None:
        return False
    if isinstance(device_map, dict):
        values = device_map.values()
    else:
        values = [device_map]
    for value in values:
        nested_values = value if isinstance(value, (list, tuple, set)) else [value]
        for item in nested_values:
            item_s = str(item).lower()
            if item_s == "cpu" or item_s == "disk" or item_s.startswith("disk"):
                return True
    return False


def _validate_llm_gpu_only(model, strict_gpu_only, label="LLM"):
    hf_device_map = getattr(model, "hf_device_map", None)
    print(f"[HY-Motion] {label} hf_device_map={hf_device_map}")
    param_counts, buffer_counts = _summarize_module_devices(model)
    print(f"[HY-Motion] {label} parameter devices: {param_counts}")
    print(f"[HY-Motion] {label} buffer devices: {buffer_counts}")

    if not strict_gpu_only:
        return

    if _device_map_has_cpu_or_disk(hf_device_map):
        raise RuntimeError(
            f"[HY-Motion] strict_gpu_only=True but model has CPU/disk offload in hf_device_map: {hf_device_map}"
        )

    bad_params = {device: count for device, count in param_counts.items() if not device.startswith("cuda:")}
    bad_buffers = {device: count for device, count in buffer_counts.items() if not device.startswith("cuda:")}
    if bad_params or bad_buffers:
        raise RuntimeError(
            f"[HY-Motion] strict_gpu_only=True but model has non-CUDA params/buffers: "
            f"params={bad_params}, buffers={bad_buffers}"
        )

    print("[HY-Motion] strict_gpu_only=True; CPU params/buffers/offload not detected")


class HYMotionLLMWrapper:
    """LLM model wrapper"""
    def __init__(self, model, tokenizer, llm_type="qwen3", max_length=512, crop_start=0, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.llm_type = llm_type
        self.max_length = max_length
        self.crop_start = crop_start
        self.device = device
        self.hidden_size = model.config.hidden_size if hasattr(model, 'config') else 4096



class HYMotionLlamaCppLLMWrapper:
    """llama.cpp LLM wrapper for experimental HY-Motion conditioning."""
    def __init__(self, llama, gguf_path, n_ctx, n_batch, n_gpu_layers, main_gpu,
                 split_mode, split_mode_value, hidden_size, device=None):
        self.llama = llama
        self.model = llama
        self.tokenizer = None
        self.gguf_path = gguf_path
        self.llm_type = "llama.cpp"
        self.backend_name = "llama.cpp"
        self.backend = "llama.cpp"
        self.n_ctx = int(n_ctx)
        self.n_batch = int(n_batch)
        self.n_gpu_layers = int(n_gpu_layers)
        self.main_gpu = int(main_gpu)
        self.split_mode = split_mode
        self.split_mode_value = split_mode_value
        self.hidden_size = int(hidden_size)
        self.device = device

    def release(self):
        llama = getattr(self, "llama", None)
        self.llama = None
        self.model = None
        if llama is not None:
            try:
                if hasattr(llama, "close"):
                    llama.close()
            except Exception as e:
                print(f"[HY-Motion] llama.cpp release warning: {e}")
            del llama

        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                # llama.cpp may own VRAM outside PyTorch; this helps PyTorch caches only.
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

class HYMotionNetworkWrapper:
    """Diffusion Network wrapper"""
    def __init__(self, network, config, mean, std, body_model=None, device=None):
        self.network = network
        self.config = config
        self.mean = mean
        self.std = std
        self.body_model = body_model
        self.device = device


class HYMotionConditioning:
    """Text encoding result"""
    def __init__(self, vtxt_raw, ctxt_raw, ctxt_length, text: List[str]):
        self.vtxt_raw = vtxt_raw
        self.ctxt_raw = ctxt_raw
        self.ctxt_length = ctxt_length
        self.text = text

    def to_hidden_state_dict(self):
        return {
            "text_vec_raw": self.vtxt_raw,
            "text_ctxt_raw": self.ctxt_raw,
            "text_ctxt_raw_length": self.ctxt_length,
        }


class HYMotionData:
    """Motion output data"""
    def __init__(self, output_dict: Dict[str, Any], text: str, duration: float, seeds: List[int]):
        self.output_dict = output_dict
        self.text = text
        self.duration = duration
        self.seeds = seeds
        self.batch_size = output_dict["keypoints3d"].shape[0] if "keypoints3d" in output_dict else 1


class HYMotionPrompterWrapper:
    """Prompt Rewriter model wrapper"""
    def __init__(self, rewriter):
        self.rewriter = rewriter


# ============================================================================
# Node 1a: HYMotion Load LLM (HuggingFace)
# ============================================================================

class HYMotionLoadLLM:
    """Load Qwen3 LLM with quantization support"""
    @classmethod
    def INPUT_TYPES(s):
        qwen_models = _scan_llm_dirs()
        if not qwen_models:
            qwen_models = ["Qwen3-8B"]
        
        return {
            "required": {
                "model_name": (qwen_models, {"default": qwen_models[0]}),
                "quantization": (["none", "int8", "int4", "bnb-4bit", "awq"], {"default": "none"}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
                "device": (_get_device_choices(), {"default": "default"}),
                "strict_gpu_only": ("BOOLEAN", {"default": True}),
                "fallback_to_cpu": ("BOOLEAN", {"default": False}),
                "allow_cpu_offload": ("BOOLEAN", {"default": False}),
                "allow_transformers_gguf_dequantization": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm"
    CATEGORY = "HY-Motion/Loaders"

    def load_llm(self, model_name="Qwen3-8B", quantization="none", offload_to_cpu=False, device="default", strict_gpu_only=True, fallback_to_cpu=False, allow_cpu_offload=False, allow_transformers_gguf_dequantization=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

        # Search multiple locations for LLM directory
        local_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", model_name)
        if not os.path.exists(local_path):
            # Try ComfyUI native LLM folder
            alt_path = os.path.join(COMFY_MODELS_DIR, "LLM", model_name)
            if os.path.exists(alt_path):
                local_path = alt_path
            else:
                raise FileNotFoundError(
                    f"LLM directory not found. Searched:\n"
                    f"  1. {os.path.join(HYMOTION_MODELS_DIR, 'ckpts', model_name)}\n"
                    f"  2. {alt_path}\n"
                    f"Please download {model_name} and place it in either location."
                )

        if strict_gpu_only and offload_to_cpu:
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids offload_to_cpu=True")
        if strict_gpu_only and fallback_to_cpu:
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids fallback_to_cpu=True")
        if strict_gpu_only and allow_cpu_offload:
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids allow_cpu_offload=True")
        if offload_to_cpu and not allow_cpu_offload:
            raise RuntimeError("[HY-Motion] offload_to_cpu=True requires allow_cpu_offload=True")

        requested_device = device
        resolved_device = torch.device("cpu") if offload_to_cpu else _resolve_device(requested_device)
        if strict_gpu_only and resolved_device.type != "cuda":
            raise RuntimeError(f"[HY-Motion] strict_gpu_only=True requires a CUDA device, got {resolved_device}")
        explicit_device = requested_device not in (None, "", "default")
        print(
            f"[HY-Motion] Loading LLM: {local_path}, model_name={model_name}, "
            f"quantization={quantization}, offload_to_cpu={offload_to_cpu}, "
            f"strict_gpu_only={strict_gpu_only}, fallback_to_cpu={fallback_to_cpu}, "
            f"allow_cpu_offload={allow_cpu_offload}, requested_device={requested_device}, "
            f"resolved_device={resolved_device}"
        )
        _log_memory_status("Before LLM load", resolved_device)

        load_kwargs = {"low_cpu_mem_usage": True, "local_files_only": True}

        def _apply_explicit_device_map():
            if resolved_device.type == "cuda":
                load_kwargs["device_map"] = {"": str(resolved_device)}
            elif resolved_device.type == "cpu":
                load_kwargs["device_map"] = "cpu"

        if offload_to_cpu:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
        elif quantization == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            if explicit_device or strict_gpu_only or not allow_cpu_offload:
                _apply_explicit_device_map()
            else:
                load_kwargs["device_map"] = "auto"
        elif quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if explicit_device or strict_gpu_only or not allow_cpu_offload:
                _apply_explicit_device_map()
            else:
                load_kwargs["device_map"] = "auto"
        elif quantization == "bnb-4bit":
            # For bnb-4bit models
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if explicit_device or strict_gpu_only or not allow_cpu_offload:
                _apply_explicit_device_map()
            else:
                load_kwargs["device_map"] = "auto"
        elif quantization == "awq":
            # For AWQ models
            try:
                from awq import AutoAWQForCausalLM
                print("[HY-Motion] Using AWQ for model loading")
                if explicit_device or strict_gpu_only or not allow_cpu_offload:
                    _apply_explicit_device_map()
                tokenizer = AutoTokenizer.from_pretrained(local_path, padding_side="right", local_files_only=True)
                model = AutoAWQForCausalLM.from_pretrained(local_path, **load_kwargs)
                model = model.eval().requires_grad_(False)
                
                # Compute crop_start
                template = [
                    {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
                    {"role": "user", "content": "{}"},
                ]
                crop_start = self._compute_crop_start(tokenizer, template)

                if explicit_device and "device_map" not in load_kwargs:
                    try:
                        model = model.to(resolved_device)
                    except Exception as e:
                        print(f"[HY-Motion] AWQ move to {resolved_device} failed, keeping loaded device: {e}")

                actual_device = _get_module_device(model, resolved_device)
                _aggressive_cpu_cleanup("after LLM GPU load")
                _log_memory_status("After LLM load", actual_device)
                _validate_llm_gpu_only(model, strict_gpu_only, "LLM")
                wrapper = HYMotionLLMWrapper(model=model, tokenizer=tokenizer, max_length=512, crop_start=crop_start, device=actual_device)
                print(f"[HY-Motion] LLM loaded, hidden_size={wrapper.hidden_size}, device={actual_device}")
                return (wrapper,)
            except ImportError:
                print("[HY-Motion] AWQ not installed, falling back to regular loading")
                if explicit_device or strict_gpu_only or not allow_cpu_offload:
                    _apply_explicit_device_map()
                else:
                    load_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(local_path, padding_side="right", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_path, **load_kwargs)
        model = model.eval().requires_grad_(False)

        # Compute crop_start
        template = [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ]
        crop_start = self._compute_crop_start(tokenizer, template)

        if quantization == "none" and "device_map" not in load_kwargs:
            model = model.to(resolved_device)

        actual_device = _get_module_device(model, resolved_device)
        _aggressive_cpu_cleanup("after LLM GPU load")
        _log_memory_status("After LLM load", actual_device)
        _validate_llm_gpu_only(model, strict_gpu_only, "LLM")
        wrapper = HYMotionLLMWrapper(model=model, tokenizer=tokenizer, max_length=512, crop_start=crop_start, device=actual_device)
        print(f"[HY-Motion] LLM loaded, hidden_size={wrapper.hidden_size}, device={actual_device}")
        return (wrapper,)

    def _compute_crop_start(self, tokenizer, template) -> int:
        def _find_subseq(a, b):
            for i in range(len(a) - len(b) + 1):
                if a[i:i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        msgs = [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": marker}]
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        full_ids = tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        return pos if pos >= 0 else max(0, len(full_ids) - 1)


# ============================================================================
# Node 1b: HYMotion Load LLM GGUF
# ============================================================================

class HYMotionLoadLLMGGUF:
    """Load Qwen3 LLM from GGUF file

    Supports loading GGUF quantized models via:
    1. transformers native GGUF support (recommended, requires transformers>=4.40)

    Place GGUF files in: ComfyUI/models/HY-Motion/ckpts/GGUF/
    """

    @classmethod
    def INPUT_TYPES(s):
        gguf_files = ["(select file)"] + _scan_gguf_files()

        return {
            "required": {
                "gguf_file": (gguf_files, {"default": "(select file)"}),
                "device_strategy": (["gpu", "cpu", "balanced"], {"default": "gpu"}),
                "device": (_get_device_choices(), {"default": "default"}),
                "strict_gpu_only": ("BOOLEAN", {"default": True}),
                "fallback_to_cpu": ("BOOLEAN", {"default": False}),
                "allow_cpu_offload": ("BOOLEAN", {"default": False}),
                "allow_transformers_gguf_dequantization": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm_gguf"
    CATEGORY = "HY-Motion/Loaders"

    def load_llm_gguf(self, gguf_file, device_strategy="gpu", device="default", strict_gpu_only=True, fallback_to_cpu=False, allow_cpu_offload=False, allow_transformers_gguf_dequantization=False):
        # 首先进行紧急内存清理
        print("[HY-Motion] Emergency memory cleanup before loading...")
        import gc
        for _ in range(10):
            gc.collect()
        
        try:
            import torch
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
        except:
            pass
        
        # 检查内存状态
        try:
            import psutil
            mem = psutil.virtual_memory()
            print(f"[HY-Motion] Initial memory status: {mem.available / (1024**3):.1f}GB available out of {mem.total / (1024**3):.1f}GB")
        except:
            pass
        
        # 延迟导入，减少内存使用
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from transformers.utils.hub import cached_file
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION
        
        # -------------------------- Memory optimization --------------------------
        gguf_path = _resolve_gguf_path(gguf_file)
        gguf_architecture = _guard_transformers_gguf_architecture(gguf_path)
        if gguf_architecture:
            print(f"[HY-Motion] GGUF architecture={gguf_architecture}")

        gguf_dir = os.path.dirname(gguf_path)
        gguf_filename = os.path.basename(gguf_path)
        if strict_gpu_only and device_strategy == "cpu":
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids GGUF device_strategy=cpu")
        if strict_gpu_only and fallback_to_cpu:
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids fallback_to_cpu=True")
        if strict_gpu_only and allow_cpu_offload:
            raise RuntimeError("[HY-Motion] strict_gpu_only=True forbids allow_cpu_offload=True")
        if device_strategy == "balanced" and not allow_cpu_offload:
            print("[HY-Motion] balanced requested but allow_cpu_offload=False; using fixed GPU placement")

        requested_device = device
        resolved_device = torch.device("cpu") if device_strategy == "cpu" else _resolve_device(requested_device)
        if strict_gpu_only and resolved_device.type != "cuda":
            raise RuntimeError(f"[HY-Motion] strict_gpu_only=True requires a CUDA device, got {resolved_device}")
        print(
            f"[HY-Motion] Loading LLM from GGUF: {gguf_path}, device_strategy={device_strategy}, "
            f"strict_gpu_only={strict_gpu_only}, fallback_to_cpu={fallback_to_cpu}, "
            f"allow_cpu_offload={allow_cpu_offload}, "
            f"allow_transformers_gguf_dequantization={allow_transformers_gguf_dequantization}, "
            f"requested_device={requested_device}, resolved_device={resolved_device}"
        )
        _log_memory_status("Before LLM load", resolved_device)

        if device_strategy != "cpu" and not allow_transformers_gguf_dequantization:
            print("[HY-Motion] allow_transformers_gguf_dequantization=False; proceeding with Transformers GGUF load because strict validation will run after load.")
            print("[HY-Motion] Temporary CPU RAM growth during GGUF dequantization may still happen; persistent CPU params/buffers/offload will be rejected when strict_gpu_only=True.")
            print("[HY-Motion] For strict GPU loading without persistent CPU-expanded weights, prefer Qwen3-8B-AWQ or Qwen3-8B-bnb-4bit instead of Transformers GGUF loading.")
        tokenizer = None
        tokenizer_loaded = False
        
        # 尝试1: 从GGUF文件所在目录加载tokenizer
        try:
            # 再次清理内存
            gc.collect()
            print("[HY-Motion] Loading tokenizer from GGUF directory...")
            tokenizer = AutoTokenizer.from_pretrained(
                gguf_dir,
                padding_side="right",
                local_files_only=True
            )
            tokenizer_loaded = True
            print(f"[HY-Motion] Tokenizer loaded from GGUF directory: {gguf_dir}")
        except Exception as e:
            print(f"[HY-Motion] Failed to load tokenizer from GGUF directory: {e}")
        
        # 尝试2: 从所有可用的Qwen3目录加载tokenizer
        if not tokenizer_loaded:
            qwen_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts")
            if os.path.exists(qwen_dir):
                for f in os.listdir(qwen_dir):
                    if tokenizer_loaded:
                        break
                    
                    full_path = os.path.join(qwen_dir, f)
                    if os.path.isdir(full_path) and ("qwen3" in f.lower() or "qwen-3" in f.lower()):
                        try:
                            # 再次清理内存
                            gc.collect()
                            print(f"[HY-Motion] Loading tokenizer from: {full_path}...")
                            tokenizer = AutoTokenizer.from_pretrained(
                                full_path,
                                padding_side="right",
                                local_files_only=True
                            )
                            tokenizer_loaded = True
                            print(f"[HY-Motion] Tokenizer loaded from: {full_path}")
                        except Exception as e:
                            print(f"[HY-Motion] Failed to load tokenizer from {full_path}: {e}")
        
        # 尝试3: 从默认的Qwen3-8B目录加载tokenizer
        if not tokenizer_loaded:
            default_tokenizer_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Qwen3-8B")
            if os.path.exists(default_tokenizer_path):
                try:
                    # 再次清理内存
                    gc.collect()
                    print(f"[HY-Motion] Loading tokenizer from default path: {default_tokenizer_path}...")
                    tokenizer = AutoTokenizer.from_pretrained(
                        default_tokenizer_path,
                        padding_side="right",
                        local_files_only=True
                    )
                    tokenizer_loaded = True
                    print(f"[HY-Motion] Tokenizer loaded from default path: {default_tokenizer_path}")
                except Exception as e:
                    print(f"[HY-Motion] Failed to load tokenizer from default path: {e}")
        
        # 清理内存
        gc.collect()
        
        # 如果所有尝试都失败，抛出错误
        if not tokenizer_loaded:
            raise FileNotFoundError(
                f"Tokenizer not found. Please ensure you have a valid Qwen3 tokenizer in any of the following locations:"
                f"1. {gguf_dir}\n"
                f"2. {os.path.join(HYMOTION_MODELS_DIR, 'ckpts')} (any Qwen3 directory)\n"
                f"3. {os.path.join(HYMOTION_MODELS_DIR, 'ckpts', 'Qwen3-8B')} (default)"
            )
        # Check transformers version for native GGUF support
        try:
            import transformers
            tf_version = tuple(map(int, transformers.__version__.split('.')[:2]))
            has_native_gguf = tf_version >= (4, 40)
        except:
            has_native_gguf = False

        if has_native_gguf:
            # Tokenizer已经在前面加载过，这里跳过重复加载
            print("[HY-Motion] Using previously loaded tokenizer")

            # -------------------------- GGUF内存优化参数 --------------------------
            load_kwargs = {
                "gguf_file": gguf_filename,
                "low_cpu_mem_usage": True,
                "dtype": torch.float16,  # 使用新的dtype参数替代已弃用的torch_dtype
                "local_files_only": True,
                "max_memory": None,  # 初始化为None，后面根据设备策略设置
                "device_map": None,  # 初始化为None，后面根据设备策略设置
                # GGUF特定优化
                "quantization_config": None,  # GGUF文件已包含量化信息，无需额外配置
                "trust_remote_code": False,  # GGUF模型不需要远程代码
                # 增强内存优化
                "use_safetensors": False,  # GGUF不需要safetensors
                "attn_implementation": "eager",  # 使用内存效率更高的attention实现
                "use_cache": False,  # 禁用缓存以减少内存使用
                "force_download": False,
                "resume_download": False
            }
            
            device = resolved_device
            
            # -------------------------- 执行前内存清理 --------------------------
            # 加载前先深度清理内存，防止内存碎片化
            print("[HY-Motion] Performing aggressive memory cleanup...")
            
            # 清理PyTorch内存
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
            
            # 清理Python内存
            import gc
            for _ in range(5):
                gc.collect()
            
            # 清理可能的缓存
            if 'torch' in sys.modules:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # 检查当前内存状态
            try:
                import psutil
                mem = psutil.virtual_memory()
                print(f"[HY-Motion] Memory status after cleanup: {mem.available / (1024**3):.1f}GB available out of {mem.total / (1024**3):.1f}GB")
                
                if mem.available < 2 * 1024**3:  # 少于2GB可用内存
                    print("[HY-Motion] WARNING: Very low memory available!")
                    print("[HY-Motion] Attempting emergency memory cleanup...")
                    
                    # 尝试释放更多内存
                    for _ in range(10):
                        gc.collect()
                        torch.cuda.empty_cache()
                        if torch.cuda.is_available():
                            torch.cuda.ipc_collect()
                    
                    mem = psutil.virtual_memory()
                    print(f"[HY-Motion] Memory status after emergency cleanup: {mem.available / (1024**3):.1f}GB available")
            except:
                pass
            
            print("[HY-Motion] Memory cleanup completed")
            
            if device_strategy == "cpu":
                load_kwargs["device_map"] = "cpu"
                load_kwargs["dtype"] = torch.float16
                print("[HY-Motion] CPU mode: Loading model entirely on CPU")
            elif device_strategy == "balanced" and allow_cpu_offload and not strict_gpu_only:
                if device.type == "cuda":
                    device_index = _cuda_device_index(device)
                    load_kwargs["device_map"] = device_index

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(device_index).total_memory
                        gpu_limit_gib = max(1, int((gpu_memory * 0.7) / (1024**3)))
                        cpu_limit_gib = 8
                        load_kwargs["max_memory"] = {
                            device_index: f"{gpu_limit_gib}GiB",
                            "cpu": f"{cpu_limit_gib}GiB"
                        }
                        print(f"[HY-Motion] Balanced mode: using up to {gpu_limit_gib}GiB of GPU memory and {cpu_limit_gib}GiB of CPU memory")
                        print("[HY-Motion] CPU offload is allowed for this load")
                else:
                    load_kwargs["device_map"] = "cpu"
                    load_kwargs["dtype"] = torch.float16
            else:
                if device.type == "cuda":
                    device_index = _cuda_device_index(device)
                    load_kwargs["device_map"] = {"": str(device)}

                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(device_index).total_memory
                        gpu_limit_gib = max(1, int((gpu_memory * 0.9) / (1024**3)))
                        load_kwargs["max_memory"] = {device_index: f"{gpu_limit_gib}GiB"}
                        print(f"[HY-Motion] GPU mode: using up to {gpu_limit_gib}GiB of GPU memory; CPU max_memory omitted")
                        print("[HY-Motion] Forcing fixed GPU device mapping")
                else:
                    load_kwargs["device_map"] = "cpu"
                    load_kwargs["dtype"] = torch.float16
            
            load_kwargs["use_cache"] = False  # 禁用模型缓存，减少内存使用
            
            # 移除不兼容的参数
            incompatible_params = ['use_flash_attention_2', 'rope_scaling', 'rope_theta']
            for param in incompatible_params:
                if param in load_kwargs:
                    del load_kwargs[param]
                    print(f"[HY-Motion] Removed incompatible parameter: {param}")
            
            # 禁用attn_implementation以减少内存使用和兼容性问题
            # 仅在模型加载成功后考虑启用
            
            # -------------------------- 尝试加载模型，带内存错误回退 --------------------------
            try:
                # 加载前深度清理内存
                print("[HY-Motion] Performing deep memory cleanup before model loading...")
                
                # 清理PyTorch内存
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                
                # 清理Python内存
                import gc
                gc.collect()
                
                # 强制垃圾回收
                for _ in range(3):
                    gc.collect()
                
                # 清理numpy内存
                if 'numpy' in sys.modules:
                    import numpy as np
                    # 尝试释放numpy缓存
                    if hasattr(np, 'ndarray'):
                        print("[HY-Motion] Numpy available, memory cleanup complete")
                
                # 加载模型
                print(f"[HY-Motion] Attempting to load model with kwargs: {load_kwargs.keys()}")
                model = AutoModelForCausalLM.from_pretrained(gguf_dir, **load_kwargs)
                model = model.eval().requires_grad_(False)
                
                # 立即清理临时变量和内存
                del load_kwargs  # 释放load_kwargs占用的内存
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                gc.collect()
                
                print("[HY-Motion] Model loaded successfully")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "memory" in str(e).lower():
                    # 内存不足，尝试回退到CPU模式
                    print(f"[HY-Motion] GPU memory error: {e}")
                    print(f"[HY-Motion] CUDA load failed on {device}")
                    if not fallback_to_cpu:
                        print("[HY-Motion] fallback_to_cpu=False; not falling back to CPU.")
                        raise
                    print("[HY-Motion] Falling back to CPU mode...")
                    
                    # 深度清理内存
                    torch.cuda.empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.ipc_collect()
                    import gc
                    gc.collect()
                    
                    # 使用更保守的CPU模式参数重新加载
                    # 获取系统内存信息
                    try:
                        import psutil
                        total_system_memory = psutil.virtual_memory().total
                        
                        # 系统内存充足（32G+），可以使用更多CPU内存
                        cpu_memory_limit = min(int(total_system_memory * 0.4 / (1024**3)), 16)  # 最多使用16GB CPU内存
                        
                        print(f"[HY-Motion] System memory: {total_system_memory / (1024**3):.1f}GB total")
                        print(f"[HY-Motion] Setting CPU memory limit: {cpu_memory_limit}GiB")
                    except ImportError:
                        # psutil不可用，使用默认值
                        print("[HY-Motion] psutil not available, using default CPU memory limit")
                        cpu_memory_limit = 12  # 默认使用12GB CPU内存
                        print(f"[HY-Motion] Setting default CPU memory limit: {cpu_memory_limit}GiB")
                    
                    cpu_load_kwargs = {
                        "gguf_file": gguf_filename,
                        "low_cpu_mem_usage": True,
                        "dtype": torch.float16,
                        "local_files_only": True,
                        "device_map": "cpu",
                        "quantization_config": None,
                        "trust_remote_code": False,
                        "use_safetensors": False,
                        "attn_implementation": "eager",
                        "max_memory": {"cpu": f"{cpu_memory_limit}GiB"}  # 根据系统内存调整CPU内存限制
                    }
                    
                    # 移除不兼容的参数
                    incompatible_params = ['use_flash_attention_2', 'rope_scaling', 'rope_theta']
                    for param in incompatible_params:
                        if param in cpu_load_kwargs:
                            del cpu_load_kwargs[param]
                            print(f"[HY-Motion] Removed incompatible parameter from CPU fallback: {param}")
                    
                    model = AutoModelForCausalLM.from_pretrained(gguf_dir, **cpu_load_kwargs)
                    model = model.eval().requires_grad_(False)
                    
                    # 清理临时变量
                    del cpu_load_kwargs
                    gc.collect()
                    
                    print("[HY-Motion] Model loaded successfully in CPU fallback mode")
                else:
                    # 其他错误，重新抛出
                    raise
            except np._core._exceptions._ArrayMemoryError as e:
                # 捕获numpy内存分配错误
                print(f"[HY-Motion] Numpy memory allocation error: {e}")
                if not fallback_to_cpu:
                    print("[HY-Motion] fallback_to_cpu=False; not falling back to CPU.")
                    raise
                print("[HY-Motion] Falling back to optimized GPU mode...")
                
                # 深度清理内存
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.ipc_collect()
                import gc
                gc.collect()
                
                # 尝试使用优化的GPU模式
                if device_strategy == "gpu" and torch.cuda.is_available() and device.type == "cuda":
                    print("[HY-Motion] Attempting optimized GPU loading...")
                    device_index = _cuda_device_index(device)
                    # 获取系统内存信息
                    try:
                        import psutil
                        total_system_memory = psutil.virtual_memory().total
                        available_system_memory = psutil.virtual_memory().available
                        
                        # 计算合理的内存限制
                        # 强制使用更多GPU内存，减少CPU内存使用
                        cpu_memory_limit = min(int(total_system_memory * 0.2 / (1024**3)), 8)  # 最多使用8GB CPU内存
                        gpu_memory_limit = int(torch.cuda.get_device_properties(device_index).total_memory * 0.9 / (1024**3))  # 使用90% GPU内存
                        
                        print(f"[HY-Motion] System memory: {total_system_memory / (1024**3):.1f}GB total, {available_system_memory / (1024**3):.1f}GB available")
                        print(f"[HY-Motion] Setting memory limits: GPU={gpu_memory_limit}GiB, CPU={cpu_memory_limit}GiB")
                    except ImportError:
                        # psutil不可用，使用默认值
                        print("[HY-Motion] psutil not available, using default memory limits")
                        cpu_memory_limit = 6  # 默认使用6GB CPU内存
                        gpu_memory_limit = int(torch.cuda.get_device_properties(device_index).total_memory * 0.9 / (1024**3))  # 使用90% GPU内存
                        print(f"[HY-Motion] Setting default memory limits: GPU={gpu_memory_limit}GiB, CPU={cpu_memory_limit}GiB")
                    
                    # 强制GPU模式：使用更积极的设备映射策略
                    optimized_load_kwargs = {
                        "gguf_file": gguf_filename,
                        "low_cpu_mem_usage": True,
                        "dtype": torch.float16,
                        "local_files_only": True,
                        "device_map": device_index,  # 强制使用特定GPU设备
                        "quantization_config": None,
                        "trust_remote_code": False,
                        "use_safetensors": False,
                        "attn_implementation": "eager",
                        "max_memory": {
                            device_index: f"{gpu_memory_limit}GiB",
                            "cpu": f"{cpu_memory_limit}GiB"  # 严格限制CPU内存使用
                        },
                        "use_cache": False,
                        "force_download": False,
                        "resume_download": False
                    }
                    
                    # 打印详细信息
                    print("[HY-Motion] Forcing GPU device mapping and reducing CPU memory usage")
                    print("[HY-Motion] This will attempt to perform more operations directly on GPU")
                    
                    # 移除不兼容的参数
                    incompatible_params = ['use_flash_attention_2', 'rope_scaling', 'rope_theta']
                    for param in incompatible_params:
                        if param in optimized_load_kwargs:
                            del optimized_load_kwargs[param]
                            print(f"[HY-Motion] Removed incompatible parameter from optimized GPU fallback: {param}")
                    
                    try:
                        model = AutoModelForCausalLM.from_pretrained(gguf_dir, **optimized_load_kwargs)
                        model = model.eval().requires_grad_(False)
                        del optimized_load_kwargs
                        gc.collect()
                        print("[HY-Motion] Model loaded successfully in optimized GPU mode")
                    except Exception as gpu_e:
                        print(f"[HY-Motion] Optimized GPU mode failed: {gpu_e}")
                        print("[HY-Motion] Falling back to minimal memory mode...")
                        
                        # 使用最小内存模式参数
                        # 获取系统内存信息
                        try:
                            import psutil
                            total_system_memory = psutil.virtual_memory().total
                            
                            # 系统内存充足（32G+），可以使用更多CPU内存
                            cpu_memory_limit = min(int(total_system_memory * 0.3 / (1024**3)), 12)  # 最多使用12GB CPU内存
                            
                            print(f"[HY-Motion] System memory: {total_system_memory / (1024**3):.1f}GB total")
                            print(f"[HY-Motion] Setting minimal CPU memory limit: {cpu_memory_limit}GiB")
                        except ImportError:
                            # psutil不可用，使用默认值
                            print("[HY-Motion] psutil not available, using default minimal CPU memory limit")
                            cpu_memory_limit = 8  # 默认使用8GB CPU内存
                            print(f"[HY-Motion] Setting default minimal CPU memory limit: {cpu_memory_limit}GiB")
                        
                        minimal_load_kwargs = {
                            "gguf_file": gguf_filename,
                            "low_cpu_mem_usage": True,
                            "dtype": torch.float16,
                            "local_files_only": True,
                            "device_map": "cpu",
                            "quantization_config": None,
                            "trust_remote_code": False,
                            "use_safetensors": False,
                            "attn_implementation": "eager",
                            "max_memory": {"cpu": f"{cpu_memory_limit}GiB"},  # 根据系统内存调整CPU内存限制
                            "use_cache": False  # 确保禁用缓存
                        }
                        
                        # 移除不兼容的参数
                        for param in incompatible_params:
                            if param in minimal_load_kwargs:
                                del minimal_load_kwargs[param]
                                print(f"[HY-Motion] Removed incompatible parameter from minimal fallback: {param}")
                        
                        model = AutoModelForCausalLM.from_pretrained(gguf_dir, **minimal_load_kwargs)
                        model = model.eval().requires_grad_(False)
                        
                        # 清理临时变量
                        del minimal_load_kwargs
                        gc.collect()
                        
                        print("[HY-Motion] Model loaded successfully in minimal memory mode")
                else:
                    # 使用最小内存模式参数
                    # 获取系统内存信息
                    try:
                        import psutil
                        total_system_memory = psutil.virtual_memory().total
                        
                        # 系统内存充足（32G+），可以使用更多CPU内存
                        cpu_memory_limit = min(int(total_system_memory * 0.3 / (1024**3)), 12)  # 最多使用12GB CPU内存
                        
                        print(f"[HY-Motion] System memory: {total_system_memory / (1024**3):.1f}GB total")
                        print(f"[HY-Motion] Setting minimal CPU memory limit: {cpu_memory_limit}GiB")
                    except ImportError:
                        # psutil不可用，使用默认值
                        print("[HY-Motion] psutil not available, using default minimal CPU memory limit")
                        cpu_memory_limit = 8  # 默认使用8GB CPU内存
                        print(f"[HY-Motion] Setting default minimal CPU memory limit: {cpu_memory_limit}GiB")
                    
                    minimal_load_kwargs = {
                        "gguf_file": gguf_filename,
                        "low_cpu_mem_usage": True,
                        "dtype": torch.float16,
                        "local_files_only": True,
                        "device_map": "cpu",
                        "quantization_config": None,
                        "trust_remote_code": False,
                        "use_safetensors": False,
                        "attn_implementation": "eager",
                        "max_memory": {"cpu": f"{cpu_memory_limit}GiB"},  # 根据系统内存调整CPU内存限制
                        "use_cache": False  # 确保禁用缓存
                    }
                    
                    # 移除不兼容的参数
                    incompatible_params = ['use_flash_attention_2', 'rope_scaling', 'rope_theta']
                    for param in incompatible_params:
                        if param in minimal_load_kwargs:
                            del minimal_load_kwargs[param]
                            print(f"[HY-Motion] Removed incompatible parameter from minimal fallback: {param}")
                    
                    model = AutoModelForCausalLM.from_pretrained(gguf_dir, **minimal_load_kwargs)
                    model = model.eval().requires_grad_(False)
                    
                    # 清理临时变量
                    del minimal_load_kwargs
                    gc.collect()
                    
                    print("[HY-Motion] Model loaded successfully in minimal memory mode")
            except Exception as e:
                # 捕获其他可能的错误
                print(f"[HY-Motion] Unexpected error loading model: {e}")
                raise
            
            # -------------------------- 内存清理 --------------------------
            # 强制释放未使用的GPU内存
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                
            # 显式删除加载过程中可能产生的临时变量
            import gc
            gc.collect()
            
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
        else:
            print("[HY-Motion] transformers<4.40, GGUF not supported")
            raise NotImplementedError(
                "GGUF support requires transformers>=4.40. "
                "Please upgrade: pip install -U transformers>=4.40"
            )

        # Compute crop_start
        template = [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ]
        crop_start = self._compute_crop_start(tokenizer, template)

        actual_device = _get_module_device(model, resolved_device)
        _aggressive_cpu_cleanup("after LLM GPU load")
        _log_memory_status("After LLM load", actual_device)
        _validate_llm_gpu_only(model, strict_gpu_only, "LLM")
        wrapper = HYMotionLLMWrapper(
            model=model,
            tokenizer=tokenizer,
            llm_type="qwen3_gguf",
            max_length=512,
            crop_start=crop_start,
            device=actual_device
        )
        
        # 再次清理内存
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            
        print(f"[HY-Motion] GGUF LLM loaded, hidden_size={wrapper.hidden_size}, device={actual_device}")
        return (wrapper,)

    def _compute_crop_start(self, tokenizer, template) -> int:
        def _find_subseq(a, b):
            for i in range(len(a) - len(b) + 1):
                if a[i:i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        msgs = [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": marker}]
        try:
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            # Some tokenizers don't support enable_thinking
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        return pos if pos >= 0 else max(0, len(full_ids) - 1)


# ============================================================================
# Node 1c: HYMotion LlamaCpp Embedding Test (Experimental)
# ============================================================================

class HYMotionLlamaCppEmbeddingTest:
    """Experimentally test llama-cpp-python token-level GGUF embeddings."""

    @classmethod
    def INPUT_TYPES(s):
        gguf_files = ["(select file)"] + _scan_gguf_files()
        return {
            "required": {
                "gguf_file": (gguf_files, {"default": "(select file)"}),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
                "n_ctx": ("INT", {"default": 512, "min": 1, "max": 131072}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 131072}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "main_gpu": ("INT", {"default": 1, "min": 0, "max": 31}),
                "split_mode": (["none", "layer", "row"], {"default": "none"}),
                "verbose": ("BOOLEAN", {"default": False}),
                "pooling_type": (["none", "mean", "cls"], {"default": "none"}),
            },
            "optional": {
                "backend_note": ("STRING", {"default": "llama-cpp-python direct in-process", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("debug",)
    FUNCTION = "test_embedding"
    CATEGORY = "HY-Motion/Experimental"

    def test_embedding(self, gguf_file, text="A person is walking forward.", n_ctx=512, n_batch=512,
                       n_gpu_layers=-1, main_gpu=1, split_mode="none", verbose=False,
                       pooling_type="none", backend_note=""):
        if pooling_type != "none":
            raise RuntimeError(
                "[HY-Motion] pooling_type must be 'none' for HY-Motion token-level conditioning. "
                "Sequence embeddings from mean/cls pooling are insufficient."
            )

        try:
            from llama_cpp import Llama
            import llama_cpp.llama_cpp as llama_cpp_lib
        except ImportError as e:
            raise ImportError(
                "[HY-Motion] llama-cpp-python is required for HY-Motion LlamaCpp Embedding Test. "
                "Install it in the ComfyUI Python environment. For CUDA builds, you may need "
                "CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 pip install --upgrade "
                "--force-reinstall --no-cache-dir llama-cpp-python"
            ) from e

        gguf_path = _resolve_gguf_path(gguf_file)
        architecture = _read_gguf_architecture(gguf_path)
        pooling_none = getattr(llama_cpp_lib, "LLAMA_POOLING_TYPE_NONE", 0)
        split_modes = {
            "none": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_NONE", 0),
            "layer": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_LAYER", 1),
            "row": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_ROW", 2),
        }
        split_mode_value = split_modes.get(split_mode, split_modes["none"])
        has_embeddings_ith = hasattr(llama_cpp_lib, "llama_get_embeddings_ith")
        llm = None

        try:
            print(
                f"[HY-Motion] Loading llama.cpp embedding test model: {gguf_path}, "
                f"architecture={architecture}, n_ctx={n_ctx}, n_batch={n_batch}, "
                f"n_gpu_layers={n_gpu_layers}, main_gpu={main_gpu}, "
                f"split_mode={split_mode}, pooling_type=none"
            )
            llm = Llama(
                model_path=gguf_path,
                embedding=True,
                pooling_type=pooling_none,
                split_mode=split_mode_value,
                n_ctx=int(n_ctx),
                n_batch=int(n_batch),
                n_gpu_layers=int(n_gpu_layers),
                main_gpu=int(main_gpu),
                verbose=bool(verbose),
            )

            tokens = llm.tokenize(text.encode("utf-8"))
            token_count = len(tokens)
            if token_count <= 0:
                raise RuntimeError("[HY-Motion] llama.cpp tokenizer returned no tokens")
            if token_count > int(n_ctx):
                raise RuntimeError(
                    f"[HY-Motion] token_count={token_count} exceeds n_ctx={n_ctx}; increase n_ctx."
                )
            if token_count > int(n_batch):
                raise RuntimeError(
                    f"[HY-Motion] token_count={token_count} exceeds n_batch={n_batch}; increase n_batch."
                )

            embeddings, total_tokens = llm.embed(text, normalize=False, truncate=False, return_count=True)
            embedding_array = np.asarray(embeddings, dtype=np.float32)
            if embedding_array.ndim == 1:
                raise RuntimeError(
                    "[HY-Motion] llama.cpp returned a single sequence embedding vector, not token-level embeddings. "
                    "HY-Motion requires shape [seq_len, hidden_size], so this result is insufficient."
                )
            if embedding_array.ndim != 2:
                raise RuntimeError(
                    f"[HY-Motion] Expected token-level embedding shape [seq_len, hidden_size], "
                    f"got {embedding_array.shape}."
                )
            if token_count > 1 and embedding_array.shape[0] == 1:
                raise RuntimeError(
                    "[HY-Motion] llama.cpp returned only one embedding row for multiple tokens. "
                    "This looks like sequence pooling and is insufficient for HY-Motion."
                )

            hidden_size = embedding_array.shape[-1]
            try:
                model_hidden_size = int(llm.n_embd())
            except Exception:
                model_hidden_size = hidden_size
            first5 = embedding_array[0, :5].tolist()
            last5 = embedding_array[-1, :5].tolist()
            debug_lines = [
                "[HY-Motion] llama.cpp token-level embedding test succeeded",
                f"gguf_path: {gguf_path}",
                f"architecture: {architecture}",
                f"backend_note: {backend_note}",
                f"api_used: Llama.embed(pooling_type=LLAMA_POOLING_TYPE_NONE); llama_get_embeddings_ith_available={has_embeddings_ith}",
                f"n_gpu_layers: {n_gpu_layers}",
                f"main_gpu: {main_gpu}",
                f"split_mode: {split_mode} ({split_mode_value})",
                f"token_count: {token_count}",
                f"total_tokens_reported_by_llama_cpp: {total_tokens}",
                f"embedding shape: {tuple(embedding_array.shape)}",
                f"hidden_size: {hidden_size}",
                f"model_hidden_size: {model_hidden_size}",
                f"dtype: {embedding_array.dtype}",
                f"first token embedding first5: {first5}",
                f"last token embedding first5: {last5}",
                "HY-Motion conditioning integration not attempted in this test node.",
            ]
            debug = "\n".join(debug_lines)
            print(debug)
            return (debug,)
        finally:
            if llm is not None:
                try:
                    if hasattr(llm, "close"):
                        llm.close()
                except Exception as e:
                    print(f"[HY-Motion] llama.cpp close warning: {e}")
                del llm
            import gc
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================================
# Node 1d: HYMotion Load LLM LlamaCpp (Experimental)
# ============================================================================

class HYMotionLoadLLMLlamaCppExperimental:
    """Load a GGUF LLM through llama-cpp-python for experimental conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        gguf_files = ["(select file)"] + _scan_gguf_files()
        return {
            "required": {
                "gguf_file": (gguf_files, {"default": "(select file)"}),
                "n_ctx": ("INT", {"default": 512, "min": 1, "max": 131072}),
                "n_batch": ("INT", {"default": 512, "min": 1, "max": 131072}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999}),
                "main_gpu": ("INT", {"default": 1, "min": 0, "max": 31}),
                "split_mode": (["none", "layer", "row"], {"default": "none"}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm"
    CATEGORY = "HY-Motion/Loaders"

    def load_llm(self, gguf_file, n_ctx=512, n_batch=512, n_gpu_layers=-1,
                 main_gpu=1, split_mode="none", verbose=False):
        try:
            from llama_cpp import Llama
            import llama_cpp.llama_cpp as llama_cpp_lib
        except ImportError as e:
            raise ImportError(
                "[HY-Motion] llama-cpp-python is required for HY-Motion Load LLM LlamaCpp Experimental. "
                "Install it in the ComfyUI Python environment. For CUDA builds, you may need "
                "CMAKE_ARGS=\"-DGGML_CUDA=on\" FORCE_CMAKE=1 pip install --upgrade "
                "--force-reinstall --no-cache-dir llama-cpp-python"
            ) from e

        gguf_path = _resolve_gguf_path(gguf_file)
        architecture = _read_gguf_architecture(gguf_path)
        pooling_none = getattr(llama_cpp_lib, "LLAMA_POOLING_TYPE_NONE", 0)
        split_modes = {
            "none": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_NONE", 0),
            "layer": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_LAYER", 1),
            "row": getattr(llama_cpp_lib, "LLAMA_SPLIT_MODE_ROW", 2),
        }
        split_mode_value = split_modes.get(split_mode, split_modes["none"])

        print(
            f"[HY-Motion] Loading llama.cpp LLM: {gguf_path}, architecture={architecture}, "
            f"n_ctx={n_ctx}, n_batch={n_batch}, n_gpu_layers={n_gpu_layers}, "
            f"main_gpu={main_gpu}, split_mode={split_mode}, pooling_type=none"
        )
        llama = Llama(
            model_path=gguf_path,
            embedding=True,
            pooling_type=pooling_none,
            split_mode=split_mode_value,
            n_ctx=int(n_ctx),
            n_batch=int(n_batch),
            n_gpu_layers=int(n_gpu_layers),
            main_gpu=int(main_gpu),
            verbose=bool(verbose),
        )

        try:
            hidden_size = int(llama.n_embd())
        except Exception:
            hidden_size = 0
        if hidden_size <= 0:
            llama.close() if hasattr(llama, "close") else None
            raise RuntimeError("[HY-Motion] Could not determine llama.cpp model hidden_size via n_embd().")

        wrapper = HYMotionLlamaCppLLMWrapper(
            llama=llama,
            gguf_path=gguf_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_gpu_layers=n_gpu_layers,
            main_gpu=main_gpu,
            split_mode=split_mode,
            split_mode_value=split_mode_value,
            hidden_size=hidden_size,
            device=f"llama.cpp main_gpu={main_gpu}",
        )
        print(
            f"[HY-Motion] llama.cpp LLM loaded, hidden_size={hidden_size}, "
            f"backend_name={wrapper.backend_name}, main_gpu={main_gpu}, split_mode={split_mode}"
        )
        return (wrapper,)
# ============================================================================
# Node 2: HYMotion Load Network
# ============================================================================

class HYMotionLoadNetwork:
    """Load Motion Diffusion Network"""
    @classmethod
    def INPUT_TYPES(s):
        model_names = _scan_hymotion_networks()
        if not model_names:
            model_names = ["HY-Motion-1.0", "HY-Motion-1.0-Lite"]
        default = "HY-Motion-1.0-Lite" if "HY-Motion-1.0-Lite" in model_names else model_names[0]
        return {
            "required": {
                "model_name": (model_names, {"default": default}),
                "device": (_get_device_choices(), {"default": "default"}),
            },
        }

    RETURN_TYPES = ("HYMOTION_NET",)
    RETURN_NAMES = ("network",)
    FUNCTION = "load_network"
    CATEGORY = "HY-Motion/Loaders"

    def load_network(self, model_name, device="default"):
        from .hymotion.utils.loaders import load_object
        from .hymotion.pipeline.body_model import WoodenMesh

        config, ckpt_path, stats_dir = _resolve_network_model(model_name)

        if config is None:
            raise FileNotFoundError(
                f"Model '{model_name}' not found. Place models in:\n"
                f"  1. {os.path.join(HYMOTION_MODELS_DIR, 'ckpts', 'tencent', model_name)}/ (with config.yml + latest.ckpt)\n"
                f"  2. ComfyUI checkpoints folder as {model_name}.ckpt"
            )

        requested_device = device
        resolved_device = _resolve_device(requested_device)
        print(f"[HY-Motion] Loading network: {model_name}, requested_device={requested_device}, resolved_device={resolved_device}")

        network = load_object(config["network_module"], config["network_module_args"])
        network.eval()

        # Load weights
        mean = torch.zeros(201)
        std = torch.ones(201)
        null_vtxt_feat = torch.randn(1, 1, 768)
        null_ctxt_input = torch.randn(1, 1, 4096)

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[HY-Motion] Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"]

            network_state = {k.replace("motion_transformer.", ""): v
                           for k, v in state_dict.items() if k.startswith("motion_transformer.")}
            network.load_state_dict(network_state, strict=False)

            mean = state_dict.get("mean", mean)
            std = state_dict.get("std", std)
            null_vtxt_feat = state_dict.get("null_vtxt_feat", null_vtxt_feat)
            null_ctxt_input = state_dict.get("null_ctxt_input", null_ctxt_input)

        # Load from stats directory
        if stats_dir is None or not os.path.exists(stats_dir):
            stats_dir = os.path.join(CURRENT_DIR, "HY-Motion-1.0", "stats")
        if os.path.exists(stats_dir):
            mean_path = os.path.join(stats_dir, "Mean.npy")
            std_path = os.path.join(stats_dir, "Std.npy")
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = torch.from_numpy(np.load(mean_path)).float()
                std = torch.from_numpy(np.load(std_path)).float()

        device = resolved_device
        network = network.to(device)
        mean = mean.to(device)
        std = std.to(device)

        wrapper = HYMotionNetworkWrapper(network=network, config=config, mean=mean, std=std, body_model=WoodenMesh(), device=device)
        wrapper.train_frames = 360
        wrapper.output_mesh_fps = 30
        wrapper.validation_steps = 50
        wrapper.input_dim = config["network_module_args"].get("input_dim", 201)
        wrapper.null_vtxt_feat = null_vtxt_feat.to(device)
        wrapper.null_ctxt_input = null_ctxt_input.to(device)

        print(f"[HY-Motion] Network loaded, device={device}")
        return (wrapper,)


# ============================================================================
# Node 2b: HYMotion Load Prompter
# ============================================================================

class HYMotionLoadPrompter:
    """Load Text2MotionPrompter LLM for prompt rewriting and duration estimation"""

    @classmethod
    def INPUT_TYPES(s):
        # Scan for local prompter models
        prompter_models = ["(auto download)"]
        prompter_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Text2MotionPrompter")
        if os.path.exists(prompter_dir):
            prompter_models.append("local: Text2MotionPrompter")

        return {
            "required": {
                "model_source": (prompter_models, {"default": "(auto download)"}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_PROMPTER",)
    RETURN_NAMES = ("prompter",)
    FUNCTION = "load_prompter"
    CATEGORY = "HY-Motion/Loaders"

    def load_prompter(self, model_source, offload_to_cpu=False):
        from .hymotion.prompt_engineering.prompt_rewrite import PromptRewriter

        if model_source == "local: Text2MotionPrompter":
            model_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Text2MotionPrompter")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Prompter model directory not found: {model_path}. Please download the Text2MotionPrompter model first and place it in the HY-Motion models directory.")
        else:
            # Auto download from HuggingFace - not recommended for offline use
            model_path = "Text2MotionPrompter/Text2MotionPrompter"

        print(f"[HY-Motion] Loading Prompter: {model_path}, offload_to_cpu={offload_to_cpu}")
        rewriter = PromptRewriter(model_path=model_path, offload_to_cpu=offload_to_cpu)
        print(f"[HY-Motion] Prompter loaded")

        return (HYMotionPrompterWrapper(rewriter),)


# ============================================================================
# Node 2c: HYMotion Rewrite Prompt
# ============================================================================

class HYMotionRewritePrompt:
    """Rewrite text prompt and estimate motion duration using LLM"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompter": ("HYMOTION_PROMPTER",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("rewritten_text", "duration")
    FUNCTION = "rewrite"
    CATEGORY = "HY-Motion/Conditioning"

    def rewrite(self, prompter: HYMotionPrompterWrapper, text: str):
        print(f"[HY-Motion] ========== Prompt Rewrite ==========")
        print(f"[HY-Motion] INPUT:  {text}")
        duration, rewritten_text = prompter.rewriter.rewrite_prompt_and_infer_time(text)
        print(f"[HY-Motion] OUTPUT: {rewritten_text}")
        print(f"[HY-Motion] DURATION: {duration:.2f}s ({int(duration * 30)} frames)")
        print(f"[HY-Motion] =====================================")
        return (rewritten_text, duration)


# ============================================================================
# Node 3: HYMotion Encode Text
# ============================================================================

class HYMotionEncodeText:
    """Encode text - CLIP loaded internally, LLM passed externally"""
    _clip_model = None
    _clip_tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("HYMOTION_LLM",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
                "device": (_get_device_choices(), {"default": "default"}),
            },
        }

    RETURN_TYPES = ("HYMOTION_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "HY-Motion/Conditioning"

    def _ensure_clip_loaded(self, device="default"):
        """Lazy load CLIP model"""
        target_device = _resolve_device(device)
        if HYMotionEncodeText._clip_model is None:
            from transformers import CLIPTextModel, CLIPTokenizer
            print("[HY-Motion] Loading CLIP (clip-vit-large-patch14)")

            local_path = _find_clip_dir()

            if not local_path:
                search_locations = [os.path.join(HYMOTION_MODELS_DIR, "ckpts")]
                try:
                    search_locations.extend(folder_paths.get_folder_paths("text_encoders"))
                except Exception:
                    pass
                locations_str = "\n- ".join([""] + search_locations)
                raise FileNotFoundError(
                    f"CLIP directory not found. Please download clip-vit-large-patch14 "
                    f"and place it in one of these locations:{locations_str}\n"
                    f"Supported directory names: {', '.join(_CLIP_VARIANTS)}"
                )

            print(f"[HY-Motion] Found CLIP model at: {local_path}")

            HYMotionEncodeText._clip_tokenizer = CLIPTokenizer.from_pretrained(local_path, max_length=77, local_files_only=True)
            HYMotionEncodeText._clip_model = CLIPTextModel.from_pretrained(local_path, local_files_only=True)
            HYMotionEncodeText._clip_model = HYMotionEncodeText._clip_model.eval().requires_grad_(False)

            print("[HY-Motion] CLIP loaded")

        clip_device = _get_module_device(HYMotionEncodeText._clip_model, target_device)
        if clip_device != target_device:
            print(f"[HY-Motion] Moving CLIP from {clip_device} to {target_device}")
            HYMotionEncodeText._clip_model = HYMotionEncodeText._clip_model.to(target_device)
            clip_device = _get_module_device(HYMotionEncodeText._clip_model, target_device)
        return clip_device

    def encode(self, llm: HYMotionLLMWrapper, text: str, device="default"):
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

        requested_device = device
        llm_device_hint = getattr(llm, "device", None)
        if requested_device in (None, "", "default"):
            if llm_device_hint is not None:
                device = _resolve_device(llm_device_hint)
                inherited_from = "llm.device"
            else:
                device = model_management.get_torch_device()
                inherited_from = "model_management.get_torch_device"
        else:
            device = _resolve_device(requested_device)
            inherited_from = "explicit"
            if llm_device_hint is not None:
                inherited_llm_device = _resolve_device(llm_device_hint)
                if inherited_llm_device != device:
                    print(
                        f"[HY-Motion] Encode Text requested_device={requested_device} "
                        f"llm_device={inherited_llm_device} warning=device mismatch; "
                        "using explicit Encode Text device"
                    )
        print(
            f"[HY-Motion] Encode Text requested_device={requested_device} "
            f"inherited_from={inherited_from} resolved_device={device}"
        )
        clip_device = self._ensure_clip_loaded(device)

        text_list = [text]
        device = clip_device

        # CLIP encoding - 优化内存使用
        enc = HYMotionEncodeText._clip_tokenizer(text_list, truncation=True, max_length=77, padding=True, return_tensors="pt")
        clip_device = _get_module_device(HYMotionEncodeText._clip_model, device)
        
        # 清理CLIP编码前的内存
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            out = HYMotionEncodeText._clip_model(input_ids=enc["input_ids"].to(clip_device), attention_mask=enc["attention_mask"].to(clip_device))
            vtxt_raw = out.pooler_output.unsqueeze(1) if out.pooler_output is not None else out.last_hidden_state.mean(1, keepdim=True)
            vtxt_raw = vtxt_raw.to(device)
        
        # 清理CLIP编码后的临时变量
        del enc, out
        torch.cuda.empty_cache()

        # LLM encoding - 优化内存使用
        template = [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": "{}"}]
        llm_text = [llm.tokenizer.apply_chat_template(
            [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        ) for t in text_list]

        max_length = llm.max_length + llm.crop_start
        llm_enc = llm.tokenizer(llm_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        llm_device = _get_module_device(llm.model, getattr(llm, "device", None) or device)
        
        # 清理LLM编码前的内存
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            llm_out = llm.model(
                input_ids=llm_enc["input_ids"].to(llm_device),
                attention_mask=llm_enc["attention_mask"].to(llm_device),
                output_hidden_states=True,
            )

            ctxt_raw = llm_out.hidden_states[-1][:, llm.crop_start:llm.crop_start + llm.max_length].contiguous().to(device)
            ctxt_length = (llm_enc["attention_mask"].sum(dim=-1) - llm.crop_start).clamp(0, llm.max_length).to(device)
        
        # 清理LLM编码后的临时变量
        del llm_enc, llm_out
        torch.cuda.empty_cache()
        
        # 确保ctxt_raw的维度与运动扩散网络期望的4096维匹配
        expected_dim = 4096
        if ctxt_raw.shape[-1] != expected_dim:
            print(f"[HY-Motion] Converting ctxt_raw from {ctxt_raw.shape} to expected {expected_dim} dimensions")
            
            # 创建一个线性转换层，将当前维度转换为4096，并确保dtype与输入一致
            conversion_layer = torch.nn.Linear(ctxt_raw.shape[-1], expected_dim, device=device, dtype=ctxt_raw.dtype)
            
            # 使用转换层将输入转换为期望的维度
            with torch.no_grad():
                ctxt_raw = conversion_layer(ctxt_raw)
            
            # 删除转换层以释放内存
            del conversion_layer
        
        print(f"[HY-Motion] Encode Text clip_device={clip_device} llm_device={llm_device} vtxt={vtxt_raw.shape} ctxt={ctxt_raw.shape}")
        return (HYMotionConditioning(vtxt_raw, ctxt_raw, ctxt_length, text_list),)



# ============================================================================
# Node 3b: HYMotion Encode Text LlamaCpp (Experimental)
# ============================================================================

class HYMotionEncodeTextLlamaCppExperimental:
    """Encode text with llama.cpp token-level embeddings and existing CLIP conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("HYMOTION_LLM",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
            },
            "optional": {
                "device": (_get_device_choices(), {"default": "default"}),
                "hidden_size_adapter": (["zero_pad_2560_to_4096", "strict_4096"], {"default": "zero_pad_2560_to_4096"}),
            },
        }

    RETURN_TYPES = ("HYMOTION_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "HY-Motion/Conditioning"

    def encode(self, llm, text: str, device="default", hidden_size_adapter="zero_pad_2560_to_4096"):
        backend_name = getattr(llm, "backend_name", None)
        if backend_name != "llama.cpp":
            raise RuntimeError(
                f"[HY-Motion] HY-Motion Encode Text LlamaCpp Experimental requires backend_name='llama.cpp', "
                f"got {backend_name!r}. Use HY-Motion Load LLM LlamaCpp Experimental."
            )

        expected_dim = 4096
        supported_llama_dims = (2560, expected_dim)
        hidden_size = int(getattr(llm, "hidden_size", 0) or 0)
        if hidden_size not in supported_llama_dims:
            raise RuntimeError(
                f"[HY-Motion] llama.cpp embedding hidden_size={hidden_size}, but this experimental node only "
                f"supports hidden_size=2560 or {expected_dim}. Projection is not implemented; stopping because "
                "an untrained projection cannot be quality-guaranteed."
            )
        if hidden_size == 2560 and hidden_size_adapter != "zero_pad_2560_to_4096":
            raise RuntimeError(
                "[HY-Motion] llama.cpp hidden_size=2560 requires hidden_size_adapter=zero_pad_2560_to_4096. "
                "strict_4096 accepts only native 4096-dim embeddings."
            )

        llama = getattr(llm, "llama", None)
        if llama is None:
            raise RuntimeError("[HY-Motion] llama.cpp model instance has been released or was not loaded.")

        requested_device = device
        device = model_management.get_torch_device() if requested_device in (None, "", "default") else _resolve_device(requested_device)
        clip_encoder = HYMotionEncodeText()
        clip_device = clip_encoder._ensure_clip_loaded(device)
        text_list = [text]

        enc = HYMotionEncodeText._clip_tokenizer(
            text_list,
            truncation=True,
            max_length=77,
            padding=True,
            return_tensors="pt",
        )
        clip_device = _get_module_device(HYMotionEncodeText._clip_model, clip_device)
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = HYMotionEncodeText._clip_model(
                input_ids=enc["input_ids"].to(clip_device),
                attention_mask=enc["attention_mask"].to(clip_device),
            )
            vtxt_raw = out.pooler_output.unsqueeze(1) if out.pooler_output is not None else out.last_hidden_state.mean(1, keepdim=True)
            vtxt_raw = vtxt_raw.to(device)
        del enc, out
        torch.cuda.empty_cache()

        tokenizer_count = len(llama.tokenize(text.encode("utf-8")))
        if tokenizer_count <= 0:
            raise RuntimeError("[HY-Motion] llama.cpp tokenizer returned no tokens")
        if tokenizer_count > int(llm.n_ctx):
            raise RuntimeError(
                f"[HY-Motion] token_count={tokenizer_count} exceeds llama.cpp n_ctx={llm.n_ctx}; reload with larger n_ctx."
            )
        if tokenizer_count > int(llm.n_batch):
            raise RuntimeError(
                f"[HY-Motion] token_count={tokenizer_count} exceeds llama.cpp n_batch={llm.n_batch}; reload with larger n_batch."
            )

        embeddings, total_tokens = llama.embed(text, normalize=False, truncate=False, return_count=True)
        embedding_array = np.asarray(embeddings, dtype=np.float32)
        if embedding_array.ndim == 1:
            raise RuntimeError(
                "[HY-Motion] llama.cpp returned a single sequence embedding vector, not token-level embeddings. "
                "HY-Motion requires shape [seq_len, hidden_size]."
            )
        if embedding_array.ndim != 2:
            raise RuntimeError(
                f"[HY-Motion] Expected llama.cpp token-level embedding shape [seq_len, hidden_size], "
                f"got {embedding_array.shape}."
            )
        embedding_dim = int(embedding_array.shape[-1])
        if embedding_dim not in supported_llama_dims:
            raise RuntimeError(
                f"[HY-Motion] llama.cpp embedding hidden_size={embedding_dim}, but this experimental node only "
                f"supports hidden_size=2560 or {expected_dim}. Projection is not implemented; stopping."
            )
        if embedding_dim == 2560 and hidden_size_adapter != "zero_pad_2560_to_4096":
            raise RuntimeError(
                "[HY-Motion] llama.cpp returned hidden_size=2560, but hidden_size_adapter is strict_4096."
            )

        token_count = int(embedding_array.shape[0])
        if token_count <= 0:
            raise RuntimeError("[HY-Motion] llama.cpp returned no token-level embedding rows")
        if tokenizer_count != token_count:
            print(
                f"[HY-Motion] llama.cpp token count warning: tokenizer_count={tokenizer_count}, "
                f"embedding_rows={token_count}; using embedding_rows for ctxt_length"
            )

        ctxt_tensor = torch.from_numpy(embedding_array).to(device=device, dtype=torch.float32)
        adapter_note = "native_4096"
        if embedding_dim == 2560:
            pad_dim = expected_dim - embedding_dim
            pad = torch.zeros(
                (*ctxt_tensor.shape[:-1], pad_dim),
                dtype=ctxt_tensor.dtype,
                device=ctxt_tensor.device,
            )
            ctxt_tensor = torch.cat([ctxt_tensor, pad], dim=-1)
            adapter_note = f"zero_pad_2560_to_4096(pad_dim={pad_dim})"
            print(
                "[HY-Motion] WARNING: EXPERIMENTAL zero-padding llama.cpp embeddings "
                "from hidden_size=2560 to 4096. Output quality is not guaranteed."
            )
        ctxt_raw = ctxt_tensor.unsqueeze(0)
        ctxt_length = torch.tensor([token_count], dtype=torch.long, device=device)
        print(
            f"[HY-Motion] Encode Text LlamaCpp clip_device={clip_device} requested_device={requested_device} "
            f"main_gpu={getattr(llm, 'main_gpu', None)} split_mode={getattr(llm, 'split_mode', None)} "
            f"hidden_size_adapter={adapter_note} vtxt={vtxt_raw.shape} ctxt={ctxt_raw.shape} "
            f"token_count={token_count} total_tokens={total_tokens}"
        )
        return (HYMotionConditioning(vtxt_raw, ctxt_raw, ctxt_length, text_list),)
# ============================================================================
# Node 4: HYMotion Generate
# ============================================================================

class HYMotionGenerate:
    """Motion generation"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network": ("HYMOTION_NET",),
                "conditioning": ("HYMOTION_COND",),
                "duration": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 4}),
                "device": (_get_device_choices(), {"default": "default"}),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "generate"
    CATEGORY = "HY-Motion"

    def generate(self, network: HYMotionNetworkWrapper, conditioning: HYMotionConditioning,
                 duration: float, seed: int, cfg_scale: float = 5.0, num_samples: int = 1, device="default"):
        import comfy.utils
        from torchdiffeq import odeint
        from .hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor

        requested_device = device
        network_device = getattr(network, "device", None)
        if network_device is not None:
            device = _resolve_device(network_device)
            inherited_from = "network.device"
            if requested_device not in (None, "", "default"):
                try:
                    explicit_device = _resolve_device(requested_device)
                    if explicit_device != device:
                        print(
                            f"[HY-Motion] Generate requested_device={requested_device} "
                            f"network_device={device} warning=device mismatch; using network.device"
                        )
                except Exception as e:
                    print(
                        f"[HY-Motion] Generate requested_device={requested_device} "
                        f"network_device={device} warning=invalid explicit device; using network.device ({e})"
                    )
        elif requested_device not in (None, "", "default"):
            device = _resolve_device(requested_device)
            inherited_from = "explicit"
        else:
            device = model_management.get_torch_device()
            inherited_from = "model_management.get_torch_device"
        print(
            f"[HY-Motion] Generate requested_device={requested_device} "
            f"inherited_from={inherited_from} resolved_device={device}"
        )
        print(f"[HY-Motion] Generate device={device}")

        network.network = network.network.to(device)
        network.mean = network.mean.to(device)
        network.std = network.std.to(device)
        network.device = device
        if hasattr(network, "null_vtxt_feat"):
            network.null_vtxt_feat = network.null_vtxt_feat.to(device)
        if hasattr(network, "null_ctxt_input"):
            network.null_ctxt_input = network.null_ctxt_input.to(device)

        length = min(max(int(duration * network.output_mesh_fps), 20), network.train_frames)
        seeds = [seed + i for i in range(num_samples)]

        print(f"[HY-Motion] Generating: {duration}s, length={length}, seeds={seeds}, device={device}")

        vtxt_input = conditioning.vtxt_raw.to(device)
        ctxt_input = conditioning.ctxt_raw.to(device)
        ctxt_length = conditioning.ctxt_length.to(device)

        if vtxt_input.shape[0] == 1 and num_samples > 1:
            vtxt_input = vtxt_input.repeat(num_samples, 1, 1)
            ctxt_input = ctxt_input.repeat(num_samples, 1, 1)
            ctxt_length = ctxt_length.repeat(num_samples)

        ctxt_mask = length_to_mask(ctxt_length, ctxt_input.shape[1]).to(device)
        x_length = torch.LongTensor([length] * num_samples).to(device)
        x_mask = length_to_mask(x_length, network.train_frames).to(device)

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            # 确保null_vtxt_feat的维度与vtxt_input匹配
            if network.null_vtxt_feat.shape[-1] != vtxt_input.shape[-1]:
                print(f"[HY-Motion] Resizing null_vtxt_feat from {network.null_vtxt_feat.shape} to match {vtxt_input.shape}")
                # 使用随机初始化的张量，维度匹配vtxt_input
                network.null_vtxt_feat = torch.randn(1, 1, vtxt_input.shape[-1], device=device)
            
            # 确保null_ctxt_input的维度与ctxt_input匹配
            if network.null_ctxt_input.shape[-1] != ctxt_input.shape[-1]:
                print(f"[HY-Motion] Resizing null_ctxt_input from {network.null_ctxt_input.shape} to match {ctxt_input.shape}")
                # 使用随机初始化的张量，维度匹配ctxt_input
                network.null_ctxt_input = torch.randn(1, 1, ctxt_input.shape[-1], device=device)
            
            # 现在可以安全地拼接
            vtxt_input = torch.cat([network.null_vtxt_feat.expand_as(vtxt_input), vtxt_input], dim=0)
            ctxt_input = torch.cat([network.null_ctxt_input.expand_as(ctxt_input), ctxt_input], dim=0)
            ctxt_mask = torch.cat([ctxt_mask] * 2, dim=0)
            x_mask = torch.cat([x_mask] * 2, dim=0)

        pbar = comfy.utils.ProgressBar(network.validation_steps)
        step = [0]

        def fn(t, x):
            x_in = torch.cat([x] * 2, dim=0) if do_cfg else x
            pred = network.network(x=x_in, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                                   timesteps=t.to(device).expand(x_in.shape[0]), x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)
            if do_cfg:
                pred_u, pred_c = pred.chunk(2)
                pred = pred_u + cfg_scale * (pred_c - pred_u)
            step[0] += 1
            pbar.update_absolute(step[0], network.validation_steps)
            return pred

        t = torch.linspace(0, 1, network.validation_steps + 1, device=device)
        y0 = torch.cat([randn_tensor((1, network.train_frames, network.input_dim),
                       generator=torch.Generator(device=device).manual_seed(s), device=device) for s in seeds])

        with torch.no_grad():
            sampled = odeint(fn, y0, t, method="euler")[-1][:, :length]

        output_dict = self._decode(sampled, network)
        print(f"[HY-Motion] Done")
        return (HYMotionData(output_dict, conditioning.text[0], duration, seeds),)

    def _decode(self, latent, net):
        from scipy.signal import savgol_filter
        from .hymotion.utils.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d, matrix_to_quaternion, quaternion_to_matrix, quaternion_fix_continuity
        from .hymotion.utils.motion_process import smooth_rotation

        mean = net.mean.to(latent.device)
        std_source = net.std.to(latent.device)
        std = torch.where(std_source < 1e-3, torch.ones_like(std_source), std_source)
        x = latent * std + mean
        B, L = x.shape[:2]

        transl = x[..., :3].clone()
        rot6d = torch.cat([x[..., 3:9].reshape(B, L, 1, 6), x[..., 9:9+126].reshape(B, L, 21, 6)], dim=2)

        # Smooth rotations
        RR = rot6d_to_rotation_matrix(rot6d)
        qq = matrix_to_quaternion(RR)
        qq = qq.moveaxis(1, 0).contiguous().view(L, -1, 4)
        qq = quaternion_fix_continuity(qq).view(L, B, 22, 4).moveaxis(0, 1)
        rot6d_s = rotation_matrix_to_rot6d(quaternion_to_matrix(torch.from_numpy(smooth_rotation(qq.cpu().numpy(), sigma=1.0)))).to(latent.device)

        transl_np = transl.cpu().numpy()
        for b in range(B):
            for j in range(3):
                transl_np[b, :, j] = savgol_filter(transl_np[b, :, j], 11, 5)
        transl_s = torch.from_numpy(transl_np).to(latent.device)

        k3d = None
        if net.body_model:
            # Run body_model on CPU (its internal tensors are on CPU)
            rot6d_cpu = rot6d_s.cpu()
            transl_cpu = transl_s.cpu()
            k3d_list = []
            vertices_list = []
            with torch.no_grad():
                for b in range(B):
                    out = net.body_model.forward({"rot6d": rot6d_cpu[b], "trans": transl_cpu[b]})
                    k3d_list.append(out["keypoints3d"])
                    vertices_list.append(out["vertices"])
            k3d = torch.stack(k3d_list, dim=0)
            vertices = torch.stack(vertices_list, dim=0)
            # Align to ground
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)
            k3d = k3d.clone()
            k3d[..., 1] -= min_y
            transl_cpu = transl_cpu.clone()
            transl_cpu[..., 1] -= min_y.squeeze(-1)
            transl_s = transl_cpu
        else:
            k3d = torch.zeros(B, L, 22, 3)

        return {"keypoints3d": k3d.cpu(), "rot6d": rot6d_s.cpu(), "transl": transl_s.cpu(),
                "root_rotations_mat": rot6d_to_rotation_matrix(rot6d_s[:, :, 0].cpu()).cpu()}


# ============================================================================
# Node 5: HYMotion Preview
# ============================================================================

class HYMotionPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motion_data": ("HYMOTION_DATA",),
            "sample_index": ("INT", {"default": 0, "min": 0, "max": 3}),
            "frame_step": ("INT", {"default": 5, "min": 1, "max": 30}),
            "image_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def render(self, motion_data, sample_index=0, frame_step=5, image_size=512):
        kpts = motion_data.output_dict.get("keypoints3d")
        if kpts is None:
            return (torch.zeros(1, image_size, image_size, 3),)

        kpts = kpts[min(sample_index, kpts.shape[0]-1)].cpu().numpy()
        frames = [self._draw(kpts[i], image_size) for i in range(0, len(kpts), frame_step)]
        return (torch.from_numpy(np.stack(frames)).float() / 255.0,)

    def _draw(self, kpts, size):
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        # Only use first 22 main joints for skeleton (ignore fingers)
        bones = [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
        colors = [(255,100,100)]*4 + [(100,100,255)]*4 + [(100,200,100)]*5 + [(255,150,50)]*4 + [(50,150,255)]*4

        # Only take first 22 joints for drawing
        kpts_draw = kpts[:22] if len(kpts) > 22 else kpts
        x, y = kpts_draw[:, 0], kpts_draw[:, 1]
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        scale = max(x.max()-x.min(), y.max()-y.min(), 0.1) * 1.3

        def px(p): return int((p[0]-cx)/scale*size + size/2), int(size/2 - (p[1]-cy)/scale*size)

        for (a, b), c in zip(bones, colors):
            if a < len(kpts_draw) and b < len(kpts_draw):
                p1, p2 = px(kpts_draw[a]), px(kpts_draw[b])
                steps = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]), 1) + 1
                for t in np.linspace(0, 1, int(steps)):
                    px_, py_ = int(p1[0]+t*(p2[0]-p1[0])), int(p1[1]+t*(p2[1]-p1[1]))
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if 0 <= px_+dx < size and 0 <= py_+dy < size:
                                img[py_+dy, px_+dx] = c

        for i in range(len(kpts_draw)):
            p = px(kpts_draw[i])
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx+dy*dy <= 16 and 0 <= p[0]+dx < size and 0 <= p[1]+dy < size:
                        img[p[1]+dy, p[0]+dx] = [50, 50, 50]
        return img


# ============================================================================
# Node 6: HYMotion Save NPZ
# ============================================================================

class HYMotionSaveNPZ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motion_data": ("HYMOTION_DATA",),
            "output_dir": ("STRING", {"default": "hymotion_npz"}),
            "filename_prefix": ("STRING", {"default": "motion"}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def save(self, motion_data, output_dir, filename_prefix):
        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts, uid = get_timestamp(), str(uuid.uuid4())[:8]

        paths = []
        for i in range(motion_data.batch_size):
            data = {k: motion_data.output_dict[k][i].cpu().numpy() if hasattr(motion_data.output_dict[k][i], 'cpu') else motion_data.output_dict[k][i]
                    for k in ["keypoints3d", "rot6d", "transl", "root_rotations_mat"] if k in motion_data.output_dict}
            data.update({"text": motion_data.text, "duration": motion_data.duration, "seed": motion_data.seeds[i] if i < len(motion_data.seeds) else 0})
            path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.npz")
            np.savez(path, **data)
            paths.append(path)
            print(f"[HY-Motion] Saved: {path}")

        return ("\n".join([os.path.relpath(p, COMFY_OUTPUT_DIR) for p in paths]),)


# ============================================================================
# Node 7: HYMotion Export FBX
# ============================================================================

class HYMotionExportFBX:
    _fbx_converter = None
    _fbx_converter_path = None  # Track the template path to detect changes

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "output_dir": ("STRING", {"default": "hymotion_fbx"}),
                "filename_prefix": ("STRING", {"default": "motion"}),
            },
            "optional": {
                "custom_fbx_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to custom FBX model (Mixamo character). Supports: 'input/3d/char.fbx', 'output/3d/char.fbx', or just '3d/char.fbx' (defaults to input/). Leave empty for default wooden boy."
                }),
                "yaw_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": "Rotate the character around Y-axis in degrees (e.g., 180 to face opposite direction)."
                }),
                "scale": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "tooltip": "Force specific scale multiplier. Leave at 0.0 for automatic height-based scaling (recommended)."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_paths",)
    FUNCTION = "export"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def _resolve_fbx_path(self, custom_fbx_path):
        """
        Resolve custom FBX path with support for relative paths.
        Rules:
        - If path starts with 'input/' or 'output/' -> resolve relative to ComfyUI root
        - If path has no such prefix (e.g., '3d/2.fbx') -> assume 'output/' prefix
        - If path is absolute and exists -> use as is
        """
        print(f"[HY-Motion] DEBUG _resolve_fbx_path called with: '{custom_fbx_path}'")

        if not custom_fbx_path or not custom_fbx_path.strip():
            print(f"[HY-Motion] DEBUG: custom_fbx_path is empty or None")
            return None

        path = custom_fbx_path.strip().replace("\\", "/")
        print(f"[HY-Motion] DEBUG: normalized path = '{path}'")

        # If absolute path and exists, use directly
        if os.path.isabs(path):
            print(f"[HY-Motion] DEBUG: path is absolute, exists={os.path.exists(path)}")
            if os.path.exists(path):
                return path

        # Get ComfyUI root directory (parent of output dir)
        comfy_root = os.path.dirname(COMFY_OUTPUT_DIR)
        print(f"[HY-Motion] DEBUG: COMFY_OUTPUT_DIR = '{COMFY_OUTPUT_DIR}'")
        print(f"[HY-Motion] DEBUG: comfy_root = '{comfy_root}'")

        # Check if path starts with input/ or output/
        if path.startswith("input/") or path.startswith("output/"):
            resolved = os.path.join(comfy_root, path)
            print(f"[HY-Motion] DEBUG: path has input/output prefix, resolved = '{resolved}'")
        else:
            # Default to input/ prefix
            resolved = os.path.join(comfy_root, "input", path)
            print(f"[HY-Motion] DEBUG: no prefix, defaulting to input, resolved = '{resolved}'")

        # Normalize path
        resolved = os.path.normpath(resolved)
        print(f"[HY-Motion] DEBUG: final resolved path = '{resolved}'")
        print(f"[HY-Motion] DEBUG: file exists = {os.path.exists(resolved)}")

        if os.path.exists(resolved):
            return resolved
        else:
            print(f"[HY-Motion] Custom FBX path not found: {resolved}")
            return None

    def export(self, motion_data, output_dir, filename_prefix, custom_fbx_path="", yaw_offset=0.0, scale=0.0):
        from .hymotion.pipeline.body_model import construct_smpl_data_dict

        print(f"[HY-Motion] ========== EXPORT FBX ==========")
        print(f"[HY-Motion] DEBUG: custom_fbx_path param = '{custom_fbx_path}'")
        print(f"[HY-Motion] DEBUG: yaw_offset = {yaw_offset}, scale = {scale}")

        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts, uid = get_timestamp(), str(uuid.uuid4())[:8]

        # Resolve custom FBX path with relative path support
        resolved_fbx_path = self._resolve_fbx_path(custom_fbx_path)
        print(f"[HY-Motion] DEBUG: resolved_fbx_path = '{resolved_fbx_path}'")

        if resolved_fbx_path:
            # Use retargeting for custom Mixamo/other FBX models
            print(f"[HY-Motion] Using RETARGET mode with custom FBX: {resolved_fbx_path}")
            return self._export_with_retarget(motion_data, out_dir, filename_prefix, ts, uid,
                                               resolved_fbx_path, yaw_offset, scale)
        else:
            # Use original wooden boy export
            print(f"[HY-Motion] Using WOODEN BOY mode (no custom FBX)")
            return self._export_wooden_boy(motion_data, out_dir, filename_prefix, ts, uid)

    def _export_wooden_boy(self, motion_data, out_dir, filename_prefix, ts, uid):
        """Original export using wooden boy template"""
        from .hymotion.pipeline.body_model import construct_smpl_data_dict

        # Lazy load FBX converter
        template_path = os.path.join(CURRENT_DIR, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")

        if HYMotionExportFBX._fbx_converter is None or HYMotionExportFBX._fbx_converter_path != template_path:
            try:
                import fbx
                from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
                HYMotionExportFBX._fbx_converter = SMPLH2WoodFBX(template_fbx_path=template_path)
                HYMotionExportFBX._fbx_converter_path = template_path
                print("[HY-Motion] FBX converter loaded (wooden boy)")
            except ImportError:
                return ("FBX SDK not installed",)
            except Exception as e:
                return (f"FBX converter error: {e}",)

        paths = []
        for i in range(motion_data.batch_size):
            try:
                rot6d = motion_data.output_dict["rot6d"][i].clone()
                transl = motion_data.output_dict["transl"][i].clone()
                smpl_data = construct_smpl_data_dict(rot6d, transl)

                path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.fbx")
                success = HYMotionExportFBX._fbx_converter.convert_npz_to_fbx(smpl_data, path)

                if success:
                    paths.append(path)
                    print(f"[HY-Motion] FBX saved: {path}")
                    # Save text description
                    txt_path = path.replace(".fbx", ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(motion_data.text)
            except Exception as e:
                print(f"[HY-Motion] FBX export error: {e}")

        if not paths:
            return ("Export failed",)
        return ("\n".join([os.path.relpath(p, COMFY_OUTPUT_DIR) for p in paths]),)

    def _export_with_retarget(self, motion_data, out_dir, filename_prefix, ts, uid, target_fbx, yaw_offset, scale):
        """Export with retargeting to custom FBX model (e.g., Mixamo)"""
        from .hymotion.pipeline.body_model import construct_smpl_data_dict

        try:
            from .hymotion.utils.retarget_fbx import (
                load_npz, load_fbx, load_bone_mapping, retarget_animation,
                apply_retargeted_animation, save_fbx, HAS_FBX_SDK
            )
        except ImportError as e:
            print(f"[HY-Motion] Retarget import error: {e}")
            return (f"Retarget module error: {e}",)

        if not HAS_FBX_SDK:
            return ("FBX SDK not installed",)

        print(f"[HY-Motion] Retargeting to custom FBX: {target_fbx}")

        paths = []
        mapping = load_bone_mapping("")  # Use built-in Mixamo mappings

        for i in range(motion_data.batch_size):
            try:
                # Create temp NPZ from motion_data
                temp_npz = os.path.join(out_dir, f"_temp_{ts}_{i}.npz")
                output_fbx = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.fbx")

                # Prepare data dict
                data_dict = {}
                for key in ['keypoints3d', 'rot6d', 'transl', 'root_rotations_mat']:
                    if key in motion_data.output_dict and motion_data.output_dict[key] is not None:
                        tensor = motion_data.output_dict[key][i]
                        if isinstance(tensor, torch.Tensor):
                            data_dict[key] = tensor.cpu().numpy()
                        else:
                            data_dict[key] = np.array(tensor)

                # Add SMPL-H full poses
                if "rot6d" in data_dict and "transl" in data_dict:
                    smpl_data = construct_smpl_data_dict(
                        torch.from_numpy(data_dict["rot6d"]),
                        torch.from_numpy(data_dict["transl"])
                    )
                    for k, v in smpl_data.items():
                        if k not in data_dict:
                            data_dict[k] = v

                np.savez(temp_npz, **data_dict)

                # Load source (NPZ) and target (FBX) skeletons
                src_skel = load_npz(temp_npz)
                tgt_man, tgt_scene, tgt_skel = load_fbx(target_fbx)

                # Retarget animation
                force_scale = scale if scale > 0 else 0.0
                rots, locs = retarget_animation(src_skel, tgt_skel, mapping, force_scale, yaw_offset, neutral_fingers=True)

                # Apply and save
                src_time_mode = tgt_scene.GetGlobalSettings().GetTimeMode()
                apply_retargeted_animation(tgt_scene, tgt_skel, rots, locs, src_skel.frame_start, src_skel.frame_end, src_time_mode)
                save_fbx(tgt_man, tgt_scene, output_fbx)

                paths.append(output_fbx)
                print(f"[HY-Motion] Retargeted FBX saved: {output_fbx}")

                # Save text description
                txt_path = output_fbx.replace(".fbx", ".txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(motion_data.text)

                # Cleanup temp file
                if os.path.exists(temp_npz):
                    os.remove(temp_npz)

            except Exception as e:
                import traceback
                print(f"[HY-Motion] Retarget error for batch {i}: {e}")
                traceback.print_exc()
                # Cleanup temp file on error
                if 'temp_npz' in locals() and os.path.exists(temp_npz):
                    os.remove(temp_npz)
                continue

        if not paths:
            return ("Export failed",)
        return ("\n".join([os.path.relpath(p, COMFY_OUTPUT_DIR) for p in paths]),)


# ============================================================================
# Node 8: HYMotion Preview Animation (Three.js with GLB Export)
# ============================================================================

class HYMotionPreviewAnimation:
    """
    Interactive 3D motion preview with Three.js.
    Supports playback controls and GLB export with skeleton animation.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
            },
            "optional": {
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 3}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "preview"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def preview(self, motion_data: HYMotionData, sample_index: int = 0):
        import json
        from .hymotion.utils.geometry import rot6d_to_rotation_matrix, matrix_to_quaternion

        idx = min(sample_index, motion_data.batch_size - 1)

        # Extract motion data for the selected sample
        rot6d = motion_data.output_dict["rot6d"][idx]  # (num_frames, 22, 6)
        transl = motion_data.output_dict["transl"][idx]  # (num_frames, 3)

        # Convert rot6d to quaternion using the same math as FBX export
        # rot6d -> rotation_matrix -> quaternion
        if hasattr(rot6d, 'cpu'):
            rot6d_tensor = rot6d.cpu()
        else:
            rot6d_tensor = torch.from_numpy(rot6d).float()

        # (num_frames, 22, 6) -> (num_frames, 22, 3, 3)
        rot_matrices = rot6d_to_rotation_matrix(rot6d_tensor)
        # (num_frames, 22, 3, 3) -> (num_frames, 22, 4) quaternion [w, x, y, z]
        quaternions = matrix_to_quaternion(rot_matrices)

        # Convert to numpy
        quaternions_np = quaternions.numpy()
        if hasattr(transl, 'cpu'):
            transl_np = transl.cpu().numpy()
        else:
            transl_np = transl

        num_frames = quaternions_np.shape[0]
        num_joints = quaternions_np.shape[1]

        # Flatten for transfer: quaternions are [w, x, y, z] format
        motion_json = json.dumps({
            "quaternions": quaternions_np.flatten().tolist(),  # (num_frames * num_joints * 4)
            "transl": transl_np.flatten().tolist(),  # (num_frames * 3)
            "num_frames": int(num_frames),
            "num_joints": int(num_joints),
            "fps": 30,
            "text": motion_data.text,
            "duration": motion_data.duration,
        })

        print(f"[HY-Motion] Preview Animation: {num_frames} frames, {num_joints} joints")
        return {"ui": {"motion_data": [motion_json]}, "result": (motion_json,)}


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "HYMotionLoadLLM": HYMotionLoadLLM,
    "HYMotionLoadLLMGGUF": HYMotionLoadLLMGGUF,
    "HYMotionLlamaCppEmbeddingTest": HYMotionLlamaCppEmbeddingTest,
    "HYMotionLoadLLMLlamaCppExperimental": HYMotionLoadLLMLlamaCppExperimental,
    "HYMotionEncodeTextLlamaCppExperimental": HYMotionEncodeTextLlamaCppExperimental,
    "HYMotionLoadNetwork": HYMotionLoadNetwork,
    "HYMotionLoadPrompter": HYMotionLoadPrompter,
    "HYMotionRewritePrompt": HYMotionRewritePrompt,
    "HYMotionEncodeText": HYMotionEncodeText,
    "HYMotionGenerate": HYMotionGenerate,
    "HYMotionPreview": HYMotionPreview,
    "HYMotionSaveNPZ": HYMotionSaveNPZ,
    "HYMotionExportFBX": HYMotionExportFBX,
    "HYMotionPreviewAnimation": HYMotionPreviewAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionLoadLLM": "HY-Motion Load LLM",
    "HYMotionLoadLLMGGUF": "HY-Motion Load LLM (GGUF)",
    "HYMotionLlamaCppEmbeddingTest": "HY-Motion LlamaCpp Embedding Test",
    "HYMotionLoadLLMLlamaCppExperimental": "HY-Motion Load LLM LlamaCpp Experimental",
    "HYMotionEncodeTextLlamaCppExperimental": "HY-Motion Encode Text LlamaCpp Experimental",
    "HYMotionLoadNetwork": "HY-Motion Load Network",
    "HYMotionLoadPrompter": "HY-Motion Load Prompter",
    "HYMotionRewritePrompt": "HY-Motion Rewrite Prompt",
    "HYMotionEncodeText": "HY-Motion Encode Text",
    "HYMotionGenerate": "HY-Motion Generate",
    "HYMotionPreview": "HY-Motion Preview",
    "HYMotionSaveNPZ": "HY-Motion Save NPZ",
    "HYMotionExportFBX": "HY-Motion Export FBX",
    "HYMotionPreviewAnimation": "HY-Motion Preview Animation (3D)",
}
