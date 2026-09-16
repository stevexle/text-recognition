"""
Automated NVIDIA TensorRT Engine Builder for ViT-Transformer Text Recognition.
Compiles Encoder and Decoder ONNX models into high-performance FP16 TensorRT
execution engines (.engine) with dynamic shape profiles.

Supports both:
  1. Python native TensorRT Builder (via `tensorrt` package - no trtexec binary required).
  2. Standalone `trtexec` CLI compiler if available on system PATH.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# Auto-detect and register NVIDIA CUDA, cuDNN, and TensorRT shared libraries on Linux
if sys.platform == "linux":
    import ctypes
    import site
    try:
        for site_pkg in site.getsitepackages():
            nvidia_dir = os.path.join(site_pkg, "nvidia")
            if os.path.isdir(nvidia_dir):
                for sub in ["cuda_runtime", "cublas", "cudnn", "cufft", "curand", "tensorrt"]:
                    lib_dir = os.path.join(nvidia_dir, sub, "lib")
                    if os.path.isdir(lib_dir):
                        if "LD_LIBRARY_PATH" in os.environ:
                            if lib_dir not in os.environ["LD_LIBRARY_PATH"]:
                                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{os.environ['LD_LIBRARY_PATH']}"
                        else:
                            os.environ["LD_LIBRARY_PATH"] = lib_dir
                        for f in sorted(os.listdir(lib_dir)):
                            if f.endswith(".so") or ".so." in f:
                                try:
                                    ctypes.CDLL(os.path.join(lib_dir, f), mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass
            trt_dir = os.path.join(site_pkg, "tensorrt")
            if os.path.isdir(trt_dir):
                for f in sorted(os.listdir(trt_dir)):
                    if f.endswith(".so") or ".so." in f:
                        try:
                            ctypes.CDLL(os.path.join(trt_dir, f), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass
    except Exception:
        pass

# Dynamic Shape Profiles for ViT-Transformer Recognition
SHAPE_PROFILES = {
    "encoder": {
        "image": {
            "min": (1, 3, 32, 256),
            "opt": (16, 3, 32, 256),
            "max": (64, 3, 32, 256),
        }
    },
    "decoder": {
        "tgt_tokens": {
            "min": (1, 1),
            "opt": (16, 16),
            "max": (64, 128),
        },
        "memory": {
            "min": (1, 256, 384),
            "opt": (16, 256, 384),
            "max": (64, 256, 384),
        }
    }
}


def shape_tuple_to_str(shape: Tuple[int, ...]) -> str:
    return "x".join(str(x) for x in shape)


def find_trtexec() -> Optional[str]:
    """Auto-detect trtexec compiler binary across PATH and standard locations."""
    found = shutil.which("trtexec")
    if found:
        return found

    venv_bin = Path(sys.prefix) / "bin" / "trtexec"
    if venv_bin.exists() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

    candidate_paths = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/cuda/bin/trtexec",
        "/usr/bin/trtexec",
    ]
    for p in candidate_paths:
        if Path(p).exists() and os.access(p, os.X_OK):
            return p

    return None


def build_engine_from_onnx_python(
    onnx_path: str,
    engine_path: str,
    shapes: Dict[str, Dict[str, Tuple[int, ...]]],
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
) -> bool:
    """
    Compile ONNX model to TensorRT engine via Python TensorRT API.
    Does not require trtexec CLI binary!
    """
    if dry_run:
        print(f"[Dry Run] Compiling {onnx_path} -> {engine_path} via TensorRT Python API")
        print(f"[Dry Run] Profiles: {shapes}, fp16={fp16}")
        return True

    try:
        import tensorrt as trt
    except ImportError as e:
        print(f"[ERROR] tensorrt Python package is required: {e}")
        return False

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)

    if hasattr(trt, "NetworkDefinitionCreationFlag") and hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, trt_logger)

    print(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(f"[ERROR] ONNX parse error: {parser.get_error(error)}")
            return False

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType") and hasattr(trt.MemoryPoolType, "WORKSPACE"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_mb * 1024 * 1024

    if fp16 and hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)

    # Configure dynamic shape profile for all inputs
    profile = builder.create_optimization_profile()
    for input_name, profile_dict in shapes.items():
        min_shape = profile_dict["min"]
        opt_shape = profile_dict["opt"]
        max_shape = profile_dict["max"]
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        print(f"  Set dynamic profile for '{input_name}': min={min_shape}, opt={opt_shape}, max={max_shape}")

    config.add_optimization_profile(profile)

    print(f"Building serialized TensorRT engine: {engine_path} (this may take 1-3 minutes)...")
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT failed to build serialized network for {onnx_path}")
        with open(engine_path, "wb") as f:
            f.write(plan)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build engine for {onnx_path}")
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())

    print(f"Successfully compiled engine via Python API: {engine_path}")
    return True


def build_engine_from_onnx_trtexec(
    trtexec_bin: str,
    onnx_path: str,
    engine_path: str,
    shapes: Dict[str, Dict[str, Tuple[int, ...]]],
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
) -> bool:
    """Compile ONNX model to TensorRT engine using standalone trtexec CLI binary."""
    min_shapes_args = []
    opt_shapes_args = []
    max_shapes_args = []

    for name, s in shapes.items():
        min_shapes_args.append(f"{name}:{shape_tuple_to_str(s['min'])}")
        opt_shapes_args.append(f"{name}:{shape_tuple_to_str(s['opt'])}")
        max_shapes_args.append(f"{name}:{shape_tuple_to_str(s['max'])}")

    cmd = [
        trtexec_bin,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={','.join(min_shapes_args)}",
        f"--optShapes={','.join(opt_shapes_args)}",
        f"--maxShapes={','.join(max_shapes_args)}",
        f"--memPoolSize=workspace:{workspace_mb}M",
    ]
    if fp16:
        cmd.append("--fp16")

    print(f"Executing trtexec command:\n{' '.join(cmd)}")
    if dry_run:
        print("[Dry Run] Skipping command execution.")
        return True

    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        print(f"[ERROR] trtexec failed with return code {res.returncode}")
        return False
    return True


def compile_model(
    model_name: str,
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_mb: int = 2048,
    dry_run: bool = False,
    force_python: bool = False,
) -> bool:
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    shapes = SHAPE_PROFILES[model_name]

    if not Path(onnx_path).exists() and not dry_run:
        print(f"[ERROR] ONNX file not found at: {onnx_path}")
        return False

    trtexec_bin = None if force_python else find_trtexec()
    if trtexec_bin:
        print(f"Using trtexec binary at: {trtexec_bin}")
        return build_engine_from_onnx_trtexec(
            trtexec_bin=trtexec_bin,
            onnx_path=onnx_path,
            engine_path=engine_path,
            shapes=shapes,
            fp16=fp16,
            workspace_mb=workspace_mb,
            dry_run=dry_run,
        )
    else:
        print("Using Python TensorRT API Builder")
        return build_engine_from_onnx_python(
            onnx_path=onnx_path,
            engine_path=engine_path,
            shapes=shapes,
            fp16=fp16,
            workspace_mb=workspace_mb,
            dry_run=dry_run,
        )


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engines for ViT-Transformer Text Recognition")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["encoder", "decoder", "all"],
        help="Which model to build: 'encoder', 'decoder', or 'all'",
    )
    parser.add_argument("--onnx-dir", type=str, default="weights/onnx", help="Directory containing exported ONNX files")
    parser.add_argument("--output-dir", type=str, default="weights/tensorrt", help="Output directory for .engine files")
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 precision (default: True)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable FP16 (use FP32)")
    parser.add_argument("--workspace-mb", type=int, default=2048, help="TensorRT workspace memory in MB (default: 2048)")
    parser.add_argument("--force-python", action="store_true", help="Force using TensorRT Python API instead of trtexec")
    parser.add_argument("--dry-run", action="store_true", help="Print build commands without executing compilation")
    args = parser.parse_args()

    targets = ["encoder", "decoder"] if args.model == "all" else [args.model]
    success = True

    for target in targets:
        onnx_file = os.path.join(args.onnx_dir, f"{target}.onnx")
        engine_file = os.path.join(args.output_dir, f"{target}.engine")
        print(f"\n=======================================================")
        print(f" Compiling {target.upper()} Engine ({onnx_file} -> {engine_file})")
        print(f"=======================================================")
        ok = compile_model(
            model_name=target,
            onnx_path=onnx_file,
            engine_path=engine_file,
            fp16=args.fp16,
            workspace_mb=args.workspace_mb,
            dry_run=args.dry_run,
            force_python=args.force_python,
        )
        if not ok:
            success = False
            print(f"[ERROR] Failed to compile {target} engine!")

    if success:
        print("\nAll target TensorRT engines processed successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
