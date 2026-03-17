"""FaceCam inference script for WAN video generation."""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict, load_state_dict_from_folder
from diffsynth.core.vram.layers import (
    enable_vram_management as _enable_vram,
    AutoWrappedModule, AutoWrappedLinear, AutoWrappedNonRecurseModule,
)
from diffsynth.pipelines.wan_video_facecam import WanVideoPipeline
from diffsynth.utils.data import save_video
from diffsynth.utils.gaussians_renderer import get_proxy_video
from diffsynth.utils.mediapipe_utils import crop_video, get_mediapipe_cond, set_face_landmarker_path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = "WARN"

# Supported model configurations. Each entry maps a model name to its
# base-model file layout and default finetuned checkpoint subdirectory.
MODEL_REGISTRY = {
    "wan2.2-14b": {
        "model_subdir": "Wan-AI/Wan2.2-I2V-A14B",
        "model_files": [
            "high_noise_model/diffusion_pytorch_model*.safetensors",
            "low_noise_model/diffusion_pytorch_model*.safetensors",
            "models_t5_umt5-xxl-enc-bf16.pth",
            "Wan2.1_VAE.pth",
        ],
        "ckpt_subdir": "wan2.2_14b",
        "dit_ckpt": "high/released_version",
        "dit2_ckpt": "low/released_version",
    },
}


# =============================================================================
# Inference Module
# =============================================================================

class WanInferenceModule(torch.nn.Module):
    """Lightweight wrapper around WanVideoPipeline for inference."""

    def __init__(self, model_paths, device="cuda", low_vram=False):
        super().__init__()

        if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, str) and device == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        self._target_device = device
        self._low_vram = low_vram

        # Always load to CPU first so that finetuned checkpoints can be
        # applied to the *unwrapped* model.  VRAM management wrapping is
        # deferred to setup_device() which must be called after all
        # load_checkpoint() calls.
        load_device = "cpu" if low_vram else device

        model_configs = []
        for path in model_paths:
            model_configs.append(ModelConfig(path=self._expand_glob(path)))

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=load_device,
            model_configs=model_configs,
        )

    @staticmethod
    def _expand_glob(path):
        """Expand shell glob patterns in *path* to a file list."""
        if isinstance(path, list):
            expanded = []
            for p in path:
                r = WanInferenceModule._expand_glob(p)
                expanded.extend(r if isinstance(r, list) else [r])
            return expanded if len(expanded) > 1 else expanded[0]

        if any(c in path for c in ("*", "?", "[")):
            matches = sorted(glob.glob(path))
            if not matches:
                raise FileNotFoundError(f"No files match pattern: {path}")
            return matches if len(matches) > 1 else matches[0]
        return path

    def load_checkpoint(self, checkpoint_path, model_name="dit"):
        """Load a finetuned checkpoint into the named sub-model."""
        if checkpoint_path is None:
            return
        if os.path.isfile(checkpoint_path):
            state_dict = load_state_dict(checkpoint_path)
        else:
            state_dict = load_state_dict_from_folder(checkpoint_path)

        model = getattr(self.pipe, model_name, None)
        if model is not None:
            load_results = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint into {model_name}: {checkpoint_path}")
            print(f"Missing keys: {len(load_results.missing_keys)}, Unexpected keys: {len(load_results.unexpected_keys)}")
        else:
            print(f"Warning: model '{model_name}' not found in pipeline")

    def setup_device(self):
        """Move models to target device. Must be called after all load_checkpoint() calls.

        When low_vram is enabled, this wraps model sub-modules with VRAM
        management wrappers *after* finetuned weights have been loaded into the
        unwrapped model, guaranteeing that the state-dict keys match.
        """
        if self._low_vram:
            self._enable_vram_management()
        # Pipeline's self.device and self.device_type control where tensors
        # are placed at runtime and how device-specific ops (e.g. empty_cache)
        # are dispatched.
        from diffsynth.core import parse_device_type
        self.pipe.device = self._target_device
        self.pipe.device_type = parse_device_type(self._target_device)

    def _enable_vram_management(self):
        from diffsynth.models.wan_video_dit import RMSNorm, MLP, DiTBlock, Head
        from diffsynth.models.wan_video_vae import RMS_norm, CausalConv3d, Upsample
        from diffsynth.models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm

        device_str = str(self._target_device)
        vram_limit = torch.cuda.mem_get_info(self._target_device)[1] / (1024 ** 3) - 2

        base_cfg = dict(
            offload_dtype=torch.bfloat16,
            offload_device="cpu",
            onload_dtype=torch.bfloat16,
            onload_device="cpu",
            preparing_dtype=torch.bfloat16,
            preparing_device=device_str,
            computation_dtype=torch.bfloat16,
            computation_device=device_str,
        )

        dit_map = {
            MLP: AutoWrappedModule,
            DiTBlock: AutoWrappedNonRecurseModule,
            Head: AutoWrappedModule,
            torch.nn.Linear: AutoWrappedLinear,
            torch.nn.Conv3d: AutoWrappedModule,
            torch.nn.LayerNorm: AutoWrappedModule,
            RMSNorm: AutoWrappedModule,
            torch.nn.Conv2d: AutoWrappedModule,
        }

        vae_map = {
            torch.nn.Linear: AutoWrappedLinear,
            torch.nn.Conv2d: AutoWrappedModule,
            RMS_norm: AutoWrappedModule,
            CausalConv3d: AutoWrappedModule,
            Upsample: AutoWrappedModule,
            torch.nn.SiLU: AutoWrappedModule,
            torch.nn.Dropout: AutoWrappedModule,
        }

        text_enc_map = {
            torch.nn.Linear: AutoWrappedLinear,
            torch.nn.Embedding: AutoWrappedModule,
            T5RelativeEmbedding: AutoWrappedModule,
            T5LayerNorm: AutoWrappedModule,
        }

        if self.pipe.text_encoder is not None:
            self.pipe.text_encoder = _enable_vram(
                self.pipe.text_encoder, text_enc_map,
                vram_config=base_cfg, vram_limit=vram_limit,
            )
        if self.pipe.dit is not None:
            self.pipe.dit = _enable_vram(
                self.pipe.dit, dit_map,
                vram_config=base_cfg, vram_limit=vram_limit,
            )
        if self.pipe.dit2 is not None:
            self.pipe.dit2 = _enable_vram(
                self.pipe.dit2, dit_map,
                vram_config=base_cfg, vram_limit=vram_limit,
            )
        if self.pipe.vae is not None:
            self.pipe.vae = _enable_vram(
                self.pipe.vae, vae_map,
                vram_config=base_cfg,
            )

        self.pipe.vram_management_enabled = True
        print(f"VRAM management enabled (limit: {vram_limit:.1f} GiB)")

    @torch.no_grad()
    def generate(
        self,
        prompt,
        negative_prompt="",
        video_cond=None,
        camera_cond=None,
        height=704,
        width=480,
        num_frames=81,
        seed=None,
        num_inference_steps=50,
        cfg_scale=5.0,
        tiled=False,
        switch_DiT_boundary=0.9,
    ):
        """Run a single generation through the pipeline."""
        return self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            video_cond=video_cond,
            camera_cond=camera_cond,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            tiled=tiled,
            switch_DiT_boundary=switch_DiT_boundary,
        )

    def inference(self, dataset, output_dir, process_index=0, num_processes=1,
                  num_inference_steps=50, seed=None):
        """Run inference over *dataset*, distributing work across processes."""
        os.makedirs(output_dir, exist_ok=True)

        indices = list(range(len(dataset)))[process_index::num_processes]
        if num_processes > 1:
            print(f"Process {process_index}/{num_processes}: {len(indices)} samples")

        for idx in tqdm(indices, desc=f"GPU {process_index}"):
            data = dataset[idx]
            video_name = data.get("video_name", f"sample_{idx}")

            save_video(data["video_cond"],
                        os.path.join(output_dir, f"{video_name}_input.mp4"),
                        fps=dataset.fps, quality=5)
            save_video(data["camera_cond"],
                        os.path.join(output_dir, f"{video_name}_camera.mp4"),
                        fps=dataset.fps, quality=5)

            video = self.generate(
                prompt=data["prompt"],
                negative_prompt=data["negative_prompt"],
                video_cond=data["video_cond"],
                camera_cond=data["camera_cond"],
                height=dataset.height,
                width=dataset.width,
                num_frames=dataset.num_frames,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )

            save_video(video, os.path.join(output_dir, f"{video_name}.mp4"),
                       fps=dataset.fps, quality=5)



# =============================================================================
# Dataset
# =============================================================================

class InferenceDataset(torch.utils.data.Dataset):
    """Dataset that loads local videos and generates random camera conditions."""

    def __init__(self, input_path, num_frames, width, height, fps, ply_path=None):
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.fps = fps
        self.ply_path = ply_path

        if os.path.isfile(input_path):
            self.data = [input_path]
        elif os.path.isdir(input_path):
            self.data = sorted(
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith((".mp4", ".mov"))
            )
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if not self.data:
            raise RuntimeError(f"No video files found in {input_path}")

    def load_video(self, video_path):
        """Load, sub-sample, crop and pad a video to target specs."""
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(orig_fps / self.fps)) if orig_fps > 0 else 1

        frames, frame_idx = [], 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                frames.append(frame)
                if len(frames) >= self.num_frames:
                    break
            frame_idx += 1
        cap.release()

        frames = crop_video(frames, self.height, self.width)
        frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        return frames

    def create_camera_cond(self, camera_params):
        """Render a proxy-video camera condition from *camera_params*."""
        min_size = min(self.height, self.width)
        kwargs = dict(camera_params=camera_params, num_frames=self.num_frames,
                      render_res=min_size)
        if self.ply_path is not None:
            kwargs["ply_path"] = self.ply_path
        proxy_video = get_proxy_video(**kwargs)

        camera_cond = []
        for proxy_frame in proxy_video:
            proxy_frame = np.pad(
                proxy_frame,
                ((0, self.height - min_size), (0, self.width - min_size), (0, 0)),
                mode="constant",
                constant_values=255,
            )
            camera_cond.append(Image.fromarray(get_mediapipe_cond(proxy_frame)))
        return camera_cond

    @staticmethod
    def random_camera_params(
        max_azimuth=45, max_elevation=30, max_fov=15,
        base_azimuth=0, base_elevation=0, base_fov=40,
        large_pose=True,
    ):
        if large_pose:
            direction = 1 if np.random.rand() < 0.5 else -1
            az_start = np.random.uniform(
                base_azimuth - direction * max_azimuth,
                base_azimuth - direction * (max_azimuth - 15))
            az_end = np.random.uniform(
                base_azimuth + direction * max_azimuth,
                base_azimuth + direction * (max_azimuth - 15))
        else:
            az_start = np.random.uniform(base_azimuth - max_azimuth,
                                         base_azimuth + max_azimuth)
            az_end = np.random.uniform(base_azimuth - max_azimuth,
                                       base_azimuth + max_azimuth)

        return {
            "start_azimuth": az_start,
            "end_azimuth": az_end,
            "start_elevation": np.random.uniform(base_elevation - max_elevation,
                                                 base_elevation + max_elevation),
            "end_elevation": np.random.uniform(base_elevation - max_elevation,
                                               base_elevation + max_elevation),
            "start_fov": np.random.uniform(base_fov - max_fov, base_fov + max_fov),
            "end_fov": np.random.uniform(base_fov - max_fov, base_fov + max_fov),
        }

    def __getitem__(self, index):
        video_path = self.data[index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames = self.load_video(video_path)
        camera_params = self.random_camera_params(large_pose=True)
        camera_cond = self.create_camera_cond(camera_params)

        return {
            "video_cond": frames,
            "camera_cond": camera_cond,
            "prompt": "A portrait of a person",
            "negative_prompt": "",
            "camera_params": camera_params,
            "video_name": video_name,
        }

    def __len__(self):
        return len(self.data)


# =============================================================================
# CLI helpers
# =============================================================================

def resolve_model_paths(model_name, model_dir):
    """Return the list of base-model file paths for *model_name*."""
    cfg = MODEL_REGISTRY[model_name]
    base = os.path.join(model_dir, cfg["model_subdir"])
    return [os.path.join(base, f) for f in cfg["model_files"]]


def resolve_ckpt_paths(model_name, ckpt_dir):
    """Return (dit_ckpt_path, dit2_ckpt_path) for *model_name*."""
    cfg = MODEL_REGISTRY[model_name]
    base = os.path.join(ckpt_dir, cfg["ckpt_subdir"])
    dit_ckpt = os.path.join(base, cfg["dit_ckpt"]) if cfg["dit_ckpt"] else None
    dit2_ckpt = os.path.join(base, cfg["dit2_ckpt"]) if cfg["dit2_ckpt"] else None
    return dit_ckpt, dit2_ckpt


def parse_args():
    parser = argparse.ArgumentParser(
        description="FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_dir", type=str, default="./models",
        help="Root directory containing base model weights.",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="./ckpts",
        help="Root directory containing FaceCam assets (checkpoints, gaussians.ply, face_landmarker).",
    )

    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a video file or directory of videos.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save generated videos.")

    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=704,
                        help="Output height.")
    parser.add_argument("--width", type=int, default=480,
                        help="Output width.")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable CPU offloading to reduce GPU memory usage.")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"GPUs: {accelerator.num_processes}")
        print(f"Resolution: {args.height}x{args.width}  |  Frames: {args.num_frames}")

    # Resolve paths from the model registry
    model_name = "wan2.2-14b"
    model_paths = resolve_model_paths(model_name, args.model_dir)
    dit_ckpt, dit2_ckpt = resolve_ckpt_paths(model_name, args.ckpt_dir)
    set_face_landmarker_path(os.path.join(args.ckpt_dir, "face_landmarker_v2_with_blendshapes.task"))

    # Build model — load base weights, apply finetuned checkpoints, then
    # enable VRAM management (if low_vram).  This order guarantees that
    # checkpoint state-dict keys match the unwrapped model.
    model = WanInferenceModule(model_paths=model_paths, device=device, low_vram=args.low_vram)
    model.load_checkpoint(dit_ckpt, model_name="dit")
    model.load_checkpoint(dit2_ckpt, model_name="dit2")
    model.setup_device()

    # Build dataset
    ply_path = os.path.join(args.ckpt_dir, "gaussians.ply")
    dataset = InferenceDataset(
        input_path=args.input_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.fps,
        ply_path=ply_path,
    )

    # Run inference
    model.inference(
        dataset,
        args.output_dir,
        process_index=accelerator.process_index,
        num_processes=accelerator.num_processes,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"Done. Results saved to: {args.output_dir}")
