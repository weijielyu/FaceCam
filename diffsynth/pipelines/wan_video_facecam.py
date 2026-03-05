"""FaceCam pipeline for WAN video generation with video and camera conditioning."""

import torch
import types
import numpy as np
from PIL import Image
from einops import repeat, rearrange
from typing import Optional, Union
from typing_extensions import Literal
from tqdm import tqdm

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_motion_controller import WanMotionControllerModel


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
            time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.in_iteration_models = ("dit", "motion_controller")
        self.in_iteration_models_2 = ("dit2", "motion_controller")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_FaceCam(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.model_fn = model_fn_wan_video

    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        # Redirect model path
        # if redirect_common_files:
        #     redirect_dict = {
        #         "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
        #         "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
        #         "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
        #         "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
        #     }
        #     for model_config in model_configs:
        #         if model_config.origin_file_pattern is None or model_config.model_id is None:
        #             continue
        #         if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
        #             print(f"Redirecting ({model_config.model_id}, {model_config.origin_file_pattern}) to {redirect_dict[model_config.origin_file_pattern]}")
        #             model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
        #             model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        
        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # FaceCam conditioning
        video_cond: Optional[list[Image.Image]] = None,
        camera_cond: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # DiT switching boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # TeaCache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # Progress bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "video_cond": video_cond,
            "camera_cond": camera_cond,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
        }
        
        # Run pipeline units
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        tgt_latent_length = (num_frames - 1) // 4 + 1
        has_facecam_cond = video_cond is not None or camera_cond is not None
        
        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and models["dit"] is not self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Concatenate video condition latents for FaceCam
            if has_facecam_cond and inputs_shared.get("video_cond_latents") is not None:
                inputs_shared["video_cond_latents"] = inputs_shared["video_cond_latents"].to(self.device)
                inputs_shared["latents"] = torch.cat(
                    (inputs_shared["latents"], inputs_shared["video_cond_latents"]), dim=2
                )
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler step
            if has_facecam_cond:
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred[:, :, :tgt_latent_length, ...],
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"][:, :, :tgt_latent_length, ...]
                )
            else:
                inputs_shared["latents"] = self.scheduler.step(
                    noise_pred,
                    self.scheduler.timesteps[progress_id],
                    inputs_shared["latents"]
                )
        
        # Decode
        self.load_models_to_device(['vae'])
        if has_facecam_cond:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, :tgt_latent_length, ...]
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        
        self.load_models_to_device([])
        return video


# =============================================================================
# Pipeline Units
# =============================================================================

class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}


class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device"),
            output_params=("noise", "latents")
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise, "latents": noise}


class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_FaceCam(PipelineUnit):
    """Process video_cond and camera_cond into visual conditioning tensor y."""
    
    def __init__(self):
        super().__init__(
            input_params=("video_cond", "camera_cond", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "y", "latents"),
            output_params=("y", "video_cond_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, video_cond, camera_cond, num_frames, height, width, tiled, tile_size, tile_stride, y, latents):
        if video_cond is None and camera_cond is None:
            return {}
            
        pipe.load_models_to_device(self.onload_model_names)
        
        # Process video_cond
        if video_cond is not None:
            video_cond_tensor = pipe.preprocess_video(video_cond)
            video_cond_latents = pipe.vae.encode(
                video_cond_tensor, 
                device=pipe.device, 
                tiled=tiled, 
                tile_size=tile_size, 
                tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        else:
            video_cond_latents = torch.zeros_like(latents)
        
        # Process camera_cond
        if camera_cond is not None:
            camera_cond_tensor = pipe.preprocess_video(camera_cond)
            camera_cond_latents = pipe.vae.encode(
                camera_cond_tensor, 
                device=pipe.device, 
                tiled=tiled, 
                tile_size=tile_size, 
                tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
        else:
            camera_cond_latents = torch.zeros_like(latents)

        # Build y tensor with camera_cond as channel conditioning
        y_dim = pipe.dit.in_dim - camera_cond_latents.shape[1] - latents.shape[1]
        if y is None:
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.cat([camera_cond_latents, y], dim=1)
        
        # Double y length for frame conditioning
        if video_cond is not None:
            y = y.repeat(1, 1, 2, 1, 1)

        return {"y": y, "video_cond_latents": video_cond_latents}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel") and pipe.use_unified_sequence_parallel:
            return {"use_unified_sequence_parallel": True}
        return {}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}


class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared.get("cfg_merge", False):
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


# =============================================================================
# TeaCache
# =============================================================================

class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported = ", ".join(self.coefficients_dict.keys())
            raise ValueError(f"{model_id} is not supported. Choose from: {supported}")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(self.coefficients)
            rel_diff = ((modulated_inp - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item()
            self.accumulated_rel_l1_distance += rescale_func(rel_diff)
            should_calc = self.accumulated_rel_l1_distance >= self.rel_l1_thresh
            if should_calc:
                self.accumulated_rel_l1_distance = 0
        
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        return hidden_states + self.previous_residual


# =============================================================================
# Temporal Tiler
# =============================================================================

class TemporalTiler_BCTHW:
    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        return repeat(t, "T -> 1 1 T 1 1")
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [name for name in tensor_names if model_kwargs.get(name) is not None]
        tensor_dict = {name: model_kwargs[name] for name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                name: tensor_dict[name][:, :, t:t_, :].to(device=computation_device, dtype=computation_dtype)
                for name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


# =============================================================================
# Model Function
# =============================================================================

def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    # Sliding window
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit, motion_controller=motion_controller,
            latents=latents, timestep=timestep, context=context,
            clip_feature=clip_feature, y=y, tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    
    context = dit.text_embedding(context)
    x = latents
    
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embedding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)
    
    # Patchify
    x = dit.patchify(x, None)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # TeaCache
    tea_cache_update = tea_cache.check(dit, x, t_mod) if tea_cache is not None else False
    
    # Unified sequence parallel
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
        pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
        chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1] - chunk.shape[1]), value=0) for chunk in chunks]
        x = chunks[get_sequence_parallel_rank()]
    
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, context, t_mod, freqs, use_reentrant=False)
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(create_custom_forward(block), x, context, t_mod, freqs, use_reentrant=False)
            else:
                x = block(x, context, t_mod, freqs)
        
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    
    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)
        x = x[:, :-pad_shape] if pad_shape > 0 else x
    
    x = dit.unpatchify(x, (f, h, w))
    return x
