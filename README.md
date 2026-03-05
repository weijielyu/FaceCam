# FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning

[Weijie Lyu](https://weijielyu.github.io/), [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/), [Zhixin Shu](https://zhixinshu.github.io/)  
University of California, Merced - Adobe Research

[![Website](https://img.shields.io/badge/Website-FaceCam?logo=googlechrome&logoColor=hsl(204%2C%2086%25%2C%2053%25)&label=FaceCam&labelColor=%23f5f5dc&color=hsl(204%2C%2086%25%2C%2053%25))](https://weijielyu.github.io/FaceCam)
[![Paper](https://img.shields.io/badge/Paper-arXiv?logo=arxiv&logoColor=%23B31B1B&label=arXiv&labelColor=%23f5f5dc&color=%23B31B1B)](https://arxiv.org/abs/2412.17812)
[![Video](https://img.shields.io/badge/Video-YouTube?logo=youtube&logoColor=%23FF0000&label=YouTube&labelColor=%23f5f5dc&color=%23FF0000)](https://youtu.be/QGJcwo44ziU)


<div align='center'>
<img alt="image" src='media/teaser.png'>
</div>

> *FaceCam* generates portrait videos with precise camera control from a single input video and a target camera trajectory.

## 🔧 Prerequisites

### Environment Setup

```bash
conda create -n facecam python=3.11 -y
conda activate facecam

# Install the package (includes core dependencies)
pip install -e .

# Additional required packages
pip install xformers # Choose a version which is compatible with your PyTorch
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization --no-build-isolation
pip install mediapipe==0.10.21
```

### Downloads

We support the Wan 2.2 14B model. Create the directory and download all required assets:

| Model | Base checkpoint | Notes |
|---|---|---|
| **Wan 2.2 14B** (`wan2.2-14b`) | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | Dual-DiT (high/low noise), highest quality |

```bash
mkdir -p models ckpts
```

**1. Base model weights** (via ModelScope):

```bash
pip install modelscope
modelscope download --model Wan-AI/Wan2.2-I2V-A14B --local_dir ./models/Wan-AI/Wan2.2-I2V-A14B
```

**2. FaceCam assets** (checkpoints, proxy 3D head, face landmarker) from [Hugging Face](https://huggingface.co/wlyu/FaceCam):

```bash
pip install huggingface_hub
huggingface-cli download wlyu/FaceCam --local-dir ./ckpts
wget -O ckpts/face_landmarker_v2_with_blendshapes.task -q \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Alternatively, download from Google Drive: [checkpoints](https://drive.google.com/file/d/1MKBsq5Nvn6EqSQAd8JwIPwxzSRyjjc-_/view?usp=sharing) and [gaussians.ply](https://drive.google.com/file/d/16nAtjP6_vPNWBFCkYdlK5EKxXFWbddhQ/view?usp=drive_link).

The expected layout:

```
models/
└── Wan-AI/
    └── Wan2.2-I2V-A14B/
        ├── high_noise_model/
        ├── low_noise_model/
        ├── models_t5_umt5-xxl-enc-bf16.pth
        └── Wan2.1_VAE.pth

ckpts/
├── face_landmarker_v2_with_blendshapes.task
├── gaussians.ply
└── wan2.2_14b/
    ├── high/released_version/
    └── low/released_version/
```

## 🚀 Inference

### Single GPU

```bash
# Wan 2.2 14B (default 704×480, 81 frames)
python inference.py \
    --model_dir ./models \
    --ckpt_dir  ./ckpts \
    --input_path ./inputs \
    --output_dir ./outputs
```

`--input_path` accepts either a single `.mp4`/`.mov` file or a directory of videos.

By default, the code generates a random camera trajectory. To use a specific trajectory instead, you can customize the `random_camera_params` function in `inference.py`.

### Multi-GPU

Use `accelerate` to distribute samples across GPUs:

```bash
accelerate launch --num_processes 4 inference.py \
    --model_dir ./models \
    --ckpt_dir  ./ckpts \
    --input_path ./inputs \
    --output_dir ./outputs
```

### Low-VRAM Mode

For GPUs with limited memory (e.g. running with 48GB VRAM), enable CPU offloading
so that only the active model component stays on GPU:

```bash
python inference.py \
    --model_dir ./models \
    --ckpt_dir  ./ckpts \
    --input_path ./inputs \
    --output_dir ./outputs \
    --low_vram
```

This trades speed for memory — the text encoder, DiTs, and VAE are moved between
CPU and GPU as needed instead of keeping everything resident.

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | `./models` | Root directory of base model weights |
| `--ckpt_dir` | `./ckpts` | Root directory of FaceCam assets (checkpoints, gaussians.ply, face_landmarker) |
| `--input_path` | *(required)* | Video file or directory of videos |
| `--output_dir` | `./outputs` | Where to save results |
| `--num_frames` | `81` | Number of frames to generate |
| `--height` | `704` | Output video height |
| `--width` | `480` | Output video width |
| `--fps` | `24` | Target FPS for input sub-sampling |
| `--num_inference_steps` | `50` | Number of denoising steps |
| `--seed` | random | Random seed for reproducibility |
| `--low_vram` | off | Enable CPU offloading to save GPU memory |

For each input video `<name>.mp4`, the script saves:
- `<name>.mp4` — the generated video
- `<name>_input.mp4` — the cropped input video
- `<name>_camera.mp4` — the camera condition visualisation

## 🎓 Training (Coming Soon)

## 📝 Citation

If you find our work useful for your research, please consider citing our paper:

```bibtex
@InProceedings{Lyu_2025_ICCV,
    author    = {Lyu, Weijie and Zhou, Yi and Yang, Ming-Hsuan and Shu, Zhixin},
    title     = {FaceLift: Learning Generalizable Single Image 3D Face Reconstruction from Synthetic Heads},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {12691-12701}
}
```

## 📄 License

Copyright 2026 Adobe Inc.

Code and model weights are licensed under [Apache-2.0 License](http://www.apache.org/licenses/LICENSE-2.0).

## 🙏 Acknowledgements

This work is built upon [Wan](https://arxiv.org/abs/2503.20314) and [DiffSynth](https://github.com/modelscope/DiffSynth-Studio). We thank the authors for their excellent work.

This is a self-reimplementation of *FaceCam*. The code has been reimplemented and the weights retrained. Results may differ slightly from those reported in the paper.