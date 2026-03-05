"""NeRSemble Dataset for multi-view human video data."""

import os
import random
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms.functional as TF

from diffsynth.utils.mediapipe_utils import get_mediapipe_cond

# Default local data root; override via NERSEMBLE_DATA_ROOT env or pass data_root explicitly
DEFAULT_NERSEMBLE_DATA_ROOT = os.environ.get("NERSEMBLE_DATA_ROOT", "/path/to/nersemble_data")

# Camera serial numbers for different viewpoints
SERIALS = [
    "222200042", "222200044", "222200046", "222200040", "222200036",
    "222200048", "220700191", "222200041", "222200037", "222200038",
    "222200047", "222200043", "222200049", "222200039", "222200045",
    "221501007"
]


class NeRSembleDataset(Dataset):
    """NeRSemble multi-view human video dataset.
    
    Args:
        data_root: Local root directory of the dataset.
        num_frames: Number of frames to sample per video.
        skip: Frame skip interval for sampling.
        num_targets: Maximum number of target views to sample.
        width: Target frame width.
        height: Target frame height.
        min_scale: Minimum scale for augmentation.
        max_scale: Maximum scale for augmentation.
        split: Dataset split ('train', 'val', or 'all').
        debug_mode: If True, only load first 5 participants.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 81,
        skip: int = 3,
        num_targets: int = 1,
        width: int = 480,
        height: int = 704,
        min_scale: float = 0.75,
        max_scale: float = 1.25,
        split: str = "train",
        debug_mode: bool = False,
    ):
        self.data_root = data_root.rstrip('/')
        self.num_frames = num_frames
        self.skip = skip
        self.num_targets = num_targets
        self.width = width
        self.height = height
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.split = split
        self.debug_mode = debug_mode

        self.data = self._build_samples()

    def _check_exists(self, path: str, is_dir: bool = False) -> bool:
        """Check if path exists."""
        return os.path.exists(path)

    def _list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        return os.listdir(path)

    def _load_image(self, path: str) -> Image.Image:
        """Load image from path."""
        return Image.open(path)

    def _build_samples(self) -> List[Dict]:
        """Build sample list from metadata CSV or directory scan."""
        metadata_path = f"{self.data_root}/metadata_sequences.csv"
        if os.path.exists(metadata_path):
            return self._build_samples_from_csv(metadata_path)
        return self._build_samples_from_dir()

    def _build_samples_from_csv(self, metadata_path: str) -> List[Dict]:
        """Build samples from metadata CSV."""
        metadata_df = pd.read_csv(metadata_path)

        sequence_names = [s for s in metadata_df.columns[2:] if s != "BACKGROUND"]
        participant_ids = metadata_df["ID"].unique()

        if self.split == "train":
            participant_ids = participant_ids[:int(len(participant_ids) * 0.98)]
        elif self.split == "val":
            participant_ids = participant_ids[int(len(participant_ids) * 0.98):]
        
        if self.debug_mode:
            participant_ids = participant_ids[:5]

        samples = []
        for p_id in tqdm(participant_ids, desc="Processing participants"):
            for s_name in sequence_names:
                if metadata_df[(metadata_df["ID"] == p_id) & (metadata_df[s_name] == "x")].empty:
                    continue

                sample_path = f"{self.data_root}/{p_id:03d}/sequences/{s_name}"
                use_folder = "masks" if self._check_exists(f"{sample_path}/masks", is_dir=True) else "images"

                samples.append({
                    "participant_id": f"{p_id:03d}",
                    "sequence_name": s_name,
                    "sample_path": sample_path,
                    "use_folder": use_folder,
                })

        return samples

    def _build_samples_from_dir(self) -> List[Dict]:
        """Scan local data directory and build sample list."""
        samples = []

        participant_dirs = sorted([
            d for d in self._list_dir(self.data_root)
            if self._check_exists(f"{self.data_root}/{d}/sequences", is_dir=True)
        ])

        if self.split == "train":
            participant_dirs = participant_dirs[:int(len(participant_dirs) * 0.98)]
        elif self.split == "val":
            participant_dirs = participant_dirs[int(len(participant_dirs) * 0.98):]

        if self.debug_mode:
            participant_dirs = participant_dirs[:5]

        for p_id in tqdm(participant_dirs, desc="Building samples"):
            seq_dir = f"{self.data_root}/{p_id}/sequences"
            for s_name in sorted(self._list_dir(seq_dir)):
                if s_name == "BACKGROUND":
                    continue

                sample_path = f"{seq_dir}/{s_name}"
                use_folder = "masks" if self._check_exists(f"{sample_path}/masks", is_dir=True) else "images"

                samples.append({
                    "participant_id": p_id,
                    "sequence_name": s_name,
                    "sample_path": sample_path,
                    "use_folder": use_folder,
                })

        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        try:
            return self._get_item(index)
        except Exception:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

    def _get_item(self, index: int) -> Dict[str, Any]:
        """Load and process a single sample."""
        data = self.data[index]
        sample_path = data["sample_path"]
        use_folder = data["use_folder"]

        # Select cameras
        cond_camera_id = random.choice(SERIALS)
        sample_num_targets = random.randint(1, self.num_targets)
        target_camera_ids = random.sample(SERIALS, sample_num_targets)

        # Determine start frame and bg_color
        if use_folder != "images":
            cam_path = f"{sample_path}/{use_folder}/cam_{cond_camera_id}"
            total_frames = len([f for f in self._list_dir(cam_path) if f.endswith(('.jpg', '.png'))])
            start_frame = random.randint(0, max(total_frames - self.skip * self.num_frames, 0))
            bg_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
        else:
            start_frame = 0
            bg_color = None

        # Load conditional video
        cond_frames = self._load_frames(
            sample_path, use_folder, cond_camera_id,
            self.num_frames, start_frame, bg_color,
        )

        # Load target videos
        sub_seq_len = self.num_frames // sample_num_targets + 1
        target_frames = []
        pose_conds = []

        for idx, cam_id in enumerate(target_camera_ids):
            sub_start = idx * sub_seq_len * self.skip + start_frame

            frames = self._load_frames(
                sample_path, use_folder, cam_id,
                sub_seq_len, sub_start, bg_color,
                first_frame_idx=start_frame
            )

            # Last frame is the reference for pose
            first_frame = frames[-1]
            frames = frames[:-1]

            # Get pose condition
            pose = get_mediapipe_cond(np.array(first_frame))
            pose_img = Image.fromarray(pose)

            target_frames.extend(frames)
            pose_conds.extend([pose_img] * len(frames))
        
        # Apply augmentation
        cond_frames = self._augment_video(cond_frames, type="still")
        aug_params = {
            "start_scale": random.uniform(1.0, self.max_scale),
            "end_scale": random.uniform(1.0, self.max_scale),
            "start_h": random.uniform(0, 1),
            "start_w": random.uniform(0, 1),
            "end_h": random.uniform(0, 1),
            "end_w": random.uniform(0, 1),
        }
        target_frames = self._augment_video(target_frames, type="moving", aug_params=aug_params)
        pose_conds = self._augment_video(pose_conds, type="moving", aug_params=aug_params)

        # Ensure correct number of frames
        cond_frames = self._pad_or_truncate(cond_frames, self.num_frames)
        target_frames = self._pad_or_truncate(target_frames, self.num_frames)
        pose_conds = self._pad_or_truncate(pose_conds, self.num_frames)

        return {
            "video": target_frames,
            "video_cond": cond_frames,
            "pose_cond": pose_conds,
            "prompt": "A portrait of a person",
        }

    def _load_frames(
        self,
        sample_path: str,
        use_folder: str,
        camera_id: str,
        num_frames: int,
        start_frame: int,
        bg_color: Optional[np.ndarray],
        first_frame_idx: Optional[int] = None,
    ) -> List[Image.Image]:
        """Load and process video frames."""
        frame_indices = list(range(start_frame, start_frame + num_frames * self.skip, self.skip))
        if first_frame_idx is not None:
            frame_indices.append(first_frame_idx)

        if use_folder == "masks":
            frames = self._load_mask_frames(sample_path, camera_id, frame_indices)
        else:
            frames = self._load_video_frames(sample_path, camera_id, frame_indices)

        # Process frames
        processed = []
        for frame in frames:
            frame = TF.resize(frame, size=(self.height, self.width), interpolation=TF.InterpolationMode.BILINEAR)

            if bg_color is not None and frame.mode == "RGBA":
                bg = Image.new("RGB", frame.size, tuple(bg_color.astype(int)))
                frame = Image.alpha_composite(bg.convert("RGBA"), frame).convert("RGB")

            processed.append(frame)

        return processed if processed else [Image.new("RGB", (self.width, self.height), (0, 0, 0))]

    def _load_mask_frames(self, sample_path: str, camera_id: str, frame_indices: List[int]) -> List[Image.Image]:
        """Load RGB + mask frames as RGBA."""
        frames = []
        rgb_folder = f"{sample_path}/rgb_images/cam_{camera_id}"
        mask_folder = f"{sample_path}/masks/cam_{camera_id}"

        for idx in tqdm(frame_indices, desc="Loading mask frames"):
            rgb_path = f"{rgb_folder}/frame_{idx:06d}.jpg"
            mask_path = f"{mask_folder}/mask_{idx:06d}.png"

            if self._check_exists(rgb_path) and self._check_exists(mask_path):
                rgb = self._load_image(rgb_path).convert("RGB")
                mask = self._load_image(mask_path).convert("L")
                rgba = rgb.copy()
                rgba.putalpha(mask)
                frames.append(rgba)

        return frames

    def _load_video_frames(self, sample_path: str, camera_id: str, frame_indices: List[int]) -> List[Image.Image]:
        """Load frames from MP4 video."""
        frames = []
        video_path = f"{sample_path}/images/cam_{camera_id}.mp4"

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [i for i in frame_indices if i < total]

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        cap.release()

        return frames

    def _pad_or_truncate(self, frames: List[Image.Image], target_len: int) -> List[Image.Image]:
        """Ensure list has exactly target_len frames."""
        if len(frames) < target_len:
            last = frames[-1] if frames else Image.new("RGB", (self.width, self.height), (0, 0, 0))
            frames = frames + [last.copy()] * (target_len - len(frames))
        elif len(frames) > target_len:
            frames = frames[:target_len]
        return frames

    def _augment_video(
        self,
        video: List[Image.Image],
        type: str = "still",
        aug_params: Optional[Dict] = None,
    ) -> List[Image.Image]:
        """Apply zoom/pan augmentation to video frames.
        
        Args:
            video: Input frames.
            type: "still" for static augmentation, "moving" for animated zoom/pan.
            aug_params: Augmentation parameters.
                For "still": scale, pos_h, pos_w
                For "moving": start_scale, end_scale, start_h, start_w, end_h, end_w
                Position values in [0, 1] where 0.5 = center.
        """
        if not video:
            return video
        
        if aug_params is None:
            aug_params = {
                "scale": random.uniform(self.min_scale, self.max_scale),
                "pos_h": random.uniform(0, 1),
                "pos_w": random.uniform(0, 1),
            }
        
        # Extract background color from corner patches of first frame
        first_frame = video[0]
        orig_w, orig_h = first_frame.size
        patch_size = min(10, orig_w // 4, orig_h // 4)
        
        upper_left = first_frame.crop((0, 0, patch_size, patch_size))
        upper_right = first_frame.crop((orig_w - patch_size, 0, orig_w, patch_size))
        ul_pixels = np.array(upper_left)
        ur_pixels = np.array(upper_right)
        bg_color = np.mean(np.concatenate([ul_pixels.reshape(-1, 3), ur_pixels.reshape(-1, 3)]), axis=0)
        bg_color_tuple = tuple(bg_color.astype(int))
        
        augmented_frames = []
        num_frames = len(video)
        
        for i, frame in enumerate(video):
            t = i / max(num_frames - 1, 1)
            
            if type == "still":
                scale = aug_params["scale"]
                pos_h = aug_params["pos_h"]
                pos_w = aug_params["pos_w"]
            else:  # type == "moving"
                scale = aug_params["start_scale"] + t * (aug_params["end_scale"] - aug_params["start_scale"])
                pos_h = aug_params["start_h"] + t * (aug_params["end_h"] - aug_params["start_h"])
                pos_w = aug_params["start_w"] + t * (aug_params["end_w"] - aug_params["start_w"])
            
            # Calculate new dimensions
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            
            # Resize frame
            resized = frame.resize((new_w, new_h), Image.BILINEAR) if scale != 1.0 else frame
            
            # Create canvas with background color
            canvas = Image.new('RGB', (orig_w, orig_h), bg_color_tuple)
            
            if scale <= 1.0:  # Shrunk image, need padding
                available_y = orig_h - new_h
                available_x = orig_w - new_w
                y_offset = int(available_y * pos_h)
                x_offset = int(available_x * pos_w)
                canvas.paste(resized, (x_offset, y_offset))
            else:  # Enlarged image, need cropping
                crop_y = int((new_h - orig_h) * pos_h)
                crop_x = int((new_w - orig_w) * pos_w)
                crop_y = max(0, min(crop_y, new_h - orig_h))
                crop_x = max(0, min(crop_x, new_w - orig_w))
                cropped = resized.crop((crop_x, crop_y, crop_x + orig_w, crop_y + orig_h))
                canvas.paste(cropped, (0, 0))
            
            augmented_frames.append(canvas)
        
        return augmented_frames


# Example usage
if __name__ == "__main__":
    from diffsynth import save_video

    dataset = NeRSembleDataset(
        data_root=DEFAULT_NERSEMBLE_DATA_ROOT,
        num_frames=21,
        skip=3,
        num_targets=4,
        split="train",
        debug_mode=True,
    )

    if len(dataset) > 0:
        sample = dataset[0]
        os.makedirs("nersemble_sample_output", exist_ok=True)
        for key in ["video", "video_cond", "pose_cond"]:
            if key in sample and sample[key]:
                save_video(sample[key], f"nersemble_sample_output/{key}.mp4", fps=24)
