"""In-the-wild video dataset for human video data."""

import os
import random
from typing import Dict, List, Optional, Any

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from diffsynth.utils.mediapipe_utils import get_mediapipe_cond, crop_video

# Default local data root; override via IN_THE_WILD_DATA_ROOT env or pass data_root explicitly
DEFAULT_IN_THE_WILD_DATA_ROOT = os.environ.get("IN_THE_WILD_DATA_ROOT", "/path/to/in_the_wild_videos")


class InTheWildDataset(Dataset):
    """In-the-wild human video dataset.
    
    Args:
        data_root: Local root directory for video files.
        num_frames: Number of frames to load per video.
        width: Target frame width.
        height: Target frame height.
        min_scale: Minimum scale for augmentation.
        max_scale: Maximum scale for augmentation.
        split: Dataset split ('train', 'val', or 'all').
        debug_mode: If True, only load first 100 samples.
    """

    def __init__(
        self,
        data_root: str,
        num_frames: int = 81,
        width: int = 480,
        height: int = 704,
        min_scale: float = 1.0,
        max_scale: float = 1.5,
        split: str = "train",
        debug_mode: bool = False,
    ):
        self.data_root = data_root.rstrip('/')
        self.num_frames = num_frames
        self.width = width
        self.height = height
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.split = split
        self.debug_mode = debug_mode

        self.data = self._build_samples()

    def _list_dir(self, path: str) -> List[str]:
        """List directory contents."""
        return os.listdir(path)

    def _build_samples(self) -> List[str]:
        """Build sample list from directory."""
        samples = self._list_dir(self.data_root)
        
        # Filter video files
        samples = [s for s in samples if s.endswith(('.mp4', '.mov'))]

        # Split dataset
        if self.split == "train":
            samples = samples[:int(len(samples) * 0.98)]
        elif self.split == "val":
            samples = samples[int(len(samples) * 0.98):]
            random.shuffle(samples)

        if self.debug_mode:
            samples = samples[:100]

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
        video_path = self.data[index % len(self.data)]

        # Load frames
        frames = self._load_video(video_path)
        
        # Get pose condition from first frame
        first_frame = np.array(frames[0])
        pose_cond = get_mediapipe_cond(first_frame)
        pose_conds = [Image.fromarray(pose_cond)] * len(frames)

        # Apply augmentation
        cond_frames = self._augment_video(frames, type="still")
        
        aug_params = {
            "start_scale": random.uniform(self.min_scale, self.max_scale),
            "end_scale": random.uniform(self.min_scale, self.max_scale),
            "start_h": random.uniform(0, 1),
            "start_w": random.uniform(0, 1),
            "end_h": random.uniform(0, 1),
            "end_w": random.uniform(0, 1),
        }
        target_frames = self._augment_video(frames, type="moving", aug_params=aug_params)
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

    def _load_video(self, video_path: str) -> List[Image.Image]:
        """Load video frames from local path."""
        frames = []
        full_path = f"{self.data_root}/{video_path}" if not video_path.startswith('/') else video_path

        cap = cv2.VideoCapture(full_path)
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError(f"No frames loaded from video: {full_path}")

        # Crop to target size
        frames = crop_video(frames, target_height=self.height, target_width=self.width)
        frames = [Image.fromarray(frame) for frame in frames]

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

    dataset = InTheWildDataset(
        data_root=DEFAULT_IN_THE_WILD_DATA_ROOT,
        num_frames=81,
        split="train",
        debug_mode=True,
    )

    if len(dataset) > 0:
        sample = dataset[0]
        os.makedirs("in_the_wild_sample_output", exist_ok=True)
        for key in ["video", "video_cond", "pose_cond"]:
            if key in sample and sample[key]:
                save_video(sample[key], f"in_the_wild_sample_output/{key}.mp4", fps=24)
