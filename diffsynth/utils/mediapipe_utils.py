"""MediaPipe utilities for face landmark detection and mesh visualization."""

from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch.distributed as dist
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configurable path to the MediaPipe face-landmarker model file.
# Call ``set_face_landmarker_path`` before the first ``get_detector`` call
# to override the default.
FACE_LANDMARKER_PATH: str = "./ckpts/face_landmarker_v2_with_blendshapes.task"

_detectors: Dict[int, vision.FaceLandmarker] = {}


def set_face_landmarker_path(path: str) -> None:
    """Override the face-landmarker model path (must be called before first detection)."""
    global FACE_LANDMARKER_PATH
    FACE_LANDMARKER_PATH = path


def get_detector() -> vision.FaceLandmarker:
    """Get or create a face detector for the current process rank."""
    current_rank = dist.get_rank() if dist.is_initialized() else 0

    if current_rank not in _detectors:
        base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        _detectors[current_rank] = vision.FaceLandmarker.create_from_options(options)

    return _detectors[current_rank]


def get_crop_params(
    image: np.ndarray, 
    target_height: int, 
    target_width: int,
    method: str = "nose"
) -> Tuple[int, int, int, int, int, int]:
    """Calculate crop parameters to fit target dimensions while centering on face.
    
    Args:
        image: Input image as numpy array.
        target_height: Target height for the cropped image.
        target_width: Target width for the cropped image.
        method: Centering method ('nose' or 'average').
        
    Returns:
        Tuple of (new_h, new_w, left, right, top, bottom) crop parameters.
    """
    if len(image.shape) != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    
    h, w, _ = image.shape
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image dimensions: {h}x{w}")
    
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Invalid target dimensions: {target_height}x{target_width}")

    # Calculate scale to fit within target dimensions
    scale_h = target_height / h
    scale_w = target_width / w
    scale = max(scale_h, scale_w)

    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))

    # Initialize cropping parameters
    left, right = 0, new_w
    top, bottom = 0, new_h

    # Detect face landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detector = get_detector()
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        # Crop to center if no face landmarks are detected
        if new_w > target_width:
            left = (new_w - target_width) // 2
            right = left + target_width
        if new_h > target_height:
            top = (new_h - target_height) // 2
            bottom = top + target_height
    else:
        # Center around face
        first_landmark = detection_result.face_landmarks[0]
        if method == "nose":
            nose_tip = first_landmark[4]
            center_x = int(nose_tip.x * new_w)
            center_y = int(nose_tip.y * new_h)
        elif method == "average":
            x_list = [landmark.x for landmark in first_landmark]
            y_list = [landmark.y for landmark in first_landmark]
            center_x = int(sum(x_list) / len(x_list) * new_w)
            center_y = int(sum(y_list) / len(y_list) * new_h)
        else:
            raise ValueError(f"Invalid method: {method}")
        
        if new_w > target_width:
            left = max(0, center_x - target_width // 2)
            right = left + target_width
            if right > new_w:
                right = new_w
                left = right - target_width
                
        if new_h > target_height:
            top = max(0, center_y - target_height // 2)
            bottom = top + target_height
            if bottom > new_h:
                bottom = new_h
                top = bottom - target_height

    return new_h, new_w, left, right, top, bottom


def crop_image(
    image: np.ndarray, 
    target_height: int = 704, 
    target_width: int = 480,
    method: str = "nose"
) -> np.ndarray:
    """Crop image to target dimensions centered on face.
    
    Args:
        image: Input image as numpy array.
        target_height: Target height for cropped image.
        target_width: Target width for cropped image.
        method: Centering method ('nose' or 'average').
        
    Returns:
        Cropped image as numpy array.
    """
    new_h, new_w, left, right, top, bottom = get_crop_params(
        image, target_height, target_width, method
    )
    cropped_image = cv2.resize(image, (new_w, new_h))
    cropped_image = cropped_image[top:bottom, left:right]
    return cropped_image


def crop_video(
    frames: List[np.ndarray], 
    target_height: int = 704, 
    target_width: int = 480,
    method: str = "nose"
) -> List[np.ndarray]:
    """Crop video frames to target dimensions using consistent parameters.
    
    Args:
        frames: List of video frames as numpy arrays.
        target_height: Target height for cropped frames.
        target_width: Target width for cropped frames.
        method: Centering method ('nose' or 'average').
        
    Returns:
        List of cropped frames as numpy arrays.
    """
    if not frames:
        raise ValueError("Empty frames list provided")
    
    if not all(isinstance(frame, np.ndarray) for frame in frames):
        raise ValueError("All frames must be numpy arrays")
    
    new_h, new_w, left, right, top, bottom = get_crop_params(
        frames[0], target_height, target_width, method
    )
    
    cropped_frames = []
    for frame in frames:
        resized_frame = cv2.resize(frame, (new_w, new_h))
        cropped_frame = resized_frame[top:bottom, left:right]
        cropped_frames.append(cropped_frame)

    return cropped_frames


def crop_reference_image(
    image: np.ndarray,
    target_height: int = 640,
    target_width: int = 448,
    crop_params: Optional[Tuple[int, int, int, int, int, int]] = None
) -> np.ndarray:
    """Crop reference image using provided or calculated crop parameters.
    
    Args:
        image: Input image as numpy array.
        target_height: Target height for cropped image.
        target_width: Target width for cropped image.
        crop_params: Optional pre-calculated crop parameters.
        
    Returns:
        Cropped image as numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be a numpy array")
    
    if crop_params is None:
        crop_params = get_crop_params(image, target_height, target_width)
    
    if len(crop_params) != 6:
        raise ValueError(f"Expected 6 crop parameters, got {len(crop_params)}")
    
    new_h, new_w, left, right, top, bottom = crop_params
    
    resized_image = cv2.resize(image, (new_w, new_h))
    cropped_image = resized_image[top:bottom, left:right]
    
    return cropped_image


def detect_face_landmarks(numpy_image: np.ndarray) -> List:
    """Detect face landmarks from image.
    
    Args:
        numpy_image: Input image as numpy array.
        
    Returns:
        List of face landmarks.
        
    Raises:
        ValueError: If no face landmarks are detected.
    """
    if not isinstance(numpy_image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    if len(numpy_image.shape) != 3 or numpy_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {numpy_image.shape}")
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    detector = get_detector()
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        raise ValueError("No face landmarks detected in the image")
    
    return detection_result.face_landmarks[0]


def draw_mediapipe_mesh(
    annotated_image: np.ndarray, 
    face_landmarks: List
) -> np.ndarray:
    """Draw MediaPipe face mesh on the image.
    
    Args:
        annotated_image: Image to draw on.
        face_landmarks: List of face landmarks.
        
    Returns:
        Annotated image with face mesh drawn.
    """
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
        for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
    )
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
    )

    return annotated_image


def get_mediapipe_cond(image: np.ndarray) -> np.ndarray:
    """Generate face mesh conditioning image.
    
    Args:
        image: Input image as numpy array.
        
    Returns:
        White image with face mesh drawn.
        
    Raises:
        ValueError: If face detection fails.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    
    # Create blank white image
    annotated_image = np.ones_like(image) * 255
    
    # Detect landmarks and draw mesh
    face_landmarks = detect_face_landmarks(image)
    annotated_image = draw_mediapipe_mesh(annotated_image, face_landmarks)

    return annotated_image
