"""Gaussian splatting renderer for proxy video generation.

This module provides rendering utilities for generating proxy videos
from 3D Gaussian models.

Dependencies:
    pip install numpy einops plyfile
    pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization
"""

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import rearrange
from plyfile import PlyData
from torch import nn

# =============================================================================
# Spherical Harmonics Constants
# =============================================================================

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601, -0.6690465435572892, 0.10578554691520431, -0.6690465435572892, 0.47308734787878004, -1.7701307697799304, 0.6258357354491761]


# =============================================================================
# Matrix Utilities
# =============================================================================

def strip_lowerdiag(L):
    """Extract lower diagonal elements from 3x3 matrices."""
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def build_rotation(r):
    """Build rotation matrices from quaternions."""
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    """Build combined scaling-rotation matrices."""
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    return R @ L


# =============================================================================
# Camera
# =============================================================================

class Camera(nn.Module):
    """Camera model with OpenCV convention."""

    def __init__(self, C2W, fxfycxcy, h, w):
        super().__init__()
        self.C2W = C2W.clone().float()
        self.W2C = self.C2W.inverse()
        self.h, self.w = h, w
        self.znear, self.zfar = 0.01, 100.0

        fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
        self.tanfovX = w / (2 * fx)
        self.tanfovY = h / (2 * fy)

        P = torch.zeros(4, 4, device=fx.device)
        P[0, 0] = 2 * fx / w
        P[1, 1] = 2 * fy / h
        P[0, 2] = 2 * (cx / w) - 1
        P[1, 2] = 2 * (cy / h) - 1
        P[2, 2] = -(self.zfar + self.znear) / (self.zfar - self.znear)
        P[3, 2] = 1.0
        P[2, 3] = -(2 * self.zfar * self.znear) / (self.zfar - self.znear)

        self.world_view_transform = self.W2C.transpose(0, 1)
        self.projection_matrix = P.transpose(0, 1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.C2W[:3, 3]


@torch.no_grad()
def get_proxy_video_cameras(
    num_views=81, w=480, h=480, radius=2.7,
    base_azimuth=270, base_elevation=0,
    up_vector=np.array([0, 0, 1]),
    start_azimuth=-40, end_azimuth=40,
    start_elevation=-20, end_elevation=20,
    start_fov=25, end_fov=45,
):
    """Generate camera parameters for proxy video rendering."""
    azimuths = np.linspace(base_azimuth + start_azimuth, base_azimuth + end_azimuth, num_views)
    elevations = np.linspace(base_elevation + start_elevation, base_elevation + end_elevation, num_views)
    hfovs = np.linspace(start_fov, end_fov, num_views)

    fxs = w / (2 * np.tan(np.deg2rad(hfovs) / 2.0))
    fxfycxcy = np.stack([fxs, fxs, np.full(num_views, w / 2.0), np.full(num_views, h / 2.0)], axis=1)

    c2ws = []
    for elev, azim in zip(elevations, azimuths):
        elev, azim = np.deg2rad(elev), np.deg2rad(azim)
        z = radius * np.sin(elev)
        base = radius * np.cos(elev)
        cam_pos = np.array([base * np.cos(azim), base * np.sin(azim), z])
        forward = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        c2w = np.eye(4)
        c2w[:3, :4] = np.concatenate((np.stack((right, -up, forward), axis=1), cam_pos[:, None]), axis=1)
        c2ws.append(c2w)

    return w, h, num_views, fxfycxcy, np.stack(c2ws, axis=0)


# =============================================================================
# Gaussian Model
# =============================================================================

class GaussianModel:
    """3D Gaussian splatting model."""

    def __init__(self, sh_degree: int, scaling_modifier=None):
        self.sh_degree = sh_degree
        self.scaling_modifier = scaling_modifier
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0) if sh_degree > 0 else None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

    def to(self, device):
        """Move model to device."""
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0 and self._features_rest is not None:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.sh_degree > 0 and self._features_rest is not None:
            return torch.cat((self._features_dc, self._features_rest), dim=1)
        return self._features_dc

    @property
    def get_scaling(self):
        scaling = torch.exp(self._scaling)
        if self.scaling_modifier is not None:
            scaling = scaling * self.scaling_modifier
        return scaling

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def load_ply(self, path):
        """Load model from PLY file."""
        plydata = PlyData.read(path)
        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = sorted(
                [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")],
                key=lambda x: int(x.split("_")[-1])
            )
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.sh_degree + 1) ** 2 - 1))

        scale_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")],
            key=lambda x: int(x.split("_")[-1])
        )
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = sorted(
            [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")],
            key=lambda x: int(x.split("_")[-1])
        )
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = torch.from_numpy(xyz.astype(np.float32))
        self._features_dc = torch.from_numpy(features_dc.astype(np.float32)).transpose(1, 2).contiguous()
        if self.sh_degree > 0:
            self._features_rest = torch.from_numpy(features_extra.astype(np.float32)).transpose(1, 2).contiguous()
        self._opacity = torch.from_numpy(np.copy(opacities).astype(np.float32)).contiguous()
        self._scaling = torch.from_numpy(scales.astype(np.float32)).contiguous()
        self._rotation = torch.from_numpy(rots.astype(np.float32)).contiguous()


# =============================================================================
# Rendering
# =============================================================================

def render_opencv_cam(pc: GaussianModel, height: int, width: int, C2W: torch.Tensor, fxfycxcy: torch.Tensor, bg_color=(1.0, 1.0, 1.0)):
    """Render Gaussians from an OpenCV camera."""
    screenspace_points = torch.empty_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device=pc.get_xyz.device)
    viewpoint_camera = Camera(C2W=C2W, fxfycxcy=fxfycxcy, h=height, w=width)
    bg = torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)

    try:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.h),
            image_width=int(viewpoint_camera.w),
            tanfovx=viewpoint_camera.tanfovX,
            tanfovy=viewpoint_camera.tanfovY,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            projmatrix_raw=viewpoint_camera.projection_matrix,
            sh_degree=pc.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
    except TypeError:
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.h),
            image_width=int(viewpoint_camera.w),
            tanfovx=viewpoint_camera.tanfovX,
            tanfovy=viewpoint_camera.tanfovY,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, *_ = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=pc.get_features,
        colors_precomp=None,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
    )
    return rendered_image


@torch.no_grad()
def render_proxy_video(pc: GaussianModel, rendering_resolution=480, num_views=81, bg_color=(1.0, 1.0, 1.0), camera_params=None):
    """Render proxy video from Gaussian model."""
    if camera_params is None:
        max_azimuth, max_elevation, max_fov = 45, 30, 15
        start_azimuth = np.random.uniform(-max_azimuth, max_azimuth)
        end_azimuth = np.random.uniform(-max_azimuth, max_azimuth)
        start_elevation = np.random.uniform(-max_elevation, max_elevation)
        end_elevation = np.random.uniform(-max_elevation, max_elevation)
        start_fov = np.random.uniform(max_fov, max_fov)
        end_fov = np.random.uniform(max_fov, max_fov)
    else:
        start_azimuth = camera_params["start_azimuth"]
        end_azimuth = camera_params["end_azimuth"]
        start_elevation = camera_params["start_elevation"]
        end_elevation = camera_params["end_elevation"]
        start_fov = camera_params["start_fov"]
        end_fov = camera_params["end_fov"]

    _, _, _, fxfycxcy, c2ws = get_proxy_video_cameras(
        num_views=num_views, w=rendering_resolution, h=rendering_resolution,
        start_azimuth=start_azimuth, end_azimuth=end_azimuth,
        start_elevation=start_elevation, end_elevation=end_elevation,
        start_fov=start_fov, end_fov=end_fov,
    )

    device = pc._xyz.device
    fxfycxcy = torch.from_numpy(fxfycxcy).float().to(device)
    c2ws = torch.from_numpy(c2ws).float().to(device)

    renderings = torch.zeros(num_views, 3, rendering_resolution, rendering_resolution, dtype=torch.float32, device=device)
    for j in range(num_views):
        renderings[j] = render_opencv_cam(pc, rendering_resolution, rendering_resolution, c2ws[j], fxfycxcy[j], bg_color=bg_color)

    renderings = renderings.detach().cpu().numpy()
    renderings = (renderings * 255).clip(0, 255).astype(np.uint8)
    return rearrange(renderings, "v c h w -> v h w c")


def get_proxy_video(camera_params=None, num_frames=81, render_res=512, ply_path="./ckpts/gaussians.ply"):
    """Generate proxy video from a local PLY file."""
    device = torch.device("cuda")
    gaussian_obj = GaussianModel(sh_degree=0)
    gaussian_obj.load_ply(ply_path)
    gaussian_obj = gaussian_obj.to(device)
    return render_proxy_video(gaussian_obj, rendering_resolution=render_res, num_views=num_frames, camera_params=camera_params)
