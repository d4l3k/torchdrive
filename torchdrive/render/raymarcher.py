from typing import Optional, Tuple

import torch
from pytorch3d.renderer import PerspectiveCameras

from pytorch3d.renderer.implicit.utils import RayBundle
from pytorch3d.transforms import RotateAxisAngle, Transform3d


class DepthEmissionRaymarcher(torch.nn.Module):
    """
    This is a Pytorch3D raymarcher that renders out both depth and emission.
    This is useful for doing a joint depth render as well as semantic class
    label rendering for semantic purposes.
    """

    def __init__(
        self,
        background: Optional[torch.Tensor] = None,
        floor: Optional[float] = 0,
        wall: bool = True,
    ) -> None:
        """
        Args:
            background: Optional background "color" for the furthest sample in a
                ray.
            floor: Optional floor where all points below that will be marked as
                solid.
            wall: whether to make the furthest point density always 1 to avoid
                failing to sum to 1 and having a too close distance. Optional if
                the VolumeSampler padding method is not zeros.
        """
        super().__init__()

        self.floor = floor
        self.background = background
        self.wall = wall

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        ray_bundle: "RayBundle",
        eps: float = 1e-10,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)` whose values range in [0, 1].
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            ray_bundle: the bundle of rays to use
            eps: A lower bound added to `rays_densities` before computing
                the absorption function (cumprod of `1-rays_densities` along
                each ray). This prevents the cumprod to yield exact 0
                which would inhibit any gradient-based learning.
        Returns:
            features_opacities: A tensor of shape `(..., feature_dim+1)`
                that concatenates two tensors along the last dimension:
                    1) features: A tensor of per-ray renders
                        of shape `(..., feature_dim)`.
                    2) opacities: A tensor of per-ray opacity values
                        of shape `(..., 1)`. Its values range between [0, 1] and
                        denote the total amount of light that has been absorbed
                        for each ray. E.g. a value of 0 corresponds to the ray
                        completely passing through a volume. Please refer to the
                        `AbsorptionOnlyRaymarcher` documentation for the
                        explanation of the algorithm that computes `opacities`.
        """
        device = rays_densities.device

        if self.wall:
            rays_densities = rays_densities.clone()
            # clamp furthest point to prob 1
            rays_densities[..., -1, 0] = 1

        feat_dim = rays_features.size(-1)
        # set last point to background color
        if (background := self.background) is not None and feat_dim > 0:
            rays_features = rays_features.clone()
            rays_features[..., -1, :] = background

        if self.floor:
            # set floor depths
            # depth = (z-z0)/vz
            floor_depth = (
                self.floor - ray_bundle.origins[..., 2]
            ) / ray_bundle.directions[..., 2]
            floor_depth[floor_depth <= 0] = 10000
            floor_depth = floor_depth.unsqueeze(3).unsqueeze(4)
            is_floor = ray_bundle.lengths.unsqueeze(4) > floor_depth
            rays_densities[is_floor] = 1
            if (background := self.background) is not None and feat_dim > 0:
                rays_features[is_floor.squeeze(-1), :] = background

        ray_shape = rays_densities.shape[:-2]
        probs = rays_densities[..., 0].cumsum_(dim=-1)
        probs = probs.clamp_(max=1)
        probs = probs.diff(dim=-1, prepend=torch.zeros((*ray_shape, 1), device=device))

        depth = (probs * ray_bundle.lengths).sum(dim=-1)
        features = (probs.unsqueeze(-1) * rays_features).sum(dim=-2)

        return depth, features


class CustomPerspectiveCameras(PerspectiveCameras):
    def __init__(
        self,
        T: torch.Tensor,
        K: torch.Tensor,
        image_size: torch.Tensor,
        **kwargs: object,
    ) -> None:
        """
        dim is (BS, h, w)
        K (in normalized K matrix by size)
        """
        BS = len(T)

        assert BS == len(image_size), f"{BS} {image_size.shape}"
        assert BS == len(K), f"{BS} {K.shape}"

        K = K.clone()
        K[:, 0] *= image_size[0, 1]
        K[:, 1] *= image_size[0, 0]
        K[:, 2, 2] = 0
        K[:, 3, 3] = 0
        K[:, 2, 3] = 1
        K[:, 3, 2] = 1
        super().__init__(
            R=torch.eye(3).expand(BS, 3, 3),
            K=K,
            image_size=image_size,
            in_ndc=False,
            # pyre-fixme[6]: got object
            **kwargs,
        )
        self.T: torch.Tensor = T.pinverse()

    def get_world_to_view_transform(self, **kwargs: object) -> Transform3d:
        r = RotateAxisAngle(180, axis="Z", device=self.T.device)
        return Transform3d(matrix=self.T.transpose(1, 2), device=self.T.device).compose(
            r
        )
