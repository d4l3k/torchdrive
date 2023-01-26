from typing import Optional, Tuple

import torch

from pytorch3d.renderer.implicit.utils import RayBundle


class DepthEmissionRaymarcher(torch.nn.Module):
    """
    This is a Pytorch3D raymarcher that renders out both depth and emission.
    This is useful for doing a joint depth render as well as semantic class
    label rendering for semantic purposes.
    """

    def __init__(self, background: Optional[torch.Tensor] = None) -> None:
        super().__init__()

        self.floor: float = 0
        self.background = background

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

        rays_densities = rays_densities.clone()

        # clamp furthest point to prob 1
        rays_densities[..., -1, 0] = 1
        # set last point to background color
        if (background := self.background) is not None:
            rays_features[..., -1, :] = background

        # set floor depths
        # depth = (z-z0)/vz
        floor_depth = (self.floor - ray_bundle.origins[..., 2]) / ray_bundle.directions[
            ..., 2
        ]
        torch.nan_to_num_(floor_depth, nan=-1.0)
        floor_depth[floor_depth <= 0] = 10000
        floor_depth = floor_depth.unsqueeze(3).unsqueeze(4)
        rays_densities[ray_bundle.lengths.unsqueeze(4) > floor_depth] = 1

        ray_shape = rays_densities.shape[:-2]
        probs = rays_densities[..., 0].cumsum_(dim=3)
        probs = probs.clamp_(max=1)
        probs = probs.diff(dim=3, prepend=torch.zeros((*ray_shape, 1), device=device))

        depth = (probs * ray_bundle.lengths).sum(dim=3)
        features = (probs.unsqueeze(-1) * rays_features).sum(dim=3)

        return depth, features
