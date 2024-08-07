from typing import Tuple, Union

import torch
from pytorch3d.ops.utils import eyes
from pytorch3d.renderer.implicit.raysampling import HeterogeneousRayBundle, RayBundle
from pytorch3d.renderer.implicit.utils import (
    _validate_ray_bundle_variables,
    ray_bundle_variables_to_ray_points,
)

from torchworld.structures.grid import Grid3d
from torchworld.transforms.transform3d import Transform3d


class GridSampler(torch.nn.Module):
    """
    A module to sample from a Grid3d at 3D points sampled along projection rays.

    This is adapted from the stock pytorch3d VolumeSampler for Grid3d with an
    additional padding_mode setting.
    """

    def __init__(
        self,
        densities: Grid3d,
        features: Grid3d,
        sample_mode: str = "bilinear",
        padding_mode: str = "zeros",
    ) -> None:
        """
        Args:
            grid: An instance of the `Grid3d` class representing a
                batch of volumes that are being rendered.
            sample_mode: Defines the algorithm used to sample the volumetric
                voxel grid. Can be either "bilinear" or "nearest".
            padding_mode: how to do padding
        """
        super().__init__()
        if not isinstance(densities, Grid3d):
            raise ValueError(
                "'densities' have to be an instance of the 'Grid3d' class."
            )
        if not isinstance(features, Grid3d):
            raise ValueError("'features' have to be an instance of the 'Grid3d' class.")
        self._densities = densities
        self._features = features
        self._sample_mode = sample_mode
        self._padding_mode = padding_mode

    # pyre-fixme[3]: Return type must be annotated.
    def _get_ray_directions_transform(self):
        """
        Compose the ray-directions transform by removing the translation component
        from the volume global-to-local coords transform.
        """
        world2local = self._features.local_to_world.inverse().get_matrix()
        # world2local = self._volumes.get_world_to_local_coords_transform().get_matrix()
        directions_transform_matrix = eyes(
            4,
            N=world2local.shape[0],
            device=world2local.device,
            dtype=world2local.dtype,
        )
        directions_transform_matrix[:, :3, :3] = world2local[:, :3, :3]
        directions_transform = Transform3d(matrix=directions_transform_matrix)
        return directions_transform

    def forward(
        self,
        ray_bundle: Union[RayBundle, HeterogeneousRayBundle],
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given an input ray parametrization, the forward function samples
        `self._volumes` at the respective 3D ray-points.
        Can also accept ImplicitronRayBundle as argument for ray_bundle.

        Args:
            ray_bundle: A RayBundle or HeterogeneousRayBundle object with the following fields:
                rays_origins_world: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                rays_directions_world: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                rays_lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.

        Returns:
            rays_densities: A tensor of shape
                `(minibatch, ..., num_points_per_ray, opacity_dim)` containing the
                density vectors sampled from the volume at the locations of
                the ray points.
            rays_features: A tensor of shape
                `(minibatch, ..., num_points_per_ray, feature_dim)` containing the
                feature vectors sampled from the volume at the locations of
                the ray points.
        """

        # take out the interesting parts of ray_bundle
        rays_origins_world = ray_bundle.origins
        rays_directions_world = ray_bundle.directions
        rays_lengths = ray_bundle.lengths

        # validate the inputs
        _validate_ray_bundle_variables(
            rays_origins_world, rays_directions_world, rays_lengths
        )
        if self._densities.shape[0] != rays_origins_world.shape[0]:
            raise ValueError("Input volumes have to have the same batch size as rays.")

        #########################################################
        # 1) convert the origins/directions to the local coords #
        #########################################################

        # origins are mapped with the world_to_local transform of the volumes
        world_to_local = self._features.local_to_world.inverse()
        pts_shape = rays_origins_world.shape
        rays_origins_local = world_to_local.transform_points(
            rays_origins_world.view(pts_shape[0], -1, 3),
        ).view(pts_shape)
        # rays_origins_local = self._volumes.world_to_local_coords(rays_origins_world)

        # obtain the Transform3d object that transforms ray directions to local coords
        directions_transform = self._get_ray_directions_transform()

        # transform the directions to the local coords
        rays_directions_local = directions_transform.transform_points(
            rays_directions_world.view(rays_lengths.shape[0], -1, 3)
        ).view(rays_directions_world.shape)

        ############################
        # 2) obtain the ray points #
        ############################

        # this op produces a fairly big tensor (minibatch, ..., n_samples_per_ray, 3)
        rays_points_local = ray_bundle_variables_to_ray_points(
            rays_origins_local, rays_directions_local, rays_lengths
        )

        ########################
        # 3) sample the volume #
        ########################

        # generate the tensor for sampling
        volumes_densities = self._densities
        dim_density = volumes_densities.shape[1]
        volumes_features = self._features

        # reshape to a size which grid_sample likes
        rays_points_local_flat = rays_points_local.view(
            rays_points_local.shape[0], -1, 1, 1, 3
        )

        # run the grid sampler on the volumes densities
        rays_densities = torch.nn.functional.grid_sample(
            volumes_densities,
            rays_points_local_flat.to(volumes_densities.dtype),
            align_corners=True,
            mode=self._sample_mode,
            padding_mode=self._padding_mode,
        )

        # permute the dimensions & reshape densities after sampling
        rays_densities = rays_densities.permute(0, 2, 3, 4, 1).view(
            *rays_points_local.shape[:-1], volumes_densities.shape[1]
        )

        # if features exist, run grid sampler again on the features densities
        if volumes_features is None:
            dim_feature = 0
            _, rays_features = rays_densities.split([dim_density, dim_feature], dim=-1)
        else:
            rays_features = torch.nn.functional.grid_sample(
                volumes_features,
                rays_points_local_flat.to(volumes_features.dtype),
                align_corners=True,
                mode=self._sample_mode,
                padding_mode=self._padding_mode,
            )

            # permute the dimensions & reshape features after sampling
            rays_features = rays_features.permute(0, 2, 3, 4, 1).view(
                *rays_points_local.shape[:-1], volumes_features.shape[1]
            )

        return rays_densities, rays_features
