import numpy as np
import pythreejs
import torch
import torch.nn.functional as F
from matplotlib import cm

from torchworld.structures.cameras import CamerasBase
from torchworld.structures.grid import Grid3d, GridImage
from torchworld.transforms.img import normalize_img_cuda
from torchworld.transforms.transform3d import Transform3d


def camera(camera: CamerasBase) -> pythreejs.Object3D:
    """
    add_camera creates a geometry object for the provided camera.

    Arguments
    ---------
    camera:
        camera to visualize
    """
    view_to_world = camera.get_world_to_view_transform().inverse().get_matrix()
    assert view_to_world.shape, (1, 4, 4)
    geo = pythreejs.AxesHelper(1)
    geo.matrixAutoUpdate = False
    geo.matrix = tuple(view_to_world.contiguous().view(-1).tolist())

    return geo


def grid_image(image: GridImage) -> pythreejs.Object3D:
    """
    add_grid_image creates a geometry object for the provided RGB image and
    camera.

    Arguments
    ---------
    image: image to render
    """
    dist = 1.0
    plane = pythreejs.BufferGeometry(
        attributes={
            "position": pythreejs.BufferAttribute(
                np.array(
                    [
                        [
                            -1.0,
                            -1.0,
                            dist,
                        ],
                        [
                            1.0,
                            -1.0,
                            dist,
                        ],
                        [
                            1.0,
                            1.0,
                            dist,
                        ],
                        [
                            -1.0,
                            1.0,
                            dist,
                        ],
                    ],
                    dtype=np.float32,
                ),
                normalized=False,
            ),
            "index": pythreejs.BufferAttribute(
                np.array(
                    [
                        0,
                        1,
                        2,
                        2,
                        3,
                        0,
                    ],
                    dtype=np.uint32,
                ),
                normalized=False,
            ),
            "uv": pythreejs.BufferAttribute(
                np.array(
                    [
                        [
                            0,
                            0,
                        ],
                        [1, 0],
                        [1, 1],
                        [0, 1],
                    ],
                    dtype=np.float32,
                ),
                normalized=False,
            ),
        },
    )

    data = image.data[0]
    data = normalize_img_cuda(data)
    data = data.permute(1, 2, 0)

    texture = pythreejs.DataTexture(
        data=data.float().numpy(),
        format="RGBFormat",
        type="FloatType",
    )

    mat = pythreejs.MeshBasicMaterial(
        map=texture,
        side=pythreejs.Side.DoubleSide,  # pyre-ignore[16]: DoubleSide
    )
    mesh = pythreejs.Mesh(plane, mat)

    T = image.camera.world_to_ndc_transform()
    T = T.inverse()
    T = T.get_matrix()

    mesh.matrixAutoUpdate = False
    mesh.matrix = tuple(T.contiguous().view(-1).tolist())

    cam = camera(image.camera)

    group = pythreejs.Group()
    group.add([mesh, cam])

    return group


def _get_cmap(palette: str, num_colors: int = 1000) -> torch.Tensor:
    cmap = cm.get_cmap(palette)
    return torch.tensor(
        [cmap(i / num_colors)[:3] for i in range(num_colors)],
    )


def grid_3d_occupancy(
    grid: Grid3d,
    p: float = 0.5,
    palette: str = "magma",
    eps: float = 1e-8,
) -> pythreejs.Object3D:
    """
    Creates a geometry object for the provided voxel grid and camera. This must
    be an occupancy grid with a single channel and values between 0 and 1.

    Arguments
    ---------
    grid: occupancy grid to render
    p: probability threshold for voxels to render
    palette: the matplotlib color palette to use
    eps: a small value to avoid numerical issues
    """
    BS, ch, Z, Y, X = grid.data.shape
    N = X * Y * Z
    device = grid.data.device
    grid_shape = grid.grid_shape()

    if ch != 1:
        raise TypeError("must have single channel")

    colors = _get_cmap(palette)

    data = grid.data.permute(0, 1, 4, 3, 2)
    grid_shape = grid_shape[::-1]

    channels = torch.meshgrid(
        *(torch.arange(-1, 1 - eps, 2 / dim, device=device) for dim in grid_shape),
        indexing="ij",
    )
    grid_points = torch.stack(channels, dim=-1).unsqueeze(0)

    # filter all boxes
    matching = data[0] >= p

    # hide boxes that are completely surrounded by other boxes
    visible = (
        F.conv3d(
            matching.int(),
            weight=torch.ones(1, 1, 3, 3, 3, dtype=torch.int, device=matching.device),
            padding=1,
        )
        < 27
    )
    matching = torch.logical_and(matching, visible)

    offsets = grid_points[matching]
    values = data[0][matching]
    colors = torch.index_select(
        colors, dim=0, index=(values * (colors.size(0) - 1)).int()
    )
    # offsets = grid.local_to_world.transform_points(offsets)
    N = offsets.size(0)

    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    vertices /= grid_shape
    vertices *= 2
    faces = np.array(
        [
            [0, 4, 3],
            [3, 4, 7],
            [1, 2, 6],
            [1, 6, 5],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 7],
            [5, 6, 7],
        ],
        dtype=np.uint32,
    )

    colors = pythreejs.InstancedBufferAttribute(
        array=colors.numpy(), meshPerAttribute=3
    )
    offsets = pythreejs.InstancedBufferAttribute(
        array=offsets.numpy(), meshPerAttribute=3
    )

    instancedGeometry = pythreejs.InstancedBufferGeometry(
        maxInstancedCount=N * 3,
        attributes={
            "position": pythreejs.BufferAttribute(array=vertices),
            "index": pythreejs.BufferAttribute(array=faces.ravel()),
            "offset": offsets,
            "color": colors,
        },
    )

    material = pythreejs.ShaderMaterial(
        vertexShader="""
            precision highp float;
            attribute vec3 offset;
            attribute vec4 rotation;
            varying vec3 vPosition;
            varying vec4 vColor;
            void main(){

                vPosition = position + 2.0 * cross( rotation.xyz, cross( rotation.xyz, position ) + rotation.w * position );
                vPosition += offset;

                vColor = vec4( color, 1.0);
                gl_Position = projectionMatrix * modelViewMatrix * vec4( vPosition, 1.0 );
            }
            """,
        fragmentShader="""
            precision highp float;
            varying vec4 vColor;
            void main() {
                gl_FragColor = vec4( vColor );
            }
            """,
        vertexColors="VertexColors",
        transparent=False,
    )

    mesh = pythreejs.Mesh(instancedGeometry, material)

    edges = pythreejs.EdgesGeometry(pythreejs.BoxGeometry(width=2, height=2, depth=2))
    line_mat = pythreejs.LineBasicMaterial(color="#a40")
    line = pythreejs.LineSegments(edges, line_mat)

    group = pythreejs.Group()
    group.add(mesh)
    group.add(line)

    T = grid.local_to_world
    T = T.get_matrix()
    group.matrixAutoUpdate = False
    group.matrix = tuple(T.contiguous().view(-1).tolist())

    return group


def path(positions: Transform3d) -> pythreejs.Object3D:
    """path renders a series of positions given a set of world_to_local
    transformation matrices.

    Arguments
    ---------
    positions: Transform3d(BS)
        world_to_local transforms to plot
    """

    group = pythreejs.Group()

    matrix = positions.inverse().get_matrix()

    BS = len(positions)
    for i in range(BS):
        geo = pythreejs.AxesHelper(1)
        geo.matrixAutoUpdate = False
        geo.matrix = tuple(matrix[i].contiguous().view(-1).tolist())
        group.add(geo)
    return group
