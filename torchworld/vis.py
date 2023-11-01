import numpy as np
import pythreejs

from torchworld.structures.cameras import CamerasBase
from torchworld.structures.grid import GridImage
from torchworld.transforms.img import normalize_img_cuda


def add_camera(scene: pythreejs.Scene, camera: CamerasBase) -> pythreejs.Object3D:
    """
    add_camera creates a geometry object for the provided camera and adds it to
    the specified scene.

    Arguments
    ---------
    scene:
        scene to use
    camera:
        camera to visualize
    """
    view_to_world = camera.get_world_to_view_transform().inverse().get_matrix()
    assert view_to_world.shape, (1, 4, 4)
    geo = pythreejs.AxesHelper(1)
    geo.matrixAutoUpdate = False
    geo.matrix = tuple(view_to_world.contiguous().view(-1).tolist())

    scene.add([geo])
    return geo


def add_grid_image(scene: pythreejs.Scene, image: GridImage) -> pythreejs.Object3D:
    cam = add_camera(scene, image.camera)
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

    # cam.add([mesh])
    scene.add([mesh])
    return mesh
