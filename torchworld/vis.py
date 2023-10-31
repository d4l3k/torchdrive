import pythreejs

from torchworld.structures.cameras import CamerasBase


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
    geo = pythreejs.AxesHelper(1)
    geo.matrixAutoUpdate = False
    geo.matrix = tuple(view_to_world.contiguous().view(-1).tolist())

    scene.add([geo])
    return geo
