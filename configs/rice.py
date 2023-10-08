from torchdrive.train_config import Datasets, TrainConfig


CONFIG = TrainConfig(
    # backbone settings
    cameras=[
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ],
    dim=256,
    cam_dim=96,
    hr_dim=384,
    backbone="rice",
    cam_encoder="regnet",
    num_encode_frames=3,
    cam_shape=(480, 640),
    num_upsamples=4,
    grid_shape=(256, 256, 16),
    # optimizer settings
    epochs=20,
    lr=1e-4,
    grad_clip=1.0,
    step_size=1000,
    # dataset
    dataset=Datasets.NUSCENES,
    dataset_path="/mnt/ext3/nuscenes",
    mask_path="n/a",  # only used for rice dataset
    num_workers=6,
    batch_size=2,
    # tasks
    det=False,
    ae=False,
    voxel=True,
    voxelsem=True,
    path=False,
)