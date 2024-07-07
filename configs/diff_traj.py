from torchdrive.train_config import Datasets, DiffTrajTrainConfig


CONFIG = DiffTrajTrainConfig(
    # backbone settings
    cameras=[
        "CAM_FRONT",
    ],
    num_frames=1,
    num_encode_frames=1,
    cam_shape=(480, 640),
    # optimizer settings
    epochs=20,
    lr=1e-4,
    grad_clip=1.0,
    step_size=1000,
    # dataset
    dataset=Datasets.NUSCENES,
    dataset_path="/home/tristanr/nuscenes",
    autolabel_path=None,  # "/mnt/ext3/autolabel2",
    mask_path="n/a",  # only used for rice dataset
    num_workers=4,
    batch_size=8,
    autolabel=False,
)
