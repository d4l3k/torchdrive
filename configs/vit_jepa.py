from torchdrive.train_config import Datasets, ViTJEPATrainConfig


CONFIG = ViTJEPATrainConfig(
    # backbone settings
    cameras=[
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ],
    num_frames=5,
    num_encode_frames=3,
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
    batch_size=10,
    autolabel=False,
)
