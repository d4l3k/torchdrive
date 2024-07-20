from torchdrive.train_config import Datasets, DiffTrajTrainConfig


CONFIG = DiffTrajTrainConfig(
    # backbone settings
    cameras=[
        "main",
    ],
    num_frames=1,
    num_encode_frames=1,
    cam_shape=(480, 640),
    # optimizer settings
    epochs=200,
    lr=1e-4,
    grad_clip=1.0,
    step_size=1000,
    # dataset
    dataset=Datasets.RICE,
    dataset_path="/mnt/ext/openape/snapshots/out-2024/index.txt",
    autolabel_path=None,
    mask_path="/mnt/ext/openape/masks/",
    num_workers=16,
    batch_size=64,
    autolabel=False,
)
