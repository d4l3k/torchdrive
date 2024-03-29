from torchdrive.data import Batch, dummy_item
from torchdrive.datasets.dataset import Dataset, Datasets


class DummyDataset(Dataset):
    # pyre-fixme[4]: Attribute must be annotated.
    NAME = Datasets.DUMMY
    cameras = ["left", "right"]
    # pyre-fixme[4]: Attribute must be annotated.
    CAMERA_OVERLAP = {}

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> Batch:
        if idx > len(self):
            raise IndexError("invalid index")
        return dummy_item()
