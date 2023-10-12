from torchdrive.data import Batch
from torchdrive.datasets.dataset import Dataset


class Autolabeler:
    """
    Autolabeler takes in a dataset and a cache location and automatically loads
    the autolabeled data based off of the batch tokens.
    """

    def __init__(self, dataset: Dataset, path: str) -> None:
        self.dataset = dataset
        self.path = path

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Batch:
        batch = self.dataset[idx]
        tokens = batch.token
