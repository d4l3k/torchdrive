import os.path
import tempfile
import unittest

import torch

from torchdrive.datasets.autolabeler import (
    AutoLabeler,
    LabelType,
    load_tensors,
    save_tensors,
)

from torchdrive.datasets.dummy import DummyDataset


class TestAutolabeler(unittest.TestCase):
    def test_save_load(self) -> None:
        with tempfile.TemporaryDirectory() as path:
            want = {"foo": torch.rand(1)}
            path = os.path.join(path, "foo.safetensors.zstd")
            save_tensors(path, want)
            out = load_tensors(path)

            self.assertEqual(out, want)

    def test_props(self) -> None:
        dataset = DummyDataset()
        labeler = AutoLabeler(dataset, "/nonexistant")
        self.assertEqual(labeler.NAME, dataset.NAME)
        self.assertEqual(labeler.cameras, dataset.cameras)
        self.assertEqual(labeler.CAMERA_OVERLAP, dataset.CAMERA_OVERLAP)

    def test_autolabeler(self) -> None:
        dataset = DummyDataset()
        with tempfile.TemporaryDirectory() as path:
            labeler = AutoLabeler(dataset, path)
            self.assertEqual(len(labeler), len(dataset))

            sem_seg_path = os.path.join(
                path,
                dataset.NAME,
                LabelType.SEM_SEG,
            )
            os.makedirs(sem_seg_path)
            det_path = os.path.join(
                path,
                dataset.NAME,
                LabelType.DET,
            )
            os.makedirs(det_path)

            batch = dataset[0]
            for token in batch.token[0]:
                tensor_path = os.path.join(
                    sem_seg_path,
                    f"{token}.safetensors.zstd",
                )
                save_tensors(
                    tensor_path,
                    {
                        "left": torch.tensor(1),
                        "right": torch.tensor(1),
                    },
                )

                tensor_path = os.path.join(
                    det_path,
                    f"{token}.safetensors.zstd",
                )
                out = {}
                for cam in dataset.cameras:
                    for i in range(10):
                        out[f"{cam}/{i}"] = torch.rand(2, 5)

                save_tensors(tensor_path, out)

            out = labeler[0]
            # pyre-fixme[16]: Optional type has no attribute `sem_seg`.
            self.assertIsNotNone(out.sem_seg)
            self.assertEqual(out.sem_seg["left"].shape, (3,))
            self.assertEqual(out.sem_seg["right"].shape, (3,))

            # pyre-fixme[16]: Optional type has no attribute `det`.
            self.assertIsNotNone(out.det)
