import unittest

import torch

from torchdrive.models.det import BDD100KDet, DetBEVDecoder, DetBEVTransformerDecoder

try:
    # pyre-fixme[21]: Could not find module `mmcv`.
    import mmcv
except ModuleNotFoundError:
    mmcv = False


class TestDet(unittest.TestCase):
    # pyre-fixme[56]: Pyre was not able to infer the type of argument `not mmcv` to
    #  decorator factory `unittest.skipIf`.
    @unittest.skipIf(not mmcv, "must have mmcv")
    def test_bdd100kdet(self) -> None:
        device = torch.device("cpu")
        m = BDD100KDet(device)
        out = m(torch.rand(1, 3, 48, 64, device=device))
        self.assertIsNotNone(out)

    def test_bdd100kdet_weights(self) -> None:
        self.assertEqual(len(BDD100KDet.WEIGHTS), len(BDD100KDet.LABELS))

    def test_det_bev_decoder(self) -> None:
        m = DetBEVDecoder(
            bev_shape=(4, 4),
            dim=16,
            num_queries=10,
            num_heads=2,
        )
        classes, bboxes = m(torch.rand(2, 16, 4, 4))
        self.assertEqual(classes.shape, (2, 10, 11))
        self.assertEqual(bboxes.shape, (2, 10, 9))

    def test_det_bev_transformer_decoder(self) -> None:
        m = DetBEVTransformerDecoder(
            bev_shape=(4, 5),
            dim=16,
            num_queries=10,
            num_heads=2,
            dim_feedforward=32,
        )
        classes, bboxes = m(torch.rand(2, 16, 4, 5))
        self.assertEqual(classes.shape, (2, 10, 11))
        self.assertEqual(bboxes.shape, (2, 10, 9))

        self.assertIsNotNone(m.decoder_params())
