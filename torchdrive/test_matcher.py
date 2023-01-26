import unittest

import torch

from torchdrive.matcher import HungarianMatcher


def rand_boxes(*shape: int) -> torch.Tensor:
    a = torch.rand(*shape, 2) * 0.5
    b = torch.rand(*shape, 2) * 0.5 + 0.5
    return torch.cat((a, b), dim=-1)


class TestMatcher(unittest.TestCase):
    def test_matcher(self) -> None:
        m = HungarianMatcher()
        BS = 2
        NUM_QUERIES = 5
        NUM_CLASSES = 4
        NUM_TARGET_BOXES = 3
        matches = m(
            outputs={
                "pred_logits": torch.rand(BS, NUM_QUERIES, NUM_CLASSES),
                "pred_boxes": rand_boxes(BS, NUM_QUERIES),
            },
            targets=[
                {
                    "labels": torch.zeros(NUM_TARGET_BOXES, dtype=torch.long),
                    "boxes": rand_boxes(NUM_TARGET_BOXES),
                }
            ]
            * BS,
            invalid_mask=torch.zeros(BS, NUM_QUERIES, dtype=torch.bool),
        )
