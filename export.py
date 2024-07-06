"""
This file is used to export torchdrive models.
"""

import argparse
import os
import os.path
from typing import Dict

import numpy as np
import onnx
import torch
import torch_tensorrt
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchdrive.data import Batch, TransferCollator
from torchdrive.train_config import create_parser, TrainConfig

from torchdrive.transforms.batch import NormalizeCarPosition
from tqdm import tqdm

# pyre-fixme[5]: Global expression must be annotated.
parser = create_parser()
parser.add_argument("--num_workers", type=int, required=True)
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--device", type=str, default="cpu")
args: argparse.Namespace = parser.parse_args()


config: TrainConfig = args.config

device = torch.device(args.device)
print(f"using device {device}")


print(f"output dir {args.output}")
os.makedirs(args.output, exist_ok=True)

model = config.create_model(device=device)


load_path: str = args.load
ckpt: Dict[str, torch.Tensor] = torch.load(
    load_path, map_location=device, weights_only=True
)
model.load_state_dict(ckpt["model"], strict=True)

dataset = config.create_dataset(smoke=True)
dataloader = DataLoader[Batch](
    dataset,
    batch_size=None,
    num_workers=args.num_workers,
    shuffle=True,
)
collator = TransferCollator(dataloader, batch_size=args.batch_size, device=device)

batch = next(iter(collator))

transform = NormalizeCarPosition(start_frame=model.num_encode_frames - 1)
batch = transform(batch)


def export_calibration() -> None:
    print("preparing calibration data")
    for cam, data in batch.color.items():
        # only grab first frame from each batch since frames are fairly similar
        # within a batch
        inp = batch.color[cam][:, 0].float()
        calib_data_path = os.path.join(args.output, f"{cam}.npy")
        with open(calib_data_path, "wb") as f:
            np.save(f, inp.numpy())


def export_cam_encoders() -> None:
    for cam, cam_model in model.camera_encoders.items():
        print(f"processing {cam}...")
        onnx_path = os.path.join(args.output, f"{cam}.onnx")

        data = torch.rand(1, 3, 480, 640)

        print(cam_model)

        cam_model.eval()

        # attach a global qconfig, which contains information about what kind
        # of observers to attach. Use 'x86' for server inference and 'qnnpack'
        # for mobile inference. Other quantization configurations such as selecting
        # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
        # can be specified here.
        # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
        # for server inference.
        # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        cam_model.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")

        # fuse the activations to preceding layers, where applicable
        # this needs to be done manually depending on the model architecture
        # model_fp32_fused = torch.ao.quantization.fuse_modules(cam_model,
        #    [['conv', 'bn', 'relu']])
        cam_model.fuse()
        model_fp32_fused = cam_model
        print(cam_model)

        # Prepare the model for QAT. This inserts observers and fake_quants in
        # the model needs to be set to train for QAT logic to work
        # the model that will observe weight and activation tensors during calibration.
        # model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())

        ## run the training loop (not shown)
        # for batch in tqdm(collator):
        #    inp = batch.color[cam].flatten(0, 1).float()
        #    model_fp32_prepared(inp)
        #    break

        ## Convert the observed model to a quantized model. This does several things:
        ## quantizes the weights, computes and stores the scale and bias value to be
        ## used with each activation tensor, fuses modules where appropriate,
        ## and replaces key operators with quantized implementations.
        # model_fp32_prepared.eval()
        # model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        torch.onnx.export(
            model_fp32_fused,
            data,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # variable length axes
                "output": {0: "batch_size"},
            },
        )

        # Check that the model is well formed
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

        # engines are version specific
        if False:
            scripted = torch.jit.trace(cam_model, example_inputs=[data])
            engine = torch_tensorrt.convert_method_to_trt_engine(
                scripted,
                inputs=[data],
                enabled_precisions={torch.half},
                ir="ts",
                truncate_long_and_double=True,
            )

            with open(os.path.join(args.output, f"{cam}.trt"), "wb") as f:
                f.write(engine)


@torch.no_grad()
def export_backbone() -> None:
    camera_features = {}
    for cam in model.cameras:
        feats = []
        for i in range(model.num_encode_frames):
            color = batch.color[cam][:, i].float()
            feats.append(model.camera_encoders[cam](color))
        camera_features[cam] = feats

    model.infer_batch = batch
    T = batch.cam_T
    out = model.infer_backbone(camera_features, T=T)


export_backbone()
