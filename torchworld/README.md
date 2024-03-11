# torchworld

This repo contains high level tensors for training 3d world representations.

The core idea is to include rich metadata about a tensor as part of it. This
uses PyTorch Tensor subclasses which allow for applying standard PyTorch models
to the world aware tensors.

See rfcs/primitives.md for more details.
