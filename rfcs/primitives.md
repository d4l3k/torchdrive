# RFC: Primitives

Many operations in a 3D world model can be broken down into certain classes of
intermediate primitives. This is proposing encoding these as high level python
classes with a series of operations to convert between them.

## Representations

### Dense

* Grid3D: a 3D grid representation with a translation (x,y,z) and a certain number of channels at each position.
* GridBEV: a 2D grid representation with a translation (x,y) and certain number of channels at each position.
* GridImage: a 2D grid representing image features (i, j), channels and corresponding Cameras, time

Is it worth supporting separate logic for BEV rather than (x,y,1) 3D grids?

### Sparse

* BoundingBoxes3D: (x,y,z) coordinates + velocities with class probabilities
* BoundingBoxesBEV: (x, y) coordinates + velocities with class probabilities
* BoundingBoxesImage: (i,j) coordinates with class probabilities
* Lidar (x, y, z) coordinates

Optional:

* CrossAttention2d: (x,y) feature encodings as "tokens"
* CrossAttention3d: (x,y,z) feature encodings as "tokens"

### Metadata

* For representations with velocities, we need to include time data. May want to
  include time data for all objects to inherently tie them to spatial temporal
  models.

## Models / Techniques

### Images

* ImageEncoder: `GridImage(3) -> Tuple[GridImage(n), ...]`
  Models that encodes image data into a series of features at different
  resolutions
* FPN: `List[GridImage(n), ...] -> List[GridImage(n), ...]`

### 3D

* SimpleBEVProject: `List[GridImage(n), ...] -> GridBEV(n)`
* SimpleBEV3DProject: `List[GridImage(n), ...] -> Grid3D(n)`

* BEVFormer: `List[GridImage(n), ...] -> GridBEV(n)`

TODO -- can we be more flexible about these representations? For sparse
transformer architectures we can potentially have the individual model
TransformerDecoders query back to the original GridImage without rasterizing a
dense 3D representation


### Lidar

* PointPillars: `Lidar -> GridBEV/Grid3D`

### Sparse

* 3D DETR: Transformer decoder `CrossAttention3d -> BoundingBoxes3D`
* `Grid3D -> CrossAttention3d`: encodes a dense grid as a set of tokens using sin/cos encoding
* Path: Transformer decoder `CrossAttention3d -> coordinates?`

## Transforms

Standard transforms

* ChannelConcat should apply to all dense models
* TokenConcat should allow merging multiple attention token sets.

## Rendering

Rendering is somewhat of an inverse operation to the image encoders. This would
leverage PyTorch3d raymarching/rendering operations in most cases.

Gaussian splatting potentially for direct 3d sparse representations?

## Losses

* Voxel: consumes `Grid3D` and images/segmentation
* Path: consumes `CrossAttention2d` or `Grid3d`
* Det: consumes `CrossAttention2d` or `Grid3d` and bbox detections

