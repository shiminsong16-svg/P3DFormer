# P3DFormer: Review-Stage Source Release

This repository provides a review-stage source snapshot for 3D instance
segmentation. It is intended for inspection of the core model and evaluation
path; it is not a complete reproducibility release.

## Release Scope

Included in this snapshot:

- Core model and inference code
- ScanNetV2 data preparation and instance evaluation
- Required attention-RPE and PointGroup CUDA operators

Not included in this snapshot:

- Training entry point, optimization pipeline, and training losses
- Pretrained model checkpoints
- Internal experiment-management utilities

The complete training pipeline and pretrained models are planned for release
upon publication.

## Environment

The code targets Linux with an NVIDIA GPU, a compatible CUDA toolkit containing
`nvcc`, and Python 3.9 or later. Install PyTorch, `torch-scatter`, and `spconv`
versions compatible with the local CUDA toolkit first. Then, from the repository
root, run:

```bash
pip install -r requirements.txt
pip install -v -e lib/attention_rpe_ops
pip install -v -e p3dformer/lib
pip install -e .
```

The Python sources have been checked for syntax consistency. A clean
end-to-end Linux/CUDA installation has not yet been validated for this
review-stage snapshot.

## ScanNetV2 Preparation

Request and download ScanNetV2 from its official website. Install the ScanNet
mesh segmentator from <https://github.com/Karbo123/segmentator>, place the
official scans under `data/scannetv2/scans/`, and then run:

```bash
cd data/scannetv2
python split_data.py
python prepare_data_inst_with_normal.py --data_split val
cd ../..
```

The validation files should appear as:

```text
data/scannetv2/val/scene*_inst_nostuff.pth
```

ScanNet data are governed by the ScanNet terms of use and are not distributed
with this repository.

## Evaluation

Evaluation requires a compatible checkpoint, which is not distributed with
this review-stage snapshot.

```bash
python tools/test.py \
  configs/scannet/p3dformer_scannet_eval.yaml \
  /path/to/p3dformer_scannet.pth
```

Use `--out results/scannetv2` to export predictions in ScanNet benchmark format.

## Availability and Use

This repository is an inspection-oriented review-stage source release. No
open-source license is granted for original project code in this snapshot.
Third-party components remain subject to their respective licenses.

## Acknowledgements

This project builds upon MAFT, Mask3D, SSTNet, Relation3D, and EASE. We thank
the authors for making their work publicly available.
