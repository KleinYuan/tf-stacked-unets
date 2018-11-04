# Summary

Tensorflow implementation of Stacked Unets [ICLR 2019 Reproducibility Challenge]: https://openreview.net/forum?id=BJgFcj0qKX


# Training Details

* Pre-train SUNet-7-128 with MS-COCO-2014 (80 classes)

* TBA


# Train

```
# Git clone this repo
# Naivgate to root dir of this project
python apps/train.py configs/stacked_unet_config.yaml
```

You can come up with your own configurations for diverse and flexible experiments by replacing `configs/stacked_unet_config.yaml`.

