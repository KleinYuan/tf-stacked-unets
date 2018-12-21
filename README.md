# Summary

Tensorflow implementation of Stacked Unets [ICLR 2019 Reproducibility Challenge]: https://openreview.net/forum?id=BJgFcj0qKX



# Train


- [X] SUNET-64 on ImageNet: First uncomment line41-43 in blocks.py then `python apps/sunet64_train.py configs/sunet_config_imagenet.yaml`

- [X] SUNET-64-u1 on CIFAR-10:  `python apps/sunet64_u1_train.py configs/sunet_config_cifar10.yaml`

- [X] SUNET-64-u2 on CIFAR-10: `python apps/sunet64_u2_train.py configs/sunet_config_cifar10.yaml`

- [X] SUNET-64-u1 on CIFAR-100:  `python apps/sunet64_u1_train.py configs/sunet_config_cifar100.yaml`

- [X] SUNET-64-u2 on CIFAR-100: `python apps/sunet64_u2_train.py configs/sunet_config_cifar100.yaml`

# Architecture

### SUNET-64

![diagram](https://user-images.githubusercontent.com/8921629/50336823-f958f000-04c3-11e9-9ed1-acd9b047edbf.png)

### SUNET-64-U1

![sunet-64-u1](https://user-images.githubusercontent.com/8921629/50336836-feb63a80-04c3-11e9-82b0-f70cfe5bf9ea.png)

### SUNET-64-U2

![sunet-64-u2](https://user-images.githubusercontent.com/8921629/50336831-fcec7700-04c3-11e9-97ce-b8820070013b.png)


# Results

