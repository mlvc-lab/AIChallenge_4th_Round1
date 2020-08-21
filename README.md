# AI-Challenge Base Code
  
You can train or test ResNet/MobileNet/MobileNetV2 on CIFAR10/CIFAR100/ImageNet.  
Specially, you can train or test on any device (CPU/sinlge GPU/multi GPU) and different device environment available.

----------

## Requirements

- `python 3.5+`
- `pytorch 1.4+`
- `torchvision 0.4+`
- `tqdm` (for downloading pretrained chekpoints)
- `sacred` (for logging on omniboard)
- `pymongo` (for logging on omniboard)

----------

## Files

- `config.py`: set configuration
- `data.py`: data loading
- `down_ckpt.py`: download pretrained checkpoints
- `main.py`: main python file for training or testing
- `models`
  - `__init__.py`
  - `mobilenet.py`
  - `mobilenetv2.py`
  - `resnet.py`
  - ...
- `utils.py`
- ...

----------

## How to train or test networks

```text
usage: main.py [-h] [-a ARCH] [--layers N] [--width-mult WM] [--datapath PATH]
               [-j N] [--epochs N] [-b N] [--optimizer OPTIMIZER] [--lr LR]
               [--momentum M] [--wd WEIGHT_DECAY] [--nest] [--sched TYPE]
               [--step-size STEP] [--milestones EPOCH [EPOCH ...]]
               [--gamma GAMMA] [--print-freq N] [-C] [-g GPU [GPU ...]]
               [--run-type TYPE] [--load FILE.pth] [--save FILE.pth] [-P]
               [--pruner PRUNER] [--prune-type PRUNE_TYPE]
               [--prune-freq PRUNE_FREQ] [--prune-rate PRUNE_RATE] [-Q]
               [--quantizer QUANTIZER] [--quant-bitw N] [--quant-bita N]
               [--quant-cfg QUANT_CFG] [-D] [--dist-type DIST_TYPE]
               [--tch-arch TCH_ARCH] [--tch-layers N] [--tch-width-mult WM]
               [--tch-load FILE.pth]
               DATA

AI-Challenge Base Code

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet | things
                        (default: cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 | resnet |
                        wideresnet (default: resnet)
  --layers N            number of layers in ResNet (default: 56)
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  --datapath PATH       where you want to load/save your dataset? (default:
                        ../data)
  -j N, --workers N     number of data loading workers (default: 8)
  --epochs N            number of total epochs to run (default: 200)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel
  --optimizer OPTIMIZER
                        name of optimizer to train the model (default: SGD)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --momentum M          momentum (default: 0.9)
  --wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        weight decay (default: 5e-4)
  --nest, --nesterov    use nesterov momentum?
  --sched TYPE, --scheduler TYPE
                        scheduler: step | multistep | exp | cosine (default:
                        step)
  --step-size STEP      period of learning rate decay / maximum number of
                        iterations for cosine annealing scheduler (default:
                        30)
  --milestones EPOCH [EPOCH ...]
                        list of epoch indices for multi step scheduler (must
                        be increasing) (default: 30 80)
  --gamma GAMMA         multiplicative factor of learning rate decay (default:
                        0.1)
  --print-freq N        print frequency (default: 100)
  -C, --cuda            use cuda?
  -g GPU [GPU ...], --gpuids GPU [GPU ...]
                        GPU IDs for using (default: 0)
  --run-type TYPE       type of run the main function e.g. train or evaluate
                        (default: train)
  --load FILE.pth       name of checkpoint for testing model (default: None)
  --save FILE.pth       name of checkpoint for saving model (default: None)
  -P, --prune           Use pruning
  --pruner PRUNER       method of pruning to apply (default: dpf)
  --prune-type PRUNE_TYPE
                        specify 'unstructured' or 'structured'
  --prune-freq PRUNE_FREQ
                        update frequency
  --prune-rate PRUNE_RATE
                        pruning rate
  -Q, --quantize        If this is set, the model layers are changed to
                        quantized layers.
  --quantizer QUANTIZER
                        method of quantization to apply (default: lsq)
  --quant-bitw N        number of bits for weights (default: 8)
  --quant-bita N        number of bits for activations (default: 32)
  --quant-cfg QUANT_CFG
                        name of quantization configuration for each type of
                        layers (default: base)
  -D, --distill         If this is set, teacher model is loaded to distill the
                        training student.
  --dist-type DIST_TYPE
                        self distillation type in \{KD, AT, SP\} (default:
                        None)
  --tch-arch TCH_ARCH   choose pre-trained teacher
  --tch-layers N        number of layers in ResNet (default: 56)
  --tch-width-mult WM   width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  --tch-load FILE.pth   name of checkpoint for teacher model (default: None)
```

## Training

#### Example codes for training various models

1. train a ResNet-56 model

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --save best.pth --epochs 300 --batch-size 128 --lr 0.1 --scheduler multistep --milestones 100 200 --gamma 0.1 --optimizer SGD --momentum 0.9 --nesterov --wd 1e-4
```

2. train a WideResNet40-4 model with multi-GPU

```shell
python main.py cifar100 -a wideresnet --layers 40 --width-mult 4 -C -g 0 1 2 3
```

3. train a MobileNetV2 model

```shell
python main.py cifar100 -a mobilenetv2 --width-mult 1.0 -C -g 0
```

4. train a EfficientNet-B0 model

```shell
python main.py cifar100 -a efficientnet --model-mult 0 -C -g 0
```

5. train a ReXNet model

```shell
python main.py cifar100 -a rexnet --width-mult 1.0 --depth-mult 1.0 -C -g 0
```

## Test

Evaluate the trained model

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --run-type evaluate --load best.pth
```
----------

# Total compression processes

This example shows the whole training process to compress the model through pruning, quantization, and distillation.

## 1. train a teacher (larger than baseline model)

train the baseline of teacher model.

```shell
python main.py cifar100 -a wideresnet --layers 40 --width-mult 4 -C -g 0 --save base.pth --epochs 200 --batch-size 64 --lr 0.1 --scheduler multistep --milestones 100 150 --gamma 0.1
```

## 2. train baseline

train the baseline.

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --save base.pth --epochs 200 --batch-size 256 --lr 0.1 --scheduler multistep --milestones 100 150 --gamma 0.1
```

or train the baseline with distillation using the previous teacher

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --save base_distilled.pth -D --dist-type AT --tch-arch wideresnet --tch-layers 40 --tch-width-mult 4 --tch-load base.pth --epochs 200 --batch-size 256 --lr 0.1 --scheduler multistep --milestones 100 150 --gamma 0.1
```

## 3. prune the baseline

prune the previous baseline.

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --load base_distilled.pth --save prune.pth -P --prune-type unstructured --prune-freq 16 --prune-rate 0.9 --epochs 300 --batch-size 128  --lr 0.2 --wd 1e-4 --nesterov --scheduler multistep --milestones 150 225 --gamma 0.1
```

or prune the previous baseline with distillation using the previous teacher

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --load base_distilled.pth --save prune.pth -P --prune-type unstructured --prune-freq 16 --prune-rate 0.9 -D --dist-type KD --tch-arch wideresnet --tch-layers 40 --tch-width-mult 4 --tch-load base.pth --epochs 300 --batch-size 128  --lr 0.2 --wd 1e-4 --nesterov --scheduler multistep --milestones 150 225 --gamma 0.1
```

## 4. quantize the pruned model

quantize the previous pruned model.

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --load base_distilled.pth --save quant.pth -Q --quantizer lsq --quant-bitw 8 --quant-bita 32 --quant-cfg base --epochs 200 --batch-size 128  --lr 0.01 --wd 1e-4 --scheduler multistep --milestones 100 150 --gamma 0.1
```

or quantize the previous pruned model with distillation using the previous teacher.

```shell
python main.py cifar100 -a resnet --layers 56 -C -g 0 --load prune_distilled.pth --save quant_distilled.pth -Q --quantizer lsq --quant-bitw 8 --quant-bita 32 --quant-cfg mask -D --dist-type KD --tch-arch wideresnet --tch-layers 40 --tch-width-mult 4 --tch-load base.pth --epochs 200 --batch-size 128  --lr 0.01 --wd 1e-4 --scheduler multistep --milestones 100 150 --gamma 0.1
```

----------

# AI-Challenge Training

## 1. using things dataset

train a baseline model.

```shell
python main.py things --datapath /dataset/things_v1 -a resnet --layers 18 -C -g 0 --save base.pth --epochs 300 --batch-size 128 --lr 0.01 --scheduler multistep --milestones 100 200 --gamma 0.1 --wd 0.00005
```

----------

## How to download a pretrained Model

The pretrained models of ResNet56 trained on CIFAR100 is only available now...

```text
usage: down_ckpt.py [-h] [-a ARCH] [--layers N] [--width-mult WM] [-o OUT]
                    DATA

positional arguments:
  DATA                  dataset: cifar10 | cifar100 | imagenet (default:
                        cifar10)

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mobilenet | mobilenetv2 | resnet
                        (default: mobilenet)
  --layers N            number of layers in ResNet (default: 56)
  --width-mult WM       width multiplier to thin a network uniformly at each
                        layer (default: 1.0)
  -o OUT, --out OUT     output filename of pretrained model from our google
                        drive
```

### Usage

```shell
python down_ckpt.py cifar100 -a resnet --layers 56 -o pretrained_ckpt.pth
```

### ResNet56/CIFAR100 pretrained checkpoint experiment detail

- #epochs: 200
- batch size: 256
- initial learning rate: 0.1
- SGD
- momentum: 0.9
- weight decay: 5e-4
- no nesterov
- multi-step scheduler
  - gamma: 0.1
  - milestones: [100, 150]

----------

## References

- [torchvision models github codes](https://github.com/pytorch/vision/tree/master/torchvision/models)
- [MobileNet Cifar GitHub (unofficial)](https://github.com/kuangliu/pytorch-cifar)
- [MobileNetV2 Cifar GitHub (unofficial)](https://github.com/tinyalpha/mobileNet-v2_cifar10)
