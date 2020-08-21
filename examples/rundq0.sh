#!/bin/bash
apt-get update
pip install opencv-python
pip install opencv-contrib-python
apt-get install -y libglib2.0-0
pip install adamp
pip install sklearn

cd /root/volume/AIChallenge_base
python main.py things --datapath /dataset/things_v4 -j 8 -a rexnet --width-mult 1.0 --depth-mult 1.0 -C -g 0 1 2 3 --save rexnet_1.0_v4_distill_quant0.pth --load rexnet_1.0.pth --transfer --src-dataset imagenet \
-Q --quantizer lsq --quant-bitw 8 --quant-bita 32 --quant-cfg base \
-D --dist-type KD --tch-arch rexnet --tch-width-mult 2.0 --tch-load base_v4.pth \
--epochs 30 --batch-size 64 --optimizer SGDP --lr 0.1 --scheduler multistep --milestones 10 20 25 --gamma 0.1 --wd 0.0005 --nesterov \
--augmentation --aug_prob 0.5 --mixed_aug
$*