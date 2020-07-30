#!/bin/bash
apt-get update
pip install opencv-python
pip install opencv-contrib-python
apt-get install -y libglib2.0-0
pip install adamp
pip install sklearn

cd /root/volume/AIChallenge_base
python main.py things --datapath /dataset/things_v5 -j 8 -a rexnet --width-mult 1.0 --depth-mult 1.0 -C -g 0 1 --save rexnet_1.0_quant0.pth --load rexnet_1.0.pth --transfer --src-dataset imagenet \
-Q --quantizer lsq --quant-bitw 8 --quant-bita 32 --quant-cfg base \
--epochs 120 --batch-size 128 --optimizer SGDP --lr 0.1 --scheduler multistep --milestones 60 90 110 --gamma 0.1 --wd 0.0005 --nesterov \
--augmentation --aug_prob 0.7 --mixed_aug
$*