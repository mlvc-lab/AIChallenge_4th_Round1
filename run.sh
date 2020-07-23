#!/bin/bash
apt-get update
apt-get install libglib2.0-0
pip install -r /root/volume/Base/requirement.txt
pip install opencv-python
pip install adamp
python /root/volume/Base/main.py -a efficientnet -C --efficient-type 0 -transfer --image-size 224 --nest things
python /root/volume/Base/main.py -a efficientnet -C --efficient-type 0 --image-size 224 --nest things
python /root/volume/Base/main.py -a mobilenetv2 -C -transfer --image-size 224 --nest things
python /root/volume/Base/main.py -a mobilenetv2 -C --image-size 224 --nest things
python /root/volume/Base/main.py -a rexnet -C --width-mult 1.0 --image-size 224 -transfer --nest things
python /root/volume/Base/main.py -a rexnet -C --width-mult 1.0 --image-size 224 --nest things
python /root/volume/Base/main.py -a resnet -C --layers 18 --image-size 224 -transfer --nest things
python /root/volume/Base/main.py -a resnet -C --layers 18 --image-size 224 --nest things
$*