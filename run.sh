#!/bin/bash
apt-get update
pip install -r /root/volume/Base/requirements.txt
pip install opencv-python
pip install opencv-contrib-python
pip install adamp
apt-get install -y libglib2.0-0
python /root/volume/Base/main.py -a rexnet --width-mult 1.3 -D --sched multistep --batch-size 256 -C -transfer --image-size 224 --nest --lr 0.1 thingsv4
$*
