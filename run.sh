#!/bin/bash
apt-get update
pip install -r /root/volume/Base/requirement.txt
pip install opencv-python
pip install opencv-contrib-python
pip install adamp
apt-get install -y libglib2.0-0
python /root/volume/Base/main.py -a efficientnet -D --sched multistep -C -transfer --image-size 224 --nest --lr 0.1 things
$*