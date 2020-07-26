#!/bin/bash
apt-get update
apt-get install libglib2.0-0
pip install -r /root/volume/Base/requirement.txt
pip install opencv-python
pip install adamp
python /root/volume/Base/main.py -a efficientnet --sched multistep -C -transfer --image-size 224 --nest things
$*