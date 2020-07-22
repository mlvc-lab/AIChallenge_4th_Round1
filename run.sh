#!/bin/bash
apt-get update
apt-get install libglib2.0-0
pip install -r /root/volume/Base/requirement.txt
pip install opencv-python
pip install adamp
python /root/volume/Base/main.py -a rexnet --width-mult 2.1 --depth-mult 2.1 -C imagenet $*