cd /root/volume/AIchallenge/distill_qt/
CUDA_VISIBLE_DEVICES=0 python distill.py --distype KD --test './checkpoint/resnet56/cifar100/AT56.pth'&
CUDA_VISIBLE_DEVICES=0 python distill.py --distype AT --test './checkpoint/resnet56/cifar100/AT56.pth'&
wait
