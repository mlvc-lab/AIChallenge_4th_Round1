import argparse
import models
from data import valid_datasets as dataset_names


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

r'''learning rate scheduler types
    - step: Decays the learning rate of each parameter group
            by gamma every step_size epochs.
    - multistep: Decays the learning rate of each parameter group
                 by gamma once the number of epoch reaches one of the milestones.
    - exp: Decays the learning rate of each parameter group by gamma every epoch.
    - cosine: Set the learning rate of each parameter group
              using a cosine annealing schedule.
'''
schedule_types = [
    'step', 'multistep', 'exp', 'cosine'
]

# sacred setting
MONGO_URI = 'mongodb://ai:aichallenge!@mlvc.khu.ac.kr:31912/aichallenge'
MONGO_DB = 'aichallenge'
# previous setting
#MONGO_URI = 'mongodb://mlvc:mlvcdatabase!@mlvc.khu.ac.kr:31912'
#MONGO_DB = 'training'


def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='AI-Challenge Base Code')
    parser.add_argument('dataset', metavar='DATA', default='cifar10',
                        choices=dataset_names,
                        help='dataset: ' +
                             ' | '.join(dataset_names) +
                             ' (default: cifar10)')     
    # for model architecture
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet)')
    parser.add_argument('--layers', default=56, type=int, metavar='N',
                        help='number of layers in ResNet (default: 56)')
    parser.add_argument('--width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')
    parser.add_argument('--depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier network (rexnet)')
    parser.add_argument('--model-mult', default=0, type=int,
                        help="e.g. efficient type (0 : b0, 1 : b1, 2 : b2 ...)")
    # for dataset
    parser.add_argument('--datapath', default='../data', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    # for learning policy
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
    parser.add_argument('--optimizer', default='SGD', type=str,
                        help='name of optimizer to train the model (default: SGD)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)',
                        dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        default=5e-4, type=float,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--nest', '--nesterov', dest='nesterov', action='store_true',
                        help='use nesterov momentum?')
    parser.add_argument('--sched', '--scheduler', dest='scheduler', metavar='TYPE',
                        default='step', type=str, choices=schedule_types,
                        help='scheduler: ' +
                             ' | '.join(schedule_types) +
                             ' (default: step)')
    parser.add_argument('--step-size', dest='step_size', default=30,
                        type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 30)')
    parser.add_argument('--milestones', metavar='EPOCH', default=[100,150], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 100 150)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    # for gpu configuration
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    # specify run type
    parser.add_argument('--run-type', default='train', type=str, metavar='TYPE',
                        help='type of run the main function e.g. train or evaluate (default: train)')
    # for ensemble evaluation
    parser.add_argument('--ensemble', dest='ensemble', action='store_true',
                        help='If this is set, apply ensemble method for evaluation.')
    parser.add_argument('--ensemble-loads', metavar='CKPT', default=[], type=str, nargs='+',
                        help='list of checkpoint files for ensemble method')
    # for load and save
    parser.add_argument('--load', default=None, type=str, metavar='FILE.pth',
                        help='name of checkpoint for testing model (default: None)')
    parser.add_argument('--save', default='ckpt.pth', type=str, metavar='FILE.pth',
                        help='name of checkpoint for saving model (default: ckpt.pth)')
    # for transfer Learning
    parser.add_argument('--transfer', dest='transfer', action="store_true",
                        help='use Imagenet for transfer learning')
    parser.add_argument('--src-dataset', default=None, type=str, metavar='FILE.pth',
                        help='name of checkpoint for transfer loading (default: None)')
    #############
    # for pruning
    parser.add_argument('-P', '--prune', dest='prune', action='store_true',
                         help='Use pruning')
    parser.add_argument('--pruner', default='dpf', type=str,
                        help='method of pruning to apply (default: dpf)')
    parser.add_argument('--prune-type', dest='prune_type', default='unstructured',
                         type=str, help='specify \'unstructured\' or \'structured\'')
    parser.add_argument('--prune-freq', dest='prune_freq', default=16, type=int,
                         help='update frequency')
    parser.add_argument('--prune-rate', dest='prune_rate', default=0.5, type=float,
                         help='pruning rate')
    ##################
    # for quantization
    parser.add_argument('-Q', '--quantize', dest='quantize', action='store_true',
                        help='If this is set, the model layers are changed to quantized layers.')
    parser.add_argument('--quantizer', default='lsq', type=str,
                        help='method of quantization to apply (default: lsq)')
    parser.add_argument('--quant-bitw', default=8, type=int, metavar='N',
                        help='number of bits for weights (default: 8)')
    parser.add_argument('--quant-bita', default=32, type=int, metavar='N',
                        help='number of bits for activations (default: 32)')
    parser.add_argument('--quant-cfg', default='base', type=str,
                        help='name of quantization configuration for each type of layers (default: base)')
    ##################
    # for distillation
    parser.add_argument('-D', '--distill', dest='distill', action='store_true',
                        help='If this is set, teacher model is loaded to distill the training student.')
    parser.add_argument('--dist-type', default=None, type=str,
                        help='self distillation type in \{KD, AT, SP\} (default: None)')
    # teacher arch
    parser.add_argument('--tch-arch', default='wideresnet', type=str,
                        help='choose pre-trained teacher')
    parser.add_argument('--tch-layers', default=56, type=int, metavar='N',
                        help='number of layers in ResNet (default: 56)')    # resnet, wideresnet
    parser.add_argument('--tch-width-mult', default=1.0, type=float, metavar='WM',
                        help='width multiplier to thin a network '
                             'uniformly at each layer (default: 1.0)')      # wideresnet, mobilenetv2, rexnet
    parser.add_argument('--tch-depth-mult', default=1.0, type=float, metavar='DM',
                         help='depth multiplier network (rexnet)')              # rexnet
    parser.add_argument('--tch-model-mult', default=0, type=int,
                        help="e.g. efficient type (0 : b0, 1 : b1, 2 : b2 ...)")     # efficientnet
    # teacher load
    parser.add_argument('--tch-load', default=None, type=str, metavar='FILE.pth',
                        help='name of checkpoint for teacher model (default: None)')
    #######################
    # for Data Augmentation
    parser.add_argument('--augmentation', default=False, dest='augmentation', action='store_true',
                        help='Set to apply data augmentation')
    parser.add_argument('--aug_type', default='cutmix', type=str,
                        help='type of augmentation: cutmix, saliencymix, cutout and mixup')
    parser.add_argument('--aug_prob', default=0.0, type=float,
                        help='cutmix probability')
    # for Cut-Mix and SaliencyMix
    parser.add_argument('--aug-beta', default=1.0, type=float,
                        help='hyperparameter beta for augmentation intensity')
    # for cutout
    parser.add_argument('--cut-nholes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--cut-length', type=int, default=16,
                        help='length of the holes')
    # for mixup
    parser.add_argument('--mixup-alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    # mixed aug
    parser.add_argument('--mixed_aug', dest='mixed_aug', action='store_true',
                        help='use mixed augmentation')

    cfg = parser.parse_args()
    return cfg
