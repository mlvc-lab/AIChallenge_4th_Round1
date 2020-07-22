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




def config():
    r"""configuration settings
    """
    parser = argparse.ArgumentParser(description='AI-Challenge Base Code')
    parser.add_argument('dataset', metavar='DATA', default='imagenet',
                        choices=dataset_names,
                        help='dataset: ' +
                             ' | '.join(dataset_names) +
                             ' (default: imagenet)')
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
                         help='wepth multiplier network (rexnet)')
    parser.add_argument('--datapath', default='/dataset/ImageNet', type=str, metavar='PATH',
                        help='where you want to load/save your dataset? (default: ../data)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel')
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
                        default='cosine', type=str, choices=schedule_types,
                        help='scheduler: ' +
                             ' | '.join(schedule_types) +
                             ' (default: cosine)')
    parser.add_argument('--step-size', dest='step_size', default=30,
                        type=int, metavar='STEP',
                        help='period of learning rate decay / '
                             'maximum number of iterations for '
                             'cosine annealing scheduler (default: 30)')
    parser.add_argument('--milestones', metavar='EPOCH', default=[30,80], type=int, nargs='+',
                        help='list of epoch indices for multi step scheduler '
                             '(must be increasing) (default: 30 80)')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='multiplicative factor of learning rate decay (default: 0.1)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path of checkpoint for testing model (default: none)')
    parser.add_argument('-E', '--evaluate', dest='evaluate', action='store_true',
                        help='test model?')
    parser.add_argument('-C', '--cuda', dest='cuda', action='store_true',
                        help='use cuda?')
    parser.add_argument('-g', '--gpuids', metavar='GPU', default=[0],
                        type=int, nargs='+',
                        help='GPU IDs for using (default: 0)')
    parser.add_argument('--efficient-type', default=0, type=int, help="select efficient type (0 : b0, 1 : b1, 2 : b2 ...)")

    # for finetuning
    parser.add_argument('-pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')

    parser.add_argument('--classnum', type=int, default=1000, help='class number when you use finetune method')

    parser.add_argument('-transfer', dest='transfer', action="store_true",help='use Imagenet for transfer learning')
    parser.add_argument('--image-size', default=224, type=int, help="input image size")
    cfg = parser.parse_args()
    return cfg
