from .qnn import *
from .iqnn import *

base = {
    'QuantConv2d': {
        # Whether to use symmetric quantization
        'symmetric': False,
        # If apply mask from pruned model
        'do_mask': False,
    },
    'QuantLinear': {
        'symmetric': False,
        'do_mask': False,
    },
    'QuantReLU': {
    },
    'QuantIdentity': {
        'symmetric': False,
    },
    'excepts': {
        'conv1': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': False,
        },
        'fc': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': False,
        },
    },
}

mask = {
    'QuantConv2d': {
        # Whether to use symmetric quantization
        'symmetric': False,
        # If apply mask from pruned model
        'do_mask': True,
    },
    'QuantLinear': {
        'symmetric': False,
        'do_mask': True,
    },
    'QuantReLU': {
    },
    'QuantIdentity': {
        'symmetric': False,
    },
    'excepts': {
        'conv1': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': False,
        },
        'fc': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': False,
        },
    },
}
