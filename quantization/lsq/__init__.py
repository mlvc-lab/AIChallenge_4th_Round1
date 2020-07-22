from .qnn import *

base = {
    'QuantConv2d': {
        # the number of bits for weights
        'nbits': 8,
        # Whether to use symmetric quantization
        'symmetric': False,
        # If apply mask from pruned model
        'do_mask': False,
    },
    'QuantLinear': {
        'nbits': 8,
        'symmetric': False,
        'do_mask': False,
    },
    'QuantReLU': {
        'nbits': 32,
    },
    'QuantIdentity': {
        'nbits': 32,
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
        # the number of bits for weights
        'nbits': 8,
        # Whether to use symmetric quantization
        'symmetric': False,
        # If apply mask from pruned model
        'do_mask': True,
    },
    'QuantLinear': {
        'nbits': 8,
        'symmetric': False,
        'do_mask': False,
    },
    'QuantReLU': {
        'nbits': 32,
    },
    'QuantIdentity': {
        'nbits': 32,
        'symmetric': False,
    },
    'excepts': {
        'conv1': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': True,
        },
        'fc': {
            'nbits': 8,
            'symmetric': False,
            'do_mask': False,
        },
    },
}
