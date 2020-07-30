import torch


def cast2int8(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            model_dict[key] = model_dict[key].type(torch.float32)
    return model_dict


def cast2float32(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            model_dict[key] = model_dict[key].type(torch.int8)
    return model_dict


def load_quant_model(model, ckpt_file, main_gpu, use_cuda: bool=True, strict=True):
    r"""Load model for training, resume training, evaluation,
    quantization and finding similar kernels for new methods
    """
    if use_cuda:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage.cuda(main_gpu))
        checkpoint = cast2float32(checkpoint)
        try:
            model.load_state_dict(checkpoint, strict)
        except:
            model.module.load_state_dict(checkpoint, strict)
    else:
        checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
        checkpoint = cast2float32(checkpoint)
        try:
            model.load_state_dict(checkpoint)
        except:
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                if k[:7] == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k[:]
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

    return checkpoint