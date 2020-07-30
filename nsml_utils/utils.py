import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import nsml

from data import TagImageInferenceDataset


def inference(model, test_path: str) -> pd.DataFrame:
    # test dataset
    testset = TagImageInferenceDataset(root_dir='{}/test_data'.format(test_path))
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # results
    y_pred = []
    filename_list = []

    # inference loop
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=-1)

            y_pred += pred.detach().cpu().tolist()
            filename_list += data['image_name']

    ret = pd.DataFrame({'file_name': filename_list, 'y_pred': y_pred})
    return ret


def bind_model(model, **kwargs):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = torch.load(filename)
        model.load_state_dict(state, strict=True)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        torch.save(model.state_dict(), filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)


def cast_float32_to_int8(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            model_dict[key] = model_dict[key].type(torch.float32)
    return model_dict


def cast_int8_to_float32(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            new_dict[key] = model_dict[key].type(torch.int8)
        elif 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
            continue
        else:
            if key not in new_dict.keys():
                new_dict[key] = model_dict[key]
    return model_dict


def bind_quant_model(model, **kwargs):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = torch.load(filename)
        state = cast_float32_to_int8(state)
        model.load_state_dict(state, strict=False)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = model.state_dict()
        state = cast_int8_to_float32(state)
        torch.save(state, filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)