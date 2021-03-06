import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import nsml

from .data import TagImageInferenceDataset


def inference(model, test_path: str) -> pd.DataFrame:
    # test dataset
    testset = TagImageInferenceDataset(root_dir='{}/test_data'.format(test_path))
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
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


def cast_int8_to_float32(model_dict):
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            model_dict[key] = model_dict[key].type(torch.float32)
    return model_dict


def cast_float32_to_int8(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'weight' in key and key.replace('weight', 'step') in model_dict.keys():
            new_dict[key] = model_dict[key].type(torch.int8)
        elif 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
            continue
        else:
            if key not in new_dict.keys():
                new_dict[key] = model_dict[key]
    return new_dict


def bind_quant_model(model, **kwargs):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = torch.load(filename)
        state = cast_int8_to_float32(state)
        model.load_state_dict(state, strict=False)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        state = model.state_dict()
        state = cast_float32_to_int8(state)
        torch.save(state, filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)


def inference_ensemble(model, test_path: str) -> pd.DataFrame:
    # test dataset
    testset = TagImageInferenceDataset(root_dir='{}/test_data'.format(test_path))
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # for ensemble
    for i in range(len(model)):
        model[i].eval()
    
    # results
    y_pred = []
    filename_list = []

    # inference loop
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x = data['image']
            x = x.to(device)
            
            # for ensemble
            logits = [k(x) for k in model]
            prob = torch.stack([F.softmax(k, dim=1) for k in logits]).mean(dim=0)
            pred = torch.argmax(prob, dim=-1)

            y_pred += pred.detach().cpu().tolist()
            filename_list += data['image_name']

    ret = pd.DataFrame({'file_name': filename_list, 'y_pred': y_pred})
    return ret


def bind_ensemble_quant_model(model, **kwargs):
    def load(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        states = torch.load(filename)
        for i, key in enumerate(states.keys()):
            state = states[key]
            state = cast_int8_to_float32(state)
            model[i].load_state_dict(state, strict=False)

    def save(save_folder, **kwargs):
        filename = os.path.join(save_folder, 'model.pth')
        states = {}
        for i in range(len(model)):
            state = model[i].state_dict()
            state = cast_float32_to_int8(state)
            states['m' + str(i).zfill(2)] = state
        torch.save(states, filename)
        print('Model saved')

    def infer(data_path, **kwargs):
        return inference_ensemble(model, data_path)

    nsml.bind(save=save, load=load, infer=infer)