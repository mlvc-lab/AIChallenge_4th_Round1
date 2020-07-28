import os

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

import nsml

from data import TagImageInferenceDataset


def inference(model, test_path: str) -> pd.DataFrame:
    # test dataset
    testset = TagImageInferenceDataset(root_dir='{}/test_data'.format(test_path),
                                       transform=test_transform)
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
