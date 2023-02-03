import torch.nn as nn
import os
import torch


def calculate_weights(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = []
    for classname in config['class']['name']:
        weights.append(len(os.listdir(os.path.join(config['train']['path'], classname))))
    
    total_datasize = sum(weights)
    weights = [(1.0 - (x / total_datasize)) for x in weights]
    return torch.tensor(weights, dtype=torch.float32).to(device)


class ClsfLoss():
    def __init__(self):
        super(ClsfLoss, self).__init__()
        pass
    def forward(self, x):
        return x


def init_loss(config):
    assert config['train']['loss_fn'] in config['LOSS_AVAILABLE'], f'"loss_fn" in config must in {config["LOSS_AVAILABLE"]}'

    if config['train']['loss_fn'] == 'custom':
        loss = ClsfLoss()
    elif config['train']['loss_fn'] == 'CE':
        weights = calculate_weights(config)
        loss = nn.CrossEntropyLoss(weight=weights)
    return loss