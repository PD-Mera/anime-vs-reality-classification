import torch


def init_optimizer(model, config):
    assert config['train']['optimizer'] in config['OPTIMIZER_AVAILABLE'], f'"optimizer" in `config.py` must in {config["OPTIMIZER_AVAILABLE"]}'
    if config['train']['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['learning_rate'])

    return optimizer
