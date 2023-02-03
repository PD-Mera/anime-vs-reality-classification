import argparse
import yaml
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def load_cfg(cfg_path):
    with open(cfg_path, mode='r') as f:
        yaml_data = f.read()

    data = yaml.load(yaml_data, Loader=yaml.Loader)
    return data

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./cfg/config.yaml', help='cfg.yaml path')
    return parser.parse_args()

def create_confusion_matrix(num_class: int):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.zeros((num_class, num_class)).to(device)

def update_confusion_matrix(confusion_matrix, predict_batch, target_batch):
    predict_batch = predict_batch.squeeze(1)
    target_batch = target_batch.squeeze(1)
    for i in range(target_batch.size(0)):
        confusion_matrix[predict_batch[i]][target_batch[i]] += 1
    return confusion_matrix

def draw_confusion_matrix(config, confusion_matrix):
    cm_config = config['confusion_matrix']
    df_cm = pd.DataFrame(confusion_matrix, index=config['class']['name'], columns=config['class']['name'])
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap='Oranges', fmt='.0f') # font size
    plt.title('Confusion Matrix', fontsize = 20, pad = 10)
    plt.xlabel('Target', fontsize = 16) # x-axis label with fontsize 16
    plt.ylabel('Predict', fontsize = 16) # x-axis label with fontsize 16
    plt.tight_layout()
    plt.savefig(config['modelname'] + '_' + cm_config['name'])
    plt.close()