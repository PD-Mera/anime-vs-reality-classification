import logging
import os
import time, datetime, math
import torch
from torch.utils.data import DataLoader

from data import LoadDataset
from config import *
from model import init_model
from optimizer import init_optimizer
from loss import init_loss
from utils import parse_opt, load_cfg, \
                  create_confusion_matrix, update_confusion_matrix, draw_confusion_matrix


def train(config):
    os.makedirs(config['train']['model_savepath'], exist_ok=True)

    logging.basicConfig(filename = os.path.join(config['log']['path'], config['log']['name']),
                        format = '%(asctime)s : %(levelname)s : %(name)s : %(message)s',
                        filemode = config['log']['mode'], )

    logger = logging.getLogger() 
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO) 

    config['logger'] = logger

    model_type = config['modelname']

    train_data = LoadDataset(config, phase = 'train')
    valid_data = LoadDataset(config, phase = 'valid')
    train_loader = DataLoader(train_data, batch_size=config['train']['batch_size'], num_workers=config['train']['num_workers'], shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=config['valid']['batch_size'], num_workers=config['valid']['num_workers'])
    
    model = init_model(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = init_optimizer(model, config)
    loss_fn = init_loss(config)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    highest_acc = 0

    batch_number = math.ceil(len(train_loader.dataset) / config['train']['batch_size'])
    logger.info("Start training loop")
    for epoch in range(config['train']['epoch']):
        # Start Training
        model.train()
        start = time.time()
        train_correct = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            pred_ = output.argmax(dim=1, keepdim=True)
            target_ = target.argmax(dim=1, keepdim=True)
            train_correct += pred_.eq(target_.view_as(pred_)).sum().item()

            if batch_number // 10 > 0:
                print_fre = batch_number // 10
            else:
                print_fre = 1
            if batch_idx % print_fre == print_fre - 1:
                iter_num = batch_idx * len(data)
                total_data = len(train_loader.dataset)
                iter_num = str(iter_num).zfill(len(str(total_data)))
                total_percent = 100. * batch_idx / len(train_loader)
                logger.info(f'Train Epoch {epoch + 1}: [{iter_num}/{total_data} ({total_percent:2.0f}%)] | Loss: {loss.item():.6f}')
                

        # Start Validating
        logger.info(f"Validating {len(valid_loader.dataset)} images")
        model.eval()
        valid_correct = 0
        c_matrix = create_confusion_matrix(config['class']['num'])

        for (data, target) in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            target = target.argmax(dim=1, keepdim=True)
            valid_correct += pred.eq(target.view_as(pred)).sum().item()
            c_matrix = update_confusion_matrix(c_matrix, pred, target)

        logger.info('Validation completed\n')

        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        valid_accuracy = 100. * valid_correct / len(valid_loader.dataset)
        logger.info('Train set: Accuracy: {}/{} ({:.2f}%)'.format(
            train_correct, len(train_loader.dataset), train_accuracy))
        logger.info('Valid set: Accuracy: {}/{} ({:.2f}%)'.format(
            valid_correct, len(valid_loader.dataset), valid_accuracy))
        logger.info(f'\nConfusion matrix of valid set\n{c_matrix}')

        
        stop = time.time()
        runtime = stop - start
        eta = int(runtime * (config['train']['epoch'] - epoch - 1))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info(f'Runing time: Epoch {epoch + 1}: {str(datetime.timedelta(seconds=int(runtime)))} | ETA: {eta}')

        torch.save(model.state_dict(), os.path.join(config['train']['model_savepath'], f'{model_type}_last.pth'))
        logger.info(f"Saving last model to {os.path.join(config['train']['model_savepath'], f'{model_type}_last.pth')}\n")

        if valid_accuracy >= highest_acc:
            highest_acc = valid_accuracy
            torch.save(model.state_dict(), os.path.join(config['train']['model_savepath'], f'{model_type}_best.pth'))
            logger.info(f"Saving best model to {os.path.join(config['train']['model_savepath'], f'{model_type}_best.pth')}\n")
            draw_confusion_matrix(config, c_matrix.cpu().numpy())

if __name__ == '__main__':
    opt = parse_opt()
    cfg = load_cfg(opt.cfg)
    train(config = cfg)