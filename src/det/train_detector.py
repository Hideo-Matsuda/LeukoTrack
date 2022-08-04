import os
import shutil
import argparse
import datetime, pytz
import json

import torch
from torch.utils.data import DataLoader
from crest_det_dataset import CrestDetDataset

import utils
import transforms as T
from engine import train_one_epoch
from detection_utils import *

import yaml

ARG_CFG_FILE = 'cfg_file'
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(f'-{ARG_CFG_FILE[0]}', f'--{ARG_CFG_FILE}', required=True, type=str)
    cfg_path = vars(parser.parse_args())[ARG_CFG_FILE]

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    dataroot = config['data']['root']
    _seq_names = config['data'].get('seq')
    if _seq_names is None:
        _seq_names =  sorted(os.listdir(os.path.join(dataroot)))
    seq_names = []
    for _seq_name in _seq_names:
        if os.path.exists(os.path.join(dataroot, _seq_name, 'img1')) and \
            os.path.exists(os.path.join(dataroot, _seq_name, 'gt/gt.txt')) and \
            os.path.exists(os.path.join(dataroot, _seq_name, 'seqinfo.ini')):
            seq_names.append(_seq_name)

    outdir = config['out']['dir']
    if os.path.exists(outdir):
        # if output directory ( for save train model ) is exist, add date/time(Asia/Tokyo) at the end.
        timestamp = datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H-%M-%S')
        outdir += '_{}'.format(timestamp)

    outfile = os.path.join(outdir, 'train_log.txt')

    gpu_id = config['train'].get('gpu_id', -1)
    if gpu_id < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(gpu_id))
    msg = 'Using {}'.format(str(device))
    print(msg)
    os.makedirs(outdir, exist_ok=True)
    with open(outfile, 'a') as f:
        print(msg, file=f)
    shutil.copy2(cfg_path, os.path.join(outdir, os.path.basename(cfg_path)))

    dataset = CrestDetDataset(dataroot, seq_names=seq_names, transforms=get_transform(train=True)) # for training
    dataset_no_random = CrestDetDataset(dataroot, seq_names=seq_names, transforms=get_transform(train=False)) # for progress visualization

    num_workers = config['data'].get('num_workers', 4)
    data_loader = DataLoader(
        dataset, batch_size=config['train'].get('batch_size', 2), shuffle=True, num_workers=num_workers, 
        collate_fn=utils.collate_fn)
    data_loader_no_random = DataLoader(
        dataset_no_random, batch_size=1, shuffle=False, num_workers=num_workers,
        collate_fn=utils.collate_fn)

    model = get_detection_model(num_classes=dataset.num_classes)
    model_path = config.get('model', {}).get('initial_model')
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    model.to(device)

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
        lr=config['train'].get('optim', {}).get('lr', 0.00001), 
        momentum=config['train'].get('optim', {}).get('momentum', 0.9), 
        weight_decay=config['train'].get('optim', {}).get('weight_decay', 0.0005))
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=config['train'].get('optim', {}).get('step_lr', {}).get('step_size', 10), 
        gamma=config['train'].get('optim', {}).get('step_lr', {}).get('gamma', 1.0))

    eval_dir = os.path.join(outdir, 'progress')

    num_epochs = config['train']['epochs']
    initial_epoch = config['train'].get('initial_epoch', 1)
    final_epoch = initial_epoch + num_epochs - 1
    num_epoch_digits = len(str(final_epoch))

    for epoch in range(initial_epoch,final_epoch + 1):
        msg = 'Epoch {} ({} / {})'.format(epoch, epoch - initial_epoch + 1, num_epochs)
        print(msg)
        with open(outfile, 'a') as f:
            print(msg, file=f)
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, outfile=outfile)
        
        if (epoch - initial_epoch + 1) % config['out'].get('visual_freq', 3) == 0 or epoch == final_epoch:
            epoch_str = f'{epoch:0{num_epoch_digits}d}'
            evaluate_and_write_result_files(model, data_loader_no_random, 
                os.path.join(eval_dir, epoch_str), outfile=outfile)
        
        if (epoch - initial_epoch + 1) % config['out'].get('save_freq', 3) == 0 or epoch == final_epoch:
            epoch_str = f'{epoch:0{num_epoch_digits}d}'
            torch.save(model.state_dict(), os.path.join(outdir, f'{epoch_str}.pth'))
    