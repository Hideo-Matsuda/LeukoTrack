import os
import argparse
import datetime, pytz
from tokenize import group
import json
import yaml

import torch
from torch.utils.data import DataLoader

from utils import collate_fn

from crest_det_dataset import CrestDetDataset
from detection_utils import *

ARG_DATAROOT = 'dataroot'
ARG_GROUP = 'group'
ARG_MODEL_PATH = 'model_path'
ARG_OUTDIR = 'outdir'
ARG_GPU_ID = 'gpu_id'
ARG_NUM_WORKERS = 'num_workers'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(f'-{ARG_DATAROOT[0]}', f'--{ARG_DATAROOT}', type=str, required=True)
    parser.add_argument(f'-{ARG_GROUP[0]}', f'--{ARG_GROUP}', type=str, required=True)
    parser.add_argument(f'-{ARG_MODEL_PATH[0]}', f'--{ARG_MODEL_PATH}', type=str, required=True)
    parser.add_argument(f'-{ARG_OUTDIR[0]}', f'--{ARG_OUTDIR}', type=str, required=True)
    parser.add_argument(f'--{ARG_GPU_ID}', type=int, default=0)
    parser.add_argument(f'--{ARG_NUM_WORKERS}', default=4, type=int)

    args = vars(parser.parse_args())
    outdir = args[ARG_OUTDIR]
    if os.path.exists(outdir):
        timestamp = datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d-%H-%M-%S')
        outdir += '_{}'.format(timestamp)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(args, f, indent=4)

    dataroot = args[ARG_DATAROOT]
    group_yaml = args[ARG_GROUP]

    with open(group_yaml) as file:
        test_group = yaml.safe_load(file)
    
    _seq_names = test_group['test_group']
    seq_names = []
    for _seq_name in _seq_names:
        if os.path.exists(os.path.join(dataroot, _seq_name, 'img1')) and \
            os.path.exists(os.path.join(dataroot, _seq_name, 'seqinfo.ini')):
            seq_names.append(_seq_name)
        else:
            print('{} or {} does not exist.'.format(os.path.join(dataroot, _seq_name, 'img1'), os.path.join(dataroot, _seq_name, 'seqinfo.ini')))

    dataset = CrestDetDataset(args[ARG_DATAROOT], seq_names=seq_names, transforms=get_transform(train=False), mode='test')
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args[ARG_NUM_WORKERS],
        collate_fn=collate_fn)

    model = get_detection_model(num_classes=dataset.num_classes)
    if args[ARG_GPU_ID] < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args[ARG_GPU_ID]))
    model.load_state_dict(torch.load(args[ARG_MODEL_PATH]))
    if args[ARG_GPU_ID] < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args[ARG_GPU_ID]))
    print('Using {}'.format(str(device)))

    model.to(device)
    evaluate_and_write_result_files(model, data_loader, outdir=outdir)
