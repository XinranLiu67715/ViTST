from __future__ import division
import warnings
from Networks.models import base_vit_384
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
warnings.filterwarnings('ignore')
import time
import torch.nn as nn
import json
from deepforest import CascadeForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from Networks.models import base_vit_384
from dataset import my_data
from torch.utils.data import DataLoader
import torch
from utils import preprocess_input
import torch.nn as nn
import pickle
from test import DataProcess 


setup_seed(args.seed)
logger = logging.getLogger('mnist_AutoML')



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    train_file = './npydata/windtrain.npy'
    test_file = './npydata/windtest.npy'
    json_path = args['json_path']
    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()
    model = base_vit_384(pretrained=True)
    model = nn.DataParallel(model)
    model = model.cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(
        [ 
            {'params': model.parameters(), 'lr': args['lr']},
        ], lr=args['lr'], weight_decay=args['weight_decay'])
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)

    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))
    torch.set_num_threads(args['workers'])
    train_data = pre_data(train_list, args, train=True)
    test_data = pre_data(val_list, args, train=False)
    start = time.time()
    for epoch in range(args['start_epoch'], args['epochs']):   
        train(train_data, model, criterion, optimizer, epoch, args, scheduler)    
        if epoch % 5 == 0 and epoch >= 10:
            end1 =  time.time()
            prec1 = test(test_data, model, args)
            end2 = time.time()
            is_best = prec1 < args['best_pred']
            args['best_pred'] = min(prec1, args['best_pred'])
            print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']), args['save_path'], end1 - start, end2 - end1)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args['pre'],
                'state_dict': model.state_dict(),
                'best_prec1': args['best_pred'],
                'optimizer': optimizer.state_dict(),
            }, is_best, args['save_path'])
    end = time.time()
    with open(json_path, "r") as load_f:
        GroundTruth_list = json.load(load_f)
        image_list = []
        variety_list = []
        measurements = GroundTruth_list["Measurements"]
        for image_key in measurements.keys():
            image_list.append(measurements[image_key]['RGBImage'])
            variety_list.append(measurements[image_key]['quantity'])
    staking_train(gt=True, split_training=False, groundtruth_list=GroundTruth_list,Json_Path = json_path)
    


def pre_data(train_list, args, train):
    print("Pre_load dataset ......")
    data_keys = {}
    count = 0
    for j in range(len(train_list)):
        Img_path = train_list[j]
        fname = os.path.basename(Img_path)
        img, gt_count = load_data(Img_path, args, train)
        blob = {}
        blob['img'] = img
        blob['gt_count'] = gt_count
        blob['fname'] = fname
        data_keys[count] = blob
        count += 1
    return data_keys

def train(Pre_data, model, criterion, optimizer, epoch, args, scheduler):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            batch_size=args['batch_size'],
                            num_workers=args['workers'],
                            args=args),
        batch_size=args['batch_size'], drop_last=False)
    args['lr'] = optimizer.param_groups[0]['lr']
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args['lr']))
    model.train()
    end = time.time()
    for i, (fname, img, gt_count) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        out = model(img) 
        out1 = out[1]
        gt_count = gt_count.type(torch.FloatTensor).cuda().unsqueeze(1)
        loss = criterion(out1, gt_count)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args['print_freq'] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
    torch.save(model.state_dict(), './out/feature.pkl')
    scheduler.step()


def staking_train(gt=True, split_training=False, transformer=None, groundtruth_list=None, Json_Path = None):
    trait = 'quantity'
    root_path = '/home/yrx/lxr/TransCrowd-main/data/wfan/'

    image_fea_tra = {}
    images = os.listdir(os.path.join(root_path, 'tr1'))
    gt_json = Json_Path
    train_data = images
    train_dataset = my_data(root_path=os.path.join(root_path, 'tr1'), output_path='output', label_path=gt_json,
                            dataset=train_data, augmentation=False, resize_shape=600,
                            traits=trait, phase='testing')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
    model = base_vit_384(pretrained=True)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load('./out/feature.pkl'), strict=False)
    model.eval()
    for _, batch in enumerate(train_loader):
        img, index_num = preprocess_input( phase='testing', input=batch)
        with torch.no_grad():
            print(type(model(img)))
            fea, pre= model(img)
        fea = fea.cpu().numpy()[0]
        image_fea_tra.update({"Image{}".format(index_num[0]): fea})
    train_x, train_y = DataProcess(image_fea_tra, groundtruth_list, gt=gt,Exisfiles = True)
    estimators = [
        ('df', CascadeForestRegressor(random_state=50, predictor="xgboost", n_estimators=4,
         n_trees=90,n_jobs=(-1))),
        ('rbf',  HistGradientBoostingRegressor())
       ]
    model = StackingRegressor(estimators=estimators) 
    model.fit(train_x, train_y)
    with open('./out/stacking.pkl', 'wb') as f:
        pickle.dump(model, f)


def test(Pre_data, model, args):
    print('begin test')
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(Pre_data, args['save_path'],
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                            ]),
                            args=args, train=False),
        batch_size=1)
    model.eval()
    for _, (fname, img, gt_count) in enumerate(test_loader):
        img = img.cuda()
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        with torch.no_grad():
            out = model(img)
            out1 = out[1]
            count = torch.sum(out1).item()
        gt_count = torch.sum(gt_count).item()
        mae += abs(gt_count - count)
        mse += abs(gt_count - count) * abs(gt_count - count)
        print('{fname} Gt {gt:.2f} Pred {pred}'.format(fname=fname[0], gt=gt_count, pred=count))
    mae = mae * 1.0 / (len(test_loader) * batch_size)
    mse = math.sqrt(mse / (len(test_loader)) * batch_size)
    nni.report_intermediate_result(mae)
    print(' \n* MAE {mae:.3f}\n'.format(mae=mae), '* MSE {mse:.3f}'.format(mse=mse))
    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    import pdb
    pdb.set_trace()
    main(params)
