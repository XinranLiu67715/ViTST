#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json
import os
from Networks.models import base_vit_384
from dataset import my_data
from torch.utils.data import DataLoader
import torch
from utils import preprocess_input, setup_seed
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
import pickle
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import pdb

setup_seed(args.seed)
logger = logging.getLogger('mnist_AutoML')

def DataProcess(filename, GroundTruth, gt=True, Exisfiles = False):
    if Exisfiles:
        wtdata = filename
    else:
        wtdata = pd.read_pickle(filename)
    x = []
    y = []
    for key in wtdata.keys():
        feature = wtdata[key]
        if gt:
            label = GroundTruth["Measurements"][key]['quantity']
            y.append(label)
        x.append(feature)
    x = np.array(x)
    y = np.array(y)
    return x, y


def test(args = None):
    y_obser = []
    y_estim = []
    json_path = args['json_path']
    with open(json_path, "r") as load_f:
        result_json = json.load(load_f)
        image_list = []
        variety_list = []
        measurements = result_json["Measurements"]
        for image_key in measurements.keys():
            image_list.append(measurements[image_key]['RGBImage'])
            variety_list.append(measurements[image_key]['quantity'])
    root_path = './data'
    trait='quantity'
    image_fea_test = {}
    images_list = []
    images_indexs = result_json['Measurements'].keys()
    for images_index in images_indexs:
        images_list.append(result_json['Measurements'][images_index]['RGBImage'])
    test_data = os.listdir(os.path.join(root_path, 'testing'))
    test_dataset = my_data(root_path=os.path.join(root_path, 'testing'), output_path='output', dataset=test_data,
                        augmentation=False, resize_shape=600, traits=trait, phase='testing')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    model = base_vit_384(pretrained=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load('./out/feature.pkl'), strict=False)
    model.eval()
    for _, batch in enumerate(test_loader):
            img, index_num = preprocess_input(phase='testing', input=batch)
            with torch.no_grad():
                fea,_ = model(img)
            fea = fea.cpu().numpy()[0]
            image_fea_test.update({"Image{}".format(index_num[0]): fea})

    test_x, test_y = DataProcess(image_fea_test, result_json, gt=True, Exisfiles = True)
    with open('./out/stacking.pkl', 'rb') as f:
        model = pickle.load(f)     
    fit_pred = model.predict(test_x)
    fit_pred= [int(i) for i in fit_pred]
    y_obser.append(test_y)
    y_estim.append(fit_pred)
    y_obser = np.array(y_obser)
    y_estim = np.array(y_estim)
    return y_obser, y_estim

def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    m, n = y_true.shape
    sum = 0
    for i in range(m):
        for j in range(n):
            sum = sum+y_true[i, j] 
    sum = sum / n
    sum_total = 0
    for i in range(m):
        for j in range(n):
            indi_error = np.abs((y_pred[i, j]  - y_true[i, j] ) / sum)
            sum_total += indi_error
        ma = (sum_total / n)  * 100
        print("MAPE: {}".format(ma))
    return ma


def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    m, n = y_true.shape
    sum_total = 0
    for i in range(m):
        for j in range(n):
            indi_error = abs((y_true[i, j] - y_pred[i, j]))
            sum_total += indi_error
        mae = sum_total / n
        print("MAE: {}".format(mae))
    return mae


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    y_obser, y_estim = test(params)
    res = mae(y_obser, y_estim)
