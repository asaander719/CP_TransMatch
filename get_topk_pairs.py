import os
import time
import numpy as np
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter

import csv
from sys import argv
import json
import pdb
from torch.nn import *
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import shutil
import yaml
from pretrain import *

from data.utility import Dataset
# from trainer.TransMatch_pretrain import TransMatch
from trainer.TransE import TransE
from trainer.TransR import TransR
from util.eval_utils import *

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def to_tensor(conf, data):
    return torch.tensor(data, dtype=torch.int64).to(conf["device"])

def _get_IJs_for_U_(conf, dataset, model, train_ij_pairs):
    new_u_ij_dict = {}
    ij_pairs = to_tensor(conf, train_ij_pairs)
    Is = ij_pairs[:, 0]
    Js = ij_pairs[:, 1]
    i_rep = model.i_embeddings_i(Is)
    j_rep = model.i_embeddings_i(Js)
    j_bias = model.i_bias_l(Js)
    vis_I = dataset.visual_features[Is]
    vis_J = dataset.visual_features[Js]
    I_visual = model.visual_nn_comp(vis_I) #bs, hidden_dim
    J_visual = model.visual_nn_comp(vis_J)
    J_bias_v = model.i_bias_v(Js)

    # for user_idx in all_user:
    for user_idx in range(len(dataset.user_map)):  
        u_idx = to_tensor(conf,user_idx)   #key
        u_rep = model.u_embeddings_l(u_idx.expand(Is.size(0))) #Is.size(0), hd
        u_rep_v = model.u_embeddings_v(u_idx.expand(Is.size(0))) #Is.size(0), hd
        if conf['pretrained_model'] == 'TransR':
            print(u_rep.size())
            projection_matrix = model.projection_matrix(u_idx.expand(Is.size(0))).view(u_rep.size(0), model.hidden_dim, model.hidden_dim).transpose(1,2)
            i_rep = torch.matmul(i_rep.unsqueeze(1), projection_matrix).squeeze(1)
            j_rep = torch.matmul(j_rep.unsqueeze(1), projection_matrix).squeeze(1)

            projection_matrix_v = model.projection_matrix_v(u_idx.expand(Is.size(0))).view(u_rep_v.size(0), model.hidden_dim, model.hidden_dim).transpose(1,2)
            I_visual = torch.matmul(I_visual.unsqueeze(1), projection_matrix_v).squeeze(1)
            J_visual = torch.matmul(J_visual.unsqueeze(1), projection_matrix_v).squeeze(1)

        distances = model.transE_predict(u_rep, i_rep, j_rep, j_bias)
        distances_v = model.transE_predict(u_rep_v, I_visual, J_visual, J_bias_v)
        distances += distances_v

        topk_scores, topk_indices = torch.topk(distances.view(-1), k=5, dim=-1)
        topk_i_j_pairs = ij_pairs[topk_indices]
        new_u_ij_dict[int(user_idx)] = topk_i_j_pairs.cpu().numpy().tolist()
        
    if conf['pretrained_model'] == 'TransR':
        with open('data/iqon_s/TransR_u_topk_ijs_dict.json', 'w') as json_file:
            json.dump(new_u_ij_dict, json_file)
    elif conf['pretrained_model'] == 'TransE':        
        with open('data/iqon_s/u_topk_ijs_dict.json', 'w') as json_file:
            json.dump(new_u_ij_dict, json_file)

    new_u_Is_Js_dict= {}
    for key, value in new_u_ij_dict.items():
        i_values = [item[0] for item in value]  # 获取 'i' 的值
        j_values = [item[1] for item in value]  # 获取 'j' 的值
        new_u_Is_Js_dict[key] = [i_values, j_values]

    if conf['pretrained_model'] == 'TransR':
        with open('data/iqon_s/TransR_u_topk_Is_Js_dict.json', 'w') as json_file:
            json.dump(new_u_Is_Js_dict, json_file)
    elif conf['pretrained_model'] == 'TransE': 
        with open('data/iqon_s/u_topk_Is_Js_dict.json', 'w') as json_file:
            json.dump(new_u_Is_Js_dict, json_file)

    return  new_u_ij_dict, new_u_Is_Js_dict

### 内存不够，目前的实验进行到完成TransR预训练，但是无法使用TransR筛选topk,pairs !!!下面两个函数只能提取TransE的前topk pairs

def _get_UJs_for_I_(conf, dataset, model, train_uj_pairs):
    new_i_uj_dict = {}
    uj_pairs = to_tensor(conf,train_uj_pairs)
    Us = uj_pairs[:, 0]
    Js = uj_pairs[:, 1]
    u_rep = model.u_embeddings_l(Us)
    j_rep = model.i_embeddings_i(Js)
    j_bias = model.i_bias_l(Js)
    vis_U = model.u_embeddings_v(Us)
    vis_J = dataset.visual_features[Js]

    J_visual = model.visual_nn_comp(vis_J)
    J_bias_v = model.i_bias_v(Js)

    # for Ihead in all_items:
    for I_idx in range(len(dataset.item_map)):  
        head_idx = to_tensor(conf,I_idx)   #key
        i_rep = model.i_embeddings_i(head_idx.expand(Us.size(0))) #Us.size(0), hd
        distances = model.transE_predict(u_rep, i_rep, j_rep, j_bias)
        I_visual = dataset.visual_features[head_idx] #2048
        I_visual = model.visual_nn_comp(I_visual)
        distances_v = model.transE_predict(vis_U, I_visual.unsqueeze(0).expand(Us.size(0),-1), J_visual, J_bias_v)
        distances += distances_v

        topk_scores, topk_indices = torch.topk(distances.view(-1), k=5, dim=-1) #k=conf['top_k_i']
        topk_u_j_pairs = uj_pairs[topk_indices]
        new_i_uj_dict[int(I_idx)] = topk_u_j_pairs.cpu().numpy().tolist()
    with open('data/iqon_s/I_topk_UJs_dict.json', 'w') as json_file:
        json.dump(new_i_uj_dict, json_file)

    new_i_Us_Js_dict = {}
    for key, value in new_i_uj_dict.items():
        i_values = [item[0] for item in value]  # 获取 'i' 的值
        j_values = [item[1] for item in value]  # 获取 'j' 的值

        new_i_Us_Js_dict[key] = [i_values, j_values]

    with open('data/iqon_s/i_topk_Us_Js_dict.json', 'w') as json_file:
        json.dump(new_i_Us_Js_dict, json_file) 

    return new_i_uj_dict, new_i_Us_Js_dict

def _get_UIs_for_J_(conf, dataset, model, train_ui_pairs):
    new_j_ui_dict = {}
    ui_pairs = to_tensor(conf,train_ui_pairs)
    Us = ui_pairs[:, 0]
    Is = ui_pairs[:, 1]
    u_rep = model.u_embeddings_l(Us)
    i_rep = model.i_embeddings_i(Is)

    vis_U = model.u_embeddings_v(Us)
    vis_I = dataset.visual_features[Is]

    I_visual = model.visual_nn_comp(vis_I)

    for J_idx in range(len(dataset.item_map)):  
        tail_idx = to_tensor(conf,J_idx)   #key
        j_rep = model.i_embeddings_i(tail_idx.expand(Us.size(0))) #Us.size(0), hd
        j_bias = model.i_bias_l(tail_idx.expand(Us.size(0)))
        J_bias_v = model.i_bias_v(tail_idx.expand(Us.size(0)))
        distances = model.transE_predict(u_rep, i_rep, j_rep, j_bias)
        J_visual = dataset.visual_features[tail_idx] #2048
        J_visual = model.visual_nn_comp(J_visual)
        distances_v = model.transE_predict(vis_U, I_visual, J_visual.unsqueeze(0).expand(Us.size(0),-1), J_bias_v)
        distances += distances_v

        topk_scores, topk_indices = torch.topk(distances.view(-1), k=5, dim=-1) #k=conf['top_k_i']
        topk_u_i_pairs = ui_pairs[topk_indices]
        new_j_ui_dict[int(J_idx)] = topk_u_i_pairs.cpu().numpy().tolist()
        
    with open('data/iqon_s/J_topk_UIs_dict.json', 'w') as json_file:
        json.dump(new_j_ui_dict, json_file)  

    new_j_Us_Is_dict = {}
    for key, value in new_j_ui_dict.items():
        i_values = [item[0] for item in value]  # 获取 'i' 的值
        j_values = [item[1] for item in value]  # 获取 'j' 的值

        new_j_Us_Is_dict[key] = [i_values, j_values]
    with open('data/iqon_s/j_topk_Us_Is_dict.json', 'w') as json_file:
        json.dump(new_j_Us_Is_dict, json_file) 

    return new_j_ui_dict, new_j_Us_Is_dict



def main():
    conf = yaml.safe_load(open("./config/train_model_config.yaml"))
    conf["dataset"] = "iqon_s"
    conf["gpu"] = 1
    conf["device"] = torch.device("cuda:%s"%conf["gpu"] if torch.cuda.is_available() else "cpu")
    dataset = Dataset(conf)
    global logger
    logger = get_logger()

    conf["user_num"] = len(dataset.user_map)
    conf["item_num"] = len(dataset.item_map)
    conf["cate_num"] = len(dataset.cate_items)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    conf['pretrained_model'] = 'TransE' # 'TransE'
    pretrain_model_file = f"{conf['pretrained_model']}.pth.tar"
    pretrain_model_dir = "model/iqon_s/pretrained_model/"
    pretrain_model_path = os.path.join(pretrain_model_dir, pretrain_model_file)
    conf['context_enhance'] = 0
    conf['path_enhance'] = 0

    if os.path.exists(pretrain_model_path):
        logger.info("=> loading model ...")
        model = torch.load(pretrain_model_path)
        print("Testing with existing model...")
        conf['use_pretrain'] = True
        model.to(conf["device"])
        logger.info(model)
    else:
        conf['pretrain_mode'] = True
        print('<<<<<<<< Start Pre-training >>>>>>>>')
        Train_Eval(conf)
        print('<<<<<<<< Pre-training End >>>>>>>>')
        conf['pretrain_mode'] = False
        model = torch.load(pretrain_model_path)
        print("Testing with existing model...")
        conf['use_pretrain'] = True
        model.to(conf["device"])
        logger.info(model)
    train_df = pd.read_csv("data/iqon_s/train.csv", header=None).astype('int')
    train_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    test_df = pd.read_csv("data/iqon_s/test.csv", header=None).astype('int')
    test_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    valid_df = pd.read_csv("data/iqon_s/val.csv", header=None).astype('int')
    valid_df.columns=["user_idx", "top_idx", "pos_bottom_idx", "neg_bottom_idx"]
    all_bottoms_id = pd.concat([train_df["pos_bottom_idx"], test_df["pos_bottom_idx"], valid_df["pos_bottom_idx"],
        train_df["neg_bottom_idx"], test_df["neg_bottom_idx"], valid_df["neg_bottom_idx"]], ignore_index=True).unique()

    train_ij_pairs = train_df[['top_idx', 'pos_bottom_idx']].drop_duplicates().values.tolist()
    train_ui_pairs = train_df[['user_idx', 'top_idx']].drop_duplicates().values.tolist()
    train_uj_pairs = train_df[['user_idx', 'pos_bottom_idx']].drop_duplicates().values.tolist()

    dataset.visual_features = dataset.visual_features.to(conf['device'])
    new_u_ij_dict, new_u_Is_Js_dict = _get_IJs_for_U_(conf, dataset, model, train_ij_pairs)
    new_i_uj_dict, new_i_Us_Js_dict = _get_UJs_for_I_(conf, dataset, model, train_uj_pairs)
    new_j_ui_dict, new_j_Us_Is_dict = _get_UIs_for_J_(conf, dataset, model, train_ui_pairs) 

if __name__ == '__main__':
    main()