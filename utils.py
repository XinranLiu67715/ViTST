
import torch
import shutil
import numpy as np
import random
     
def save_checkpoint(state,is_best, task_id, filename='checkpoint.pth'):
    torch.save(state, './'+str(task_id)+'/'+filename)
    if is_best:
        shutil.copyfile('./'+str(task_id)+'/'+filename, './'+str(task_id)+'/'+'model_best.pth')


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_input( phase, input):
    if 1==1:
        if phase == 'training':
            img, gt, index = input
            gt = gt.squeeze(1)
        if phase == 'testing':
            img, index = input
        else:
            img, gt, index = input

    if phase == 'training':
        return img, gt, index
    else:
        return img, index