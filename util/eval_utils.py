import math
import numpy as np

def getHIT_MRR(pred, target_items):
    hit= 0.
    mrr = 0.
    p_1 = []
    for p in range(len(pred)):
        pre = pred[p]
        if pre in target_items:
            hit += 1
            if pre not in p_1:
                p_1.append(pre)
                mrr = 1./(p+1)

    return hit, mrr


def getNDCG(rank_list, target_items):
    dcg = 0
    idcg = IDCG(len(target_items))
    for i in range(len(rank_list)):
        item_id = rank_list[i]
        if item_id not in target_items:
            continue
        rank = i + 1
        dcg += 1./math.log(rank+1, 2)

    return dcg/idcg


def IDCG(n):
    idcg = 0.
    for i in range(n):
        idcg += 1./math.log(i+2, 2)

    return idcg


def get_metrics(grd, grd_cnt, pred, topk):
    REC, MRR, NDCG = [],[],[]
    for each_grd, each_grd_cnt, each_pred in zip(grd, grd_cnt, pred):
        NDCG.append(getNDCG(each_pred[:topk], [each_grd][:each_grd_cnt]))
        hit, mrr = getHIT_MRR(each_pred[:topk], [each_grd][:each_grd_cnt])
        REC.append(hit)
        MRR.append(mrr)
        
    REC = np.mean(REC)
    MRR = np.mean(MRR)
    NDCG = np.mean(NDCG)

    return REC, MRR, NDCG

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, pretrain_mode, patience, verbose=False, delta=0, trace_func=print):
        """
        Args:
            pretrain_mode (bool): Define the training mode, Default: True, pretrain_mode / if False, TransMatch training mode
            patience (int): How long to wait after last time validation loss improved.
                            Default: 8
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.pretrain_mode = pretrain_mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_auc_max = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_auc):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # self.pretrain_mode = False
                # if self.counter >= self.patience + self.patience:
                #     self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

    # def save_checkpoint(self, val_auc, model):
    #     '''Saves model when validation auc increase.'''
    #     if self.verbose:
    #         self.trace_func(f'Validation auc increase ({self.val_auc_max:.6f} --> {val_auc:.6f}).  Saving model ...')
    #     self.val_auc_max = val_auc