import os

from tqdm import tqdm
from easydict import EasyDict as ed

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchinfo import summary
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from metrics import sequential_accuracy, eval_accuracy

'''
* ModelCheckpoint
* EarlyStopping
* LearningRateScheduler (미완성)
'''

class ModelCheckpoint:
    def __init__(self, monitor='loss', save_path='./ckpts', save_best_only=True):
        if not os.path.exists(save_path): os.mkdir(save_path)
        self.monitor = monitor
        self.save_best_only = save_best_only
        
        self.save_path = save_path
        self.best_score = 0.0
        
        self.last_file = os.path.join(save_path, 'last_model.pt')
        self.best_file = os.path.join(save_path, 'best_model.pt')
        # self.last_model = os.path.join(save_path, 'last_model.pt')
        # self.best_model = os.path.join(save_path, 'best_model.pt')

    def monitoring(self, model, new_score, epoch):
        assert model, 'model 인자가 전달되지 않았습니다.'

        if epoch % 5 == 0:
            self.epoch_file = os.path.join(self.save_path, f'Epoch{epoch}_model.pt')
            torch.save(model, self.epoch_file)
            
        if not self.save_best_only:
            # torch.save(model.state_dict(), self.last_file)
            torch.save(model, self.last_file)
            
        if new_score > self.best_score:
            self.best_score = new_score
            # torch.save(model.state_dict(), self.best_file)
            torch.save(model, self.best_file)



class EarlyStopping:
    def __init__(self, monitor, patience, greater_is_better=False, start_from_epoch=0):
        self.monitor = monitor
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.start_from_epoch = start_from_epoch
        self.p_cnt = 0
        self.best_score = 0.0
    
    def monitoring(self, epoch, new_score):
        if self.start_from_epoch > epoch: return True
        
        if self.greater_is_better:
            if new_score < self.best_score:   self.p_cnt += 1
            else                          :   self.p_cnt = 0
        
        else:
            if new_score > self.best_score:   self.p_cnt += 1
            else                          :   self.p_cnt = 0
        
        # 조기종료 FLAG 전달
        if self.p_cnt >= self.patience:   return False
        else                          :   return True


class LearningRateScheduler:
    def __init__(self):
        pass
    


def validation_step(model, val_loader, device, loss_fn, epoch, eval_per_frame=False):
    #Validation step 
    model.eval()
    val_loss_step, val_accuracy_step = [], []
    val_loop = tqdm(val_loader, desc=f"Validation",
                      leave=True,
                      bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
                      )   

    with torch.no_grad():
        for X_batch, y_batch in val_loop:
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            y_pred = torch.permute(y_pred, (0, 2, 1))
            
            loss =loss_fn(y_pred, y_batch)
            accuracy = sequential_accuracy(y_pred, y_batch)
            
            
            val_loss_step.append(loss)
            val_accuracy_step.append(accuracy)
        
        # Epoch의 Validation 결과
        val_loss_epoch = torch.tensor(val_loss_step).mean()
        val_accuracy_epoch = torch.tensor(val_accuracy_step).mean()
    
    msg = '              val_loss: %.6f  val_accuracy: %.6f'%(val_loss_epoch, val_accuracy_epoch)
    print(msg)
    val_loop.close()

    val_scores = ed()
    val_scores.val_loss = val_loss_epoch
    val_scores.val_accuracy = val_accuracy_epoch
    return val_scores


def validation_step_kp(model, val_loader, device, loss_fn, epoch, num_classes, eval_per_frame=False):
    #Validation step 
    
    model.eval()
    val_loss_step, val_accuracy_step, val_f1_step = [], [], []
    val_loop = tqdm(val_loader, desc=f"Validation",
                      leave=True,
                      bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}'
                      )   

    with torch.no_grad():
        for X_batch, y_batch in val_loop:
            # y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            # y_pred = torch.permute(y_pred, (0, 2, 1))
            
            loss =loss_fn(y_pred, y_batch)
            accuracy = multiclass_accuracy(y_pred, y_batch)
            f1_score = multiclass_f1_score(y_pred, y_batch, num_classes=num_classes, average='macro')
            # accuracy = sequential_accuracy(y_pred, y_batch)
            
            val_loss_step.append(loss)
            val_accuracy_step.append(accuracy)
            val_f1_step.append(f1_score)
        
        # Epoch의 Validation 결과
        val_loss_epoch = torch.tensor(val_loss_step).mean()
        val_accuracy_epoch = torch.tensor(val_accuracy_step).mean()
        val_f1_epoch = torch.tensor(val_f1_step).mean()
    
    msg = '              val_loss: %.6f  val_accuracy: %.6f  val_f1_score: %.6f'%(val_loss_epoch, val_accuracy_epoch, val_f1_epoch)
    print(msg)
    val_loop.close()

    val_scores = ed()
    val_scores.val_loss = val_loss_epoch
    val_scores.val_accuracy = val_accuracy_epoch
    val_scores.val_f1 = val_f1_epoch
    return val_scores