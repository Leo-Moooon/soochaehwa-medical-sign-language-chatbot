import os, sys
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score

modules_path = '/Users/GitHub/Projects/main_projects/hand_signals/modules'
sys.path.append(modules_path)


def eval_accuracy(y_pred, y_batch, seq_second=True):
    '''
    if seq_second == True:
        y_pred.shape: (batch, timestamp, num_classes)
        y_batch.shape: (batch, timestamp)
    
    elif seq_second == False:
        y_pred.shape: (batch, num_classes, timestamp)
        y_batch.shape: (batch, timestamp)
    '''
    
    total = y_batch.size(0) * y_batch.size(1)
    if seq_second: _, predicted = y_pred.max(1)
    else: _, predicted = y_pred.max(2)

    correct = predicted.eq(y_batch).sum().item()
    
    return correct / total
        
    
    
def sequential_accuracy(y_pred, y_batch):
    # y_pred shape: (batch, num_classes, timestamp) : CrossEntropyLoss 계산을 위해 num_classes를 앞으로 땡김
    # y_batch shape: (batch, timestamp) 
    maxlen = len(y_pred[0][0])
    
    acc_list = []
    for i in range(maxlen):
        acc = multiclass_accuracy(y_pred[:, :, i], y_batch[:, i])
        acc_list.append(acc)
    
    return float(sum(acc_list) / maxlen)