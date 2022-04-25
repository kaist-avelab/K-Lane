'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import numpy as np

EPS = 1e-16

def calc_measures(arr_label, arr_pred, mode = 'conf', is_wo_offset = False):
    if mode == 'conf':
        if is_wo_offset:
            TP, FP, FN, TN = calc_measures_conf_wo_offset(arr_label, arr_pred)
        else:
            TP, FP, FN, TN = calc_measures_conf(arr_label, arr_pred)
    else:
        if is_wo_offset:
            TP, FP, FN, TN = calc_measures_cls_wo_offset(arr_label, arr_pred)
        else:
            TP, FP, FN, TN = calc_measures_cls(arr_label, arr_pred)

    # print(TP, FP, FN, TN)

    accuracy = (TP+TN)/(TP+TN+FP+FN+EPS)
    precision = TP/(TP+FP+EPS)
    recall = TP/(TP+FN+EPS)
    f1 = (2*TP/(2*TP+FP+FN+EPS))

    return accuracy, precision, recall, f1

def calc_measures_conf(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: accuracy, precision, recall, f1
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    for row in range(1,143):
        for col in range(1,143):
            label = 0
            pred = 0
            pred_enhanced = 0
            label_enhanced = 0
            
            if ((temp_pred[row, col] == temp_label[row,col]) or
                (temp_pred[row, col+1] == temp_label[row,col]) or
                (temp_pred[row, col-1] == temp_label[row,col]) or
                (temp_pred[row-1, col+1] == temp_label[row,col]) or
                (temp_pred[row-1, col-1] == temp_label[row,col]) or
                (temp_pred[row+1, col-1] == temp_label[row,col]) or
                (temp_pred[row+1, col+1] == temp_label[row,col]) or
                (temp_pred[row-1, col] == temp_label[row,col]) or
                (temp_pred[row+1, col] == temp_label[row,col])
            ):
                pred_enhanced = OCCUPIED
            
            if ((temp_label[row, col] == temp_pred[row,col]) or
                (temp_label[row, col+1] == temp_pred[row,col]) or
                (temp_label[row, col-1] == temp_pred[row,col]) or
                (temp_label[row-1, col-1] == temp_pred[row,col]) or
                (temp_label[row-1, col+1] == temp_pred[row,col]) or
                (temp_label[row+1, col-1] == temp_pred[row,col]) or
                (temp_label[row+1, col+1] == temp_pred[row,col]) or
                (temp_label[row-1, col] == temp_pred[row,col]) or
                (temp_label[row+1, col] == temp_pred[row,col])
            ):
                label_enhanced = OCCUPIED
            
            label = temp_label[row,col]
            pred = temp_pred[row,col]
            
            if (label == OCCUPIED):
                if(pred_enhanced == OCCUPIED):
                    TP += 1
                else:
                    FN += 1

            if (pred == OCCUPIED):
                if(label_enhanced == NOT_OCCUPIED):
                    FP += 1

    TN = 144*144 - TP - FP - FN
    
    return TP, FP, FN, TN

def calc_measures_conf_wo_offset(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: f1-score
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    OCCUPIED = 1.
    NOT_OCCUPIED = 0.

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    for row in range(144):
        for col in range(144):
            label = 0
            pred = 0
            pred_enhanced = 0
            label_enhanced = 0
            
            if (temp_pred[row, col] == temp_label[row,col]):
                pred_enhanced = OCCUPIED
            
            if (temp_label[row, col] == temp_pred[row,col]):
                label_enhanced = OCCUPIED
            
            label = temp_label[row,col]
            pred = temp_pred[row,col]
            
            if (label == OCCUPIED):
                if(pred_enhanced == OCCUPIED):
                    TP += 1
                else:
                    FN += 1

            if (pred == OCCUPIED):
                if(label_enhanced == NOT_OCCUPIED):
                    FP += 1
    
    TN = 144*144 - TP - FP - FN

    return TP, FP, FN, TN

def calc_measures_cls(arr_label, arr_pred):
    '''
    * in : arr_label (np.array, float, 144x144)
    * in : arr_pred (np.array, float, 144x144)
    * out: f1-score
    '''

    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # TP, FP, FN, #TN
    for j in range(1,143):
        for i in range(1,143):
            if not (temp_label[j,i] == 255): # Lane
                is_tp = False
                for jj in range(-1,2):
                    for ii in range(-1,2):
                        is_tp = is_tp or (temp_pred[j+jj,i+ii] == temp_label[j,i])
                if is_tp:
                    TP += 1
                else:
                    FN += 1
            else: # Not Lane
                if not (temp_pred[j,i] == 255):
                    FP += 1
                else:
                    TN += 1
    
    return TP, FP, FN, TN

def calc_measures_cls_wo_offset(arr_label, arr_pred):
    temp_label = arr_label.copy()
    temp_pred = arr_pred.copy()

    # F1 = TP/(TP+0.5*(FP+FN))
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    # TP, FP, FN, #TN
    for j in range(144):
        for i in range(144):
            if not (temp_label[j,i] == 255): # Lane
                if temp_pred[j,i] == temp_label[j,i]:
                    TP += 1
                else:
                    FN += 1
            else: # Not Lane
                if not (temp_pred[j,i] == 255):
                    FP += 1
                else:
                    TN += 1
    
    return TP, FP, FN, TN
