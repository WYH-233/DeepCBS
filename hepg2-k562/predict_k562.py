import datetime

import torch

import matplotlib.pyplot as plt

import torch.nn as nn

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from model import DeepCBS_Network

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, precision_recall_curve

from sklearn.model_selection import StratifiedKFold, KFold

import util_1

from datasets import gm12878

import random
sed=18
l1_alpha=1e-5


# 返回每一折多个epoch中的最优模型



def getDataSet(X_test_,y_test_):

    

 

    test_DataSet=gm12878(X_test_,y_test_)

   

   

  
    test_DataLoader = DataLoader(dataset=test_DataSet, batch_size=test_Batch_Size, shuffle=True)

    return  test_DataLoader

def test(myDataLoader, best_model_name):

    
    name='predict_k562'
    model.load_state_dict(torch.load(best_model_name))

    output_list = []

    output_result_list = []

    correct_list = []
    with torch.no_grad():

        for step, (x, y) in enumerate(myDataLoader):

            model.eval()

            output,_,_ = model(x)

            output_list += output.cpu().detach().numpy().tolist()

            output = (output > 0.5).int()
            output_result_list += output.cpu().detach().numpy().tolist()

       

            correct_list += y.cpu().detach().numpy().tolist()


    y_pred=np.array(output_result_list)

    y_true=np.array(correct_list)

    accuracy=accuracy_score(y_true,y_pred)
    ROC, PR, F1,threshold,fpr,tpr = util_1.draw_ROC_Curve(output_list, output_result_list, correct_list,  name)
    return ROC, PR, F1,fpr,tpr

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    test_Batch_Size = 64
    X = np.load('../all_sequence_ctcf_k562.npy')
    y =np.load('../all_label_ctcf_k562.npy')
    
    np.random.seed(sed)
    np.random.shuffle(X)
    np.random.seed(sed)
    np.random.shuffle(y)
    import random
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    
    X_test_=X
    y_test_=y
   
    test_Dataloader=getDataSet(X_test_,y_test_)
    model = DeepCBS_Network().to(device)
    Epoch=30
    loss_func = nn.BCELoss().to(device)
    best_model_name = 'all_data_model_'+str(Epoch)+'.pkl'
    ROC, PR, F1,fpr,tpr = test(test_Dataloader, best_model_name)
 
    print("ROC:{}\t PR:{}\t F1:{}\tfpr:{}\t tpr:{}".format(ROC,PR,F1,fpr,tpr))

   

