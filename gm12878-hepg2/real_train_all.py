import datetime

import torch

import matplotlib.pyplot as plt

import torch.nn as nn

import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

from model import BCL_Network

from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score, precision_recall_curve

from sklearn.model_selection import StratifiedKFold, KFold

import util_1

from datasets import gm12878

import random
sed=8
l1_alpha=1e-5


# 返回每一折多个epoch中的最优模型

def train(myDataLoader):

    best = 0
    threshold_best_model=0

    train_loss_list=[]

    test_loss_list=[]

    for epoch in range(Epoch):

        train_loss=0

        for step, (x, y) in enumerate(myDataLoader):

            model.train()

            output,_,_= model(x)

            output=output.squeeze(-1)

            loss = loss_func(output, y)
            temp=loss.item()
            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += torch.sum(abs(param))
            temp+=l1_alpha*regularization_loss
            train_loss+=float(temp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= myDataLoader.__len__()

        train_loss_list.append(train_loss)
        print('Train set: Average loss:{:.4f}'.format(train_loss))
    model_name ='all_data_'+'model_' +str(Epoch)+ '.pkl'
    torch.save(model.state_dict(), model_name)
    return model_name

def getDataSet(X_train,y_train):

    

    train_DataSet=gm12878(X_train,y_train)

  
   

    train_DataLoader = DataLoader(dataset=train_DataSet, batch_size=Batch_Size, shuffle=True)

  
    
    return train_DataLoader


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

    now_time = datetime.date.today()
    # 常用参数

    Batch_Size = 128
    test_Batch_Size = 64
    LR = 0.0001
    Epoch = 30
    X = np.load('../all_sequence_ctcf_gm12878.npy')
    
    y =np.load('../all_label_ctcf_gm12878.npy')
    
    np.random.seed(sed)
    np.random.shuffle(X)
    np.random.seed(sed)
    np.random.shuffle(y)
    import random
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
  
    train_DataLoader=getDataSet( X,y)
    model = BCL_Network().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    loss_func = nn.BCELoss().to(device)
    best_model_name = train(train_DataLoader)
   
 
 
   

    print("#################################")

   

