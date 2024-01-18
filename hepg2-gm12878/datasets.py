import torch
from torch.utils.data import Dataset
import numpy as np
class gm12878(Dataset):
    def __init__(self,x,y):
        super(gm12878,self).__init__()
        self.X,self.Y=self.load_data(x,y)
    def __getitem__(self, index):
        return (self.X[index][:][:],self.Y[index])
    def __len__(self):
        return len(self.X)
    def load_data(self,x,y):
        X = torch.from_numpy(x).type(torch.FloatTensor).cuda()
        Y= torch.from_numpy(y).type(torch.FloatTensor).cuda()
        return X,Y
