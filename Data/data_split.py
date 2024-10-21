import torch
import numpy as np
from sklearn.model_selection import train_test_split

data = torch.load('data.pth')

data_train , data_test = train_test_split(data, test_size=0.3 , shuffle=True)
torch.save(data_test,'data_test.pth')
torch.save(data_train,'data_train.pth')
