from  periodictable import elements
import csv
import os
import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset


#for ele in elements:
#    print(ele.symbol)
def is_nan(x):
    return (x != x)


def create_template():
    num_elem = len(elements._element)+1
    value_list = [0.0]*num_elem
    res = np.array(value_list)
    res = np.expand_dims(res, axis=1)
    return res

# check if string contain int or float number
# or it is a true string 
def string_or_number(s):
    try:
        z = int(s)
        return z
    except ValueError:
        try:
            z = float(s)
            return z
        except ValueError:
            return s

dataset = []


def split_elename_and_value(row_data):
    """input name with shape into name and shape."""
  
    """
        MATCHING NUMBER:
        1. matches first digit ('\d')
         - \d matches all digits
        2. if present, matches decimal point ('\.?')
         - \. matches '.'
         - '?' means 'match 1-or-0 times'
        3. matches subsequent decimal digits ('\d*')
         - \d matches all digits
         - '*' means 'match >=0 times'
        
        MATCHING ELEMENT NAME:
        1. matches letters
         - [a-z A-Z] matches all letters
         - + matches 1-or-more times
    """
    name_pattern = r"\d\.?\d*|[a-z A-Z]+"
    splits = re.findall(name_pattern, row_data)
 
    # splits is to in format of pair of element name, value (all in strng)
    # the number of pairs varies 
    print(elements.symbol)

    # locate all symbol positions
    symbol_pos = []
    index = 0
    while index < len(splits):
        ele_sym = splits[index]
        print(type(ele_sym))
        if type(string_or_number(ele_sym)) == str:
            try:
                ele = elements.symbol(ele_sym)
                symbol_pos.append(index)
            except ValueError as msg:
                print(str(msg))

        index = index+1

    element_id = []
    element_value = []
    total_value = 0
    for pos in symbol_pos:
        init_pos = pos
        ele = elements.symbol(splits[init_pos])
        element_id.append(ele.number)
        init_pos = init_pos+1
        if init_pos < len(splits):
            value = string_or_number(splits[init_pos])
            if isinstance(value, int) or isinstance(value, float):
                element_value.append(value)
                total_value = total_value + value
            else:
                element_value.append(-1)
        else:
            element_value.append(-1)

    if element_value.count(-1) > 0:
        split_value = (1-total_value)/element_value.count(-1)
        element_value = [split_value if item == -1 else item for item in element_value]
         

    return list(zip(element_id, element_value))


def parse_file(file, sheet):
    data = pd.read_excel(open(file, 'rb'), sheet_name=sheet)
    
    np_arr = np.empty([1, 120, 1])
    label = []

    for index, row in data.iterrows():
        print(row.Compound)

        if(is_nan(row.Compound)==False):
            data_point = create_template()
            ret = split_elename_and_value(row.Compound)

            if(is_nan(row.Tc)==False):
                if isinstance(row.Tc, int) or isinstance(row.Tc, float):
                    for item in ret:
                        data_point[item[0]] = item[1]
            
                    data_point = np.expand_dims(data_point, axis=0)
                    np_arr= np.concatenate((np_arr, data_point), axis=0)
                    label.append(row.Tc)

    return np_arr[1:, :], np.array(label)

print(os.getcwd())


#x_train, y_train = parse_file('./AIML_work/superconductor.xlsx', 'Sheet1')
x_train, y_train = parse_file('superconductor.xlsx', 'Sheet1')
x_train_neg, y_train_neg = parse_file('superconductor.xlsx', 'Non-SC')
#x_train, y_train = parse_file('superconductor.xlsx', 'Sheet1')

x_train = np.concatenate((x_train, x_train_neg), axis=0)
y_train = np.concatenate((y_train, y_train_neg), axis=0)

y_train_1 = np.where(y_train > 0, 1, 0)
y_train = np.concatenate((y_train.reshape((-1, 1)), y_train_1.reshape((-1, 1))), axis=1)

def create_model():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 7, stride=1, padding=(5,5))
            self.conv1_1 = nn.Conv2d(32, 32, 5, stride=1, padding=(3,3))
            self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
            self.dropout1 = nn.Dropout(0.25)
            self.dropout2 = nn.Dropout(0.5)
            self.fc1 = nn.Linear(768, 64)
            self.fc2 = nn.Linear(64, 1)
            self.fc3 = nn.Linear(64, 2)
            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv1_1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x1 = self.fc2(x)
            output = torch.flatten(x1)
            x2 = self.fc3(x)
            x2 = self.softmax(x2)
            output1 = x2
            return [output, output1]
    model = Net()
    return model


x_train = x_train.reshape((-1, 1, 10, 12))*100

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)


train_ds = TensorDataset(x_train, y_train)


net = create_model()
optimizer = optim.Adam(net.parameters(), lr=0.0001)
#loss_fn = torch.nn.L1Loss(reduction='sum')
loss_fn = torch.nn.MSELoss(reduction='sum')

epochs = 5000

train_loader = DataLoader(train_ds, 64, drop_last=True, shuffle=False, num_workers=1)


net.double()

for epoch in range(epochs):

    net.train()

    loss = {}

    for xb, yb in train_loader:
         
        def closure():
            optimizer.zero_grad()
            y_split = np.split(yb, 2, axis=1)
            [pred, pred1] = net(xb)
            pred1_split = np.split(pred1, 2, axis=1)
            cls =  (pred1_split[0] < pred1_split[1]).float()
            loss = loss_fn(pred.flatten(), y_split[0].flatten()) + loss_fn(cls.flatten(), y_split[1].flatten())
            loss.backward()
            return loss

        optimizer.step(closure)

    net.eval()
    
    valid_loss = 0
    with torch.no_grad():
        for xb, yb in train_loader:
            y_split = np.split(yb, 2, axis=1)
            [pred, pred1] = net(xb)
            pred1_split = np.split(pred1, 2, axis=1)
            cls =  (pred1_split[0] < pred1_split[1]).float()
            loss1 = loss_fn(pred.flatten(), y_split[0].flatten()) 
            loss2 = loss_fn(cls.flatten(), y_split[1].flatten())
            valid_loss = valid_loss + loss1 + loss2

    print(epoch, (valid_loss/(len(train_loader)*64)))

net.float()

dummy_input = torch.randn(1, 1, 10, 12)


torch.onnx.export(net, dummy_input ,"model_pytorch.onnx", verbose=True)   


