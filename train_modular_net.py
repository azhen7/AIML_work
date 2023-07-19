
import os
print(os.getcwd())

from  periodictable import elements
import csv
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
import matplotlib.pyplot as plt


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    #print(elements.symbol)

    # locate all symbol positions
    symbol_pos = []
    index = 0
    while index < len(splits):
        ele_sym = splits[index]
        #print(type(ele_sym))
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
                element_value.append(max(0, value))
                total_value = total_value + value
            else:
                element_value.append(0)
        else:
            element_value.append(0)

    if  total_value >= 1:
        element_value[:] = [x / total_value for x in element_value]
        

    total_value = min(1, max(0, sum(element_value)))    
    
    if element_value.count(0) > 0:
        split_value = (1-total_value)/element_value.count(0)
        element_value = [split_value if item == -1 else item for item in element_value]
         

    return list(zip(element_id, element_value))


def parse_file(file, sheet):
    data = pd.read_excel(open(file, 'rb'), sheet_name=sheet)
    
    np_arr = np.empty([1, 120, 1])
    label = []

    for index, row in data.iterrows():
        #print(row.Compound)

        if(is_nan(row.Compound)==False):
            data_point = create_template()
            ret = split_elename_and_value(row.Compound)

            if(is_nan(row.Tc)==False):
                if isinstance(row.Tc, int) or isinstance(row.Tc, float):
                    for item in ret:
                        data_point[item[0]] = item[1]
            
                    data_point = np.expand_dims(data_point, axis=0)
                    np_arr= np.concatenate((np_arr, data_point), axis=0)
                    if row.Tc > 0:
                        label.append([0.0, 1.0, row.Tc])
                    else:
                        label.append([1.0, 0.0, 0])

    return np_arr[1:, :], np.array(label)

x_train, y_train = parse_file('seperated_results.xlsx', 'Fail')
x_train_neg, y_train_neg = parse_file('seperated_results.xlsx', 'Success')

x_train = np.concatenate((x_train, x_train_neg), axis=0)
y_train = np.concatenate((y_train, y_train_neg), axis=0)


# model constuction
def create_model():

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.seq1 = nn.Sequential(
                nn.Conv2d(1, 32, 7, stride=1, padding=(3,3)),
                nn.ReLU(),
                nn.Conv2d(32, 32, 5, stride=1, padding=(2,2)),
                nn.ReLU()
                )

            self.seq2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2),
                nn.Softmax(dim=1)
            )

            self.seq3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 1, stride=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, 1, stride=1),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)
            )

        def forward(self, x):
            x = self.seq1(x)
            output1 = self.seq2(x)
            output2 = self.seq3(x)
            return [output2, output1]
    model = Net()
    return model


x_train = x_train.reshape((-1, 1, 10, 12))*100


x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

train_ds = TensorDataset(x_train, y_train)

train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])

total_tc_train = 0
for i in range(len(train_ds)):
    total_tc_train = total_tc_train + (train_ds[i][1][2])

print(len(train_ds), total_tc_train/len(train_ds))

total_tc_val = 0
for i in range(len(val_ds)):
    total_tc_val = total_tc_val + (val_ds[i][1][2])

print(len(val_ds), total_tc_val/len(val_ds))


net = create_model()
net.to(dev)


optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_fn = torch.nn.MSELoss(reduction='sum')

epochs = 5000

train_loader = DataLoader(train_ds, 64, drop_last=True, shuffle=True, num_workers=1)
val_loader = DataLoader(val_ds, 1, drop_last=True, shuffle=True, num_workers=1)

pred_absdiff = []
train_accuracy = []
train_loss = []
net.double()

# prediction branch training
for epoch in range(epochs):

    if epoch == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    net.train()
   
    for xb, yb in train_loader:
         
        def closure():
            optimizer.zero_grad()
            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc 
            pred = net(xb_g)
            loss = loss_fn(pred[0].flatten(), yb_g_split_1.flatten())
            loss.backward()
            return loss

        optimizer.step(closure)

    net.eval()
    valid_loss = 0
    diff = 0
    count = 0
    with torch.no_grad():
        for xb, yb in train_loader:
      
            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc 
            pred = net(xb_g)
            valid_loss = valid_loss + loss_fn(pred[0].flatten(), yb_g_split_1.flatten())
            
            pred_data = pred[0].to(torch.device("cpu"))
            pred_data = pred_data.numpy().flatten()
            gt_data = yb_g_split[1].numpy().flatten()
           
           
            for i in range(len(pred_data)):
               diff = diff + np.absolute(pred_data[i] - gt_data[i])

            class_data = pred[1].to(torch.device("cpu"))
            gt_data = yb_g_split[0].numpy()
            for i in range(len(class_data)):
                pred_result = 1
                if class_data[i][0] > class_data[i][1]:
                    pred_result = 0
                gt_data_result = 0
                if gt_data[i][0] < gt_data[i][1]:
                    gt_data_result = 1    
                if pred_result == gt_data_result:
                    count+=1


    pred_absdiff.append((diff/(len(train_loader)*64)))
    train_accuracy.append((count/(len(train_loader)*64)))
    
    temp = valid_loss.to(torch.device("cpu"))
    train_loss.append((temp/(len(train_loader)*64)))
    print(epoch, (temp/(len(train_loader)*64)), (diff/(len(train_loader)*64)), (count/(len(train_loader)*64)))    


fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),pred_absdiff)
plt.title('Accuray')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['Train Pred Acc'],loc = 'lower right')
plt.savefig("./pred_train_accuracy.png",dpi = 600)

np.savetxt("pred_train_accu.csv", pred_absdiff, delimiter=",", fmt='%f')

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_loss)
plt.title('MSE Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['Train MSE Loss'],loc = 'lower right')
plt.savefig("./pred_train_loss.png",dpi = 600)

np.savetxt("pred_train_loss.csv", train_loss, delimiter=",", fmt='%f')

for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0001

net.train()

# freeze main + prediction branch 
for param in net.seq1.parameters():
    param.requires_grad = False

for param in net.seq3.parameters():
    param.requires_grad = False

epochs = 5000
pred_absdiff = []
train_accuracy = []
train_loss = []

# train classification branch 
for epoch in range(epochs):
    if epoch == 3000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001

    net.train()
   
    for xb, yb in train_loader:
         
        def closure():
            optimizer.zero_grad()
            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc 
            pred = net(xb_g)
            loss = loss_fn(pred[1], yb_g_split_0)
            loss.backward()
            return loss

        optimizer.step(closure)

    net.eval()
    valid_loss = 0
    diff = 0
    count = 0
    with torch.no_grad():
        for xb, yb in train_loader:
      
            xb_g = xb.to(dev)
            yb_g_split = np.split(yb, [2], 1)
            yb_g_split_0 = (yb_g_split[0]).to(dev) #class
            yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc 
            pred = net(xb_g)
            valid_loss = valid_loss + loss_fn(pred[1], yb_g_split_0)
            
            pred_data = pred[0].to(torch.device("cpu"))
            pred_data = pred_data.numpy().flatten()
            gt_data_tc = yb_g_split[1].numpy().flatten()
           
            class_data = pred[1].to(torch.device("cpu"))
            gt_data = yb_g_split[0].numpy()
            for i in range(len(class_data)):
                pred_result = 1
                if class_data[i][0] > class_data[i][1]:
                    pred_result = 0
                gt_data_result = 0
                if gt_data[i][0] < gt_data[i][1]:
                    gt_data_result = 1    
                if pred_result == gt_data_result:
                    count+=1

                if pred_result == gt_data_result:
                    if pred_result == 1:
                        diff = diff + np.absolute(pred_data[i] - gt_data_tc[i])
               
         


    pred_absdiff.append((diff/(len(train_loader)*64)))
    train_accuracy.append((count/(len(train_loader)*64)))
    
    temp = valid_loss.to(torch.device("cpu"))
    train_loss.append((temp/(len(train_loader)*64)))
    print(epoch, (temp/(len(train_loader)*64)), (diff/(len(train_loader)*64)), (count/(len(train_loader)*64)))

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_accuracy)
plt.title('Accuray')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend(['Train Class Acc'],loc = 'lower right')
plt.savefig("./class_accuracy.png",dpi = 600)

np.savetxt("class_train_accu.csv", train_accuracy, delimiter=",", fmt='%f')

fig = plt.figure()
plt.subplot(111)
plt.plot(np.arange(1,epochs+1),train_loss)
plt.title('MSE Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['Train MSE Loss'],loc = 'lower right')
plt.savefig("./class_train_loss.png",dpi = 600)

np.savetxt("class_train_loss.csv", train_loss, delimiter=",", fmt='%f')

# full model testing
net.eval()
diff = 0
count = 0
with torch.no_grad():
    for xb, yb in val_loader:
      
        xb_g = xb.to(dev)
        yb_g_split = np.split(yb, [2], 1)
        yb_g_split_0 = (yb_g_split[0]).to(dev) #class
        yb_g_split_1 = (yb_g_split[1]).to(dev) #Tc 
        pred = net(xb_g)
           
        pred_data = pred[0].to(torch.device("cpu"))
        pred_data = pred_data.numpy().flatten()
        gt_data_tc = yb_g_split[1].numpy().flatten()
           
        class_data = pred[1].to(torch.device("cpu"))
        gt_data = yb_g_split[0].numpy()
        for i in range(len(class_data)):
            pred_result = 1
            if class_data[i][0] > class_data[i][1]:
                pred_result = 0
            gt_data_result = 0
            if gt_data[i][0] < gt_data[i][1]:
                gt_data_result = 1    
            if pred_result == gt_data_result:
                count+=1
     
            if pred_result == gt_data_result:
                if pred_result == 1:
                    diff = diff + np.absolute(pred_data[i] - gt_data_tc[i])

print((diff/(len(val_loader)*1)), (count/(len(val_loader)*1)))

# model export to onnx
net.to(torch.device("cpu"))
net.float()
dummy_input = torch.randn(1, 1, 10, 12)
torch.onnx.export(net, dummy_input, "model_pytorch.onnx", verbose=True)
