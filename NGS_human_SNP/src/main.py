import os, sys, glob
import json
import numpy as np
import random
import pdb
import pandas as pd
import tqdm
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split

torch.cuda.is_available()

json_files = glob.glob('../Peng_lib4_classes/*.json')

node_features, global_features = [],[]
for f in json_files:
    df = pd.read_json(f)
    # print(df['node_features'].values[0])
    for node in df['node_features'].values:
        node_features.append(node)

    for g in df['global_features'].values:
        g_array = [g['K(obs)'],g['nG<x t^>'],g['nG[x]{t^*}'],g['nG[x t^]'],g['T']]
        global_features.append(g_array)
        
    
nf = torch.Tensor(node_features)
gf = torch.Tensor(global_features)
# print(nf.shape)
# print(gf)

stat_file = '../processed_201802/DSP1998_feature_stats.json'

s_df = pd.read_json(stat_file)
# display(s_df)
# print(s_df['std']['K(obs)'])
# print(type(s_df['std']['K(obs)']))
std_kobs = torch.from_numpy(np.asarray(s_df['std']['K(obs)']))
# print(type(std_kobs))

conversion_matrix = [
        [1,0],  #   A: is_purine,  !makes_3_H_bonds
        [0,1],  #   C: !is_purine, makes_3_H_bonds
        [0,0],  #   T: !is_purine, !makes_3_H_bonds
        [1,1]   #   G: is_purine,  makes_3_H_bonds
    ]
cm_tensor = torch.Tensor(conversion_matrix)
seq_mask = np.array([1,1,0,1,0]).reshape([1,-1])
seq_mask_tensor = torch.from_numpy(seq_mask)
# print(seq_mask_tensor)
# new_f = np.concatenate([np.matmul(np.array(nf[:][:][:][:4]), conversion_matrix), nf[:][:][:][4:]])
apply_cv = torch.matmul(nf[:,:,:4],cm_tensor)
other_f = nf[:,:,4:]

new_nf = torch.cat([apply_cv,other_f],dim=2)
# print(new_nf.shape)
node_data = new_nf * seq_mask_tensor
# print(masked_f.shape)

class PengDataset(torch.utils.data.Dataset):
    def __init__(self, node_data, global_data):
        super(PengDataset, self).__init__()
        self.node_data = node_data
        self.global_data = global_data

    def __getitem__(self,index):
        X = self.node_data[index]
        y = self.global_data[index]
        return X, y

    def __len__(self):
        return len(self.node_data)

X, y = node_data, gf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

train_dataset = PengDataset(X_train, y_train)
val_dataset = PengDataset(X_val, y_val)
test_dataset = PengDataset(X_test, y_test)

batch_size = 999
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(len(train_dataset))
print(train_dataset.node_data.shape)
print(len(val_dataset))
print(val_dataset.node_data.shape)
print(len(test_dataset))
print(test_dataset.node_data.shape)

def xavier_init(b,n):
    shape = (n, b)
    #print('shape==>', shape, (*shape))/
    scale = np.sqrt((6.0) / (b + n))
    out = torch.from_numpy(scale * (2 * np.random.rand(*shape).astype(np.float32) -1))
    #print(type(out), out.shape)
    return out 

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__() # 모델 연산 정의
        self.fc1 = nn.Linear(260, 256, bias=False)
        # self.bn1 = nn.BatchNorm1d(num_features=256)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128, bias=False)
        # self.bn2 = nn.BatchNorm1d(num_features=128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(128, 2, bias=False)
        # self.bn3 = nn.BatchNorm1d(num_features=2)
        # self.dropout3 = nn.Dropout(0.8)
        self.fc1.weight = nn.Parameter(xavier_init(260,256))
        self.fc2.weight = nn.Parameter(xavier_init(256,128))
        self.fc3.weight = nn.Parameter(xavier_init(128,2))

        # torch.nn.init.xavier_uniform_(self.fc1.weight, gain=torch.nn.init.calculate_gain("linear"))
        # torch.nn.init.xavier_uniform_(self.fc2.weight, gain=torch.nn.init.calculate_gain("linear"))
        # torch.nn.init.xavier_uniform_(self.fc3.weight, gain=torch.nn.init.calculate_gain("linear"))
    
    def forward(self, x): # 모델 연산의 순서를 정의
        # print('run mlp foward',x)
        x = F.relu(self.fc1(x)) # Linear 계산 후 활성화 함수 ReLU를 적용한다.  
        # x = self.bn1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        # x = self.bn3(x)
        # x = self.dropout3(x)
        return x

class DualBiGRU(nn.Module):
    def __init__(self, node_data, gf):
        super(DualBiGRU,self).__init__()
        self.node_data = node_data
        self.gf = gf
        self.mlp = MLP()
        self.gru1 = nn.GRU(5, 128, num_layers=1, bidirectional=True, batch_first=True)
        self.gru2 = nn.GRU(5, 128, num_layers=1, bidirectional=True, batch_first=True)
        
        '''
        모델을 하나로 같이 쓰면 파라미터가 공유되서 안된다. 따로 선언해서 써야함.
        node input을 (:,:80,5) -> target, (:,80:,5) -> probe로 나눠서 bigru를 통과시킨 후에 합쳐서 MLP로 넘겨주는 부분을 어떻게 만들지?
        '''
    def forward(self, x, gf):
        #print('x shape, gf shape in model foward def ==>', x.shape, gf.shape )
        tx = x[:,:80,:]
        px = x[:,80:,:]
        # print('tx, px shape==>',tx.shape, px.shape)
        _, th = self.gru1(tx)
        _, ph = self.gru2(px)
        
        # print('th, ph, gf shape==>',th.shape, ph.shape, gf.shape)
        h = torch.cat([th[0] + ph[0], th[1] + ph[1]], 1)
        # print('h, h shape==>', h, h.shape)
        # # print(type(h[0]), type(gf))

        h = torch.cat([h, gf], dim=1)
        
        # print('h shape==>',h.shape)
        mlp_output = self.mlp(h)
        # print('pred mean==>', mlp_output[0])
        return (torch.reshape(mlp_output[:,0], [-1]), torch.reshape(abs(mlp_output[:,1]), [-1]))

class Trainer():
    def __init__ (self, trainloader, valloader, testloader, model, criterion, optimizer, device):
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader        
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs = 1):
        self.model.to(self.device)
        start_time = time.time()

        for epoch in range(epochs):
            losses = []
            accuracy = batch_size = 0
            self.model.train()
            for iter, (x_batch, y_batch) in tqdm.tqdm(enumerate(self.trainloader)):
                # print('Batch [%d]' % iter)
                # print('x and y batch shape==>', x_batch.shape, y_batch.shape)
                # print(x_batch, y_batch[:,1:])
                batch_size += x_batch.size(dim=0)
                # print('batch_size_iter', batch_size)
                gf_batch = y_batch[:,1:]
                # print(gf_batch.shape, gf_batch)
                
                y_batch = y_batch[:,:1]
                # print('y batch shape==>', y_batch.shape)

                x_batch = x_batch.to(self.device)
                gf_batch = gf_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                
                pred_mean, pred_pricision = self.model(x_batch,gf_batch)
                # import pdb; pdb.set_trace()
                # print('pred_mean==>', pred_mean.shape)
                # self.model.zero_grad()

                # loss = self.criterion(pred_mean[0], y_batch)
                # print('pred_mean==>',pred_mean)
                loss = torch.mean(0.5 * ((pred_mean.unsqueeze(-1) - y_batch)**2))
                accuracy += torch.sum((pred_mean.unsqueeze(-1) - y_batch)**2)
                # import pdb; pdb.set_trace()
                # print(f"GRU Grad: {self.model.gru.weight.grad}")
                # print(f"MLP Grad: {self.model.mlp.weight.grad}")
                # loss.register_hook(lambda grad: print(grad))
                # print('accuracy_in_batch ==>', accuracy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                losses.append(loss.item())
                # break
            
            accuracy = std_kobs * torch.sqrt((accuracy / batch_size))
            # print('batch_size, accuracy==>', batch_size, accuracy)
            
            end_time = time.time() - start_time
            end_time = str(datetime.timedelta(seconds=end_time))[:-7]
            print('Time [%s], Epoch [%d/%d], loss: %.4f, accuracy: %.4f'
                  % (end_time, epoch+1, epochs, np.mean(losses), accuracy))
            
            if(epoch % 100 == 0):
                val_accuracy = batch_size = 0
                # def validate(self):
                val_losses = []
                self.model.eval() 
                with torch.no_grad():

                    for x_batch, y_batch in self.valloader:

                        batch_size += x_batch.size(dim=0)

                        gf_batch = y_batch[:,1:]               
                        y_batch = y_batch[:,:1]

                        x_batch = x_batch.to(self.device)
                        gf_batch = gf_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        pred_mean, pred_pricision = self.model(x_batch,gf_batch)
            
                        val_loss = torch.mean(0.5 * ((pred_mean.unsqueeze(-1) - y_batch)**2))
                        val_accuracy += torch.sum((pred_mean.unsqueeze(-1) - y_batch)**2)
                        val_losses.append(val_loss.item())

                    val_accuracy = std_kobs * torch.sqrt((val_accuracy / batch_size))

                    print('Validation== loss: %.4f, accuracy: %.4f' % (np.mean(val_losses), val_accuracy))
        
    def test(self):
        self.model.to(self.device)
        test_losses = []
        test_accuracy = batch_size = 0

        self.model.eval() 
        with torch.no_grad():

            for x_batch, y_batch in self.valloader:

                batch_size += x_batch.size(dim=0)

                gf_batch = y_batch[:,1:]               
                y_batch = y_batch[:,:1]

                x_batch = x_batch.to(self.device)
                gf_batch = gf_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred_mean, pred_pricision = self.model(x_batch,gf_batch)
    
                test_loss = torch.mean(0.5 * ((pred_mean.unsqueeze(-1) - y_batch)**2))
                test_accuracy += torch.sum((pred_mean.unsqueeze(-1) - y_batch)**2)
                test_losses.append(test_loss.item())

            test_accuracy = std_kobs * torch.sqrt((test_accuracy / batch_size))

            print('Testß== loss: %.4f, accuracy: %.4f' % (np.mean(test_losses), test_accuracy))

model = DualBiGRU(node_data, gf)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device('cuda')

trainer = Trainer(trainloader = train_dataloader, 
                  valloader = val_dataloader,
                  testloader = test_dataloader,
                  model = model,
                  criterion=criterion,
                  optimizer = optimizer,
                  device=device)

trainer.train(epochs=1000)
test_acc = trainer.test()
# print('test_acc:', test_acc)