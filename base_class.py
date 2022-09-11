from re import U
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import Adam
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# Coffee Net for regression
class CoffeeNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, limit_interval):
        super().__init__()
        limit_interval = torch.tensor(limit_interval, dtype=torch.float32)
        self.scale = limit_interval[1,:] - limit_interval[0,:]
        self.lower_bound = limit_interval[0,:]
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim)
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear4', self.fc4)
        self.block.add_module('sigmoid', nn.Sigmoid())
    def forward(self, x):
        return self.scale * self.block(x) + self.lower_bound
        # return self.block(x) 

"""Coffee Net Classification for whether TAV>1"""
class CoffeeNet_TAV(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_units[0])
        self.fc2  = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3  = nn.Linear(hidden_units[1], hidden_units[2])
        self.fc4  = nn.Linear(hidden_units[2], output_dim)
        self.block = nn.Sequential()
        self.block.add_module('linear1', self.fc1)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear2', self.fc2)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear3', self.fc3)
        self.block.add_module('relu', nn.ReLU())
        self.block.add_module('linear4', self.fc4)
        self.block.add_module('sigmoid', nn.Sigmoid())
    def forward(self, x):
        return self.block(x) 


"""SK-SCORE for tensor type"""
def sk_score(u, utrue, sample_weight=None):
    u = u.detach().numpy(); utrue = utrue.detach().numpy()
    return r2_score(utrue, u, sample_weight = sample_weight)


"""ROC-Curve"""
def ROC(utrue, pred, method = 'aps'):
    if method == 'aps':
        average_precision = average_precision_score(pred, utrue)
        cm = confusion_matrix(utrue, pred)
        return average_precision, cm
    """PR curve"""
    if method == 'pr':
        precision, recall, _ = precision_recall_curve(utrue, pred)
        return precision, recall
    """ROC curve"""
    if method == 'roc':
        fpr, tpr, _ = roc_curve(pred, utrue)
        ra_score = roc_auc_score(utrue, pred)
        return fpr, tpr, ra_score




# Train the NN
def DNN_train(x_train, y_train, x_test, y_test, args, limit_interval = [[10, 80],[20, 100]], model_name = 'DNN_model', need_fig = True):

    x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype= torch.float32), torch.tensor(y_train, dtype= torch.float32),torch.tensor(x_test, dtype= torch.float32),torch.tensor(y_test, dtype= torch.float32)


    input_dim = len(x_train[0,:])
    output_dim = len(y_train[0,:])

    net = CoffeeNet(input_dim, output_dim, args['hidden_units'], limit_interval)
    batch_num = len(x_train[:,0]) // args['batch_size']

    criterion = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr = args['lr'], weight_decay = args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_decay_step'], gamma=args['lr_decay_rate'])

    t0 = time.time(); train_his = []; test_his = []

    for epoch in range(args['epoch']):
        net.eval()
        with torch.no_grad():
            y_dnn_test = net(x_test)
            test_loss = float(criterion(y_dnn_test, y_test))
        net.train()
        train_loss = 0
        for i in range(batch_num):
            x_train_batch = x_train[i * args['batch_size']: (i+1) * args['batch_size'], : ]
            y_train_batch = y_train[i * args['batch_size']: (i+1) * args['batch_size'], : ]
            y_dnn_train_batch = net(x_train_batch)
            train_loss_batch  = criterion(y_dnn_train_batch, y_train_batch)

            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()
            train_loss += float(train_loss_batch)
        scheduler.step()
        train_loss /= batch_num
        train_his.append(train_loss); test_his.append(test_loss)
        if epoch % 5 == 2:
            print('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
                        .format(epoch, train_loss, test_loss, int(time.time()-t0), optimizer.param_groups[0]['lr']))    


            state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'ini_args':args}
            if test_his[-1] > test_his[-2] and epoch > 2000:
                break
    if need_fig:
        torch.save(state ,f'./model/{model_name}.pth')
        plt.semilogy(np.arange(10, len(train_his)), train_his[10:], label = 'train')
        plt.semilogy(np.arange(10, len(test_his)), test_his[10:], label = 'test')
        plt.legend()
        plt.title('Loss figure')  
        plt.savefig(f'./pic/{model_name}+loss_fig.png') 
    return net, sk_score(y_train, net(x_train)), sk_score(y_test, net(x_test))

"""TAV Net Train"""
def DNN_train_TAV_classification(x_train, y_train, x_test, y_test, args,  model_name = 'TAV_DNN_model', need_fig = True):

    x_train, y_train, x_test, y_test = torch.tensor(x_train, dtype= torch.float32), torch.tensor(y_train, dtype= torch.float32),torch.tensor(x_test, dtype= torch.float32),torch.tensor(y_test, dtype= torch.float32)


    input_dim = len(x_train[0,:])
    output_dim = len(y_train[0,:])

    net = CoffeeNet_TAV(input_dim, output_dim, args['hidden_units'])
    batch_num = len(x_train[:,0]) // args['batch_size']

    criterion = nn.BCELoss()
    optimizer = Adam(net.parameters(), lr = args['lr'], weight_decay = args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_decay_step'], gamma=args['lr_decay_rate'])

    t0 = time.time(); train_his = []; test_his = []

    for epoch in range(args['epoch']):
        net.eval()
        with torch.no_grad():
            y_dnn_test = net(x_test)
            test_loss = float(criterion(y_dnn_test, y_test))
        net.train()
        train_loss = 0
        for i in range(batch_num):
            x_train_batch = x_train[i * args['batch_size']: (i+1) * args['batch_size'], : ]
            y_train_batch = y_train[i * args['batch_size']: (i+1) * args['batch_size'], : ]
            y_dnn_train_batch = net(x_train_batch)
            train_loss_batch  = criterion(y_dnn_train_batch, y_train_batch)

            optimizer.zero_grad()
            train_loss_batch.backward()
            optimizer.step()
            train_loss += float(train_loss_batch)
        scheduler.step()
        train_loss /= batch_num
        train_his.append(train_loss); test_his.append(test_loss)
        if epoch % 5 == 2:
            print('epoch: {}\t train loss: {:.4e}   test loss: {:.4e}   time cost: {} s   lr:{:.2e}'
                        .format(epoch, train_loss, test_loss, int(time.time()-t0), optimizer.param_groups[0]['lr']))    

            state = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch, 'ini_args':args}
            if test_his[-1] > test_his[-2] and epoch > 2000:
                break
    if need_fig:
        torch.save(state ,f'./model/{model_name}.pth')
        plt.semilogy(np.arange(10, len(train_his)), train_his[10:], label = 'train')
        plt.semilogy(np.arange(10, len(test_his)), test_his[10:], label = 'test')
        plt.legend()
        plt.title('Loss figure')  
        plt.savefig(f'./pic/{model_name}+loss_fig.png') 
    return net, sk_score(y_train, net(x_train)), sk_score(y_test, net(x_test))

