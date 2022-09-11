import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from matplotlib.font_manager import FontProperties  # 导入FontProperties
from base_func import *
from base_class import *

# Load the data and original data (before zscore)
data_x = pd.read_csv('data\data_x.csv', index_col=0)
data_y = pd.read_csv('data\data_y.csv', index_col=0)
data = pd.concat([data_y, data_x],axis=1)
col_x = data_x.columns
data_x = data_x.to_numpy()
data_y = data_y.to_numpy()

original_data = pd.read_csv(r'data\final_results_processed.csv', index_col=0)


# Train DNN args
args = {
    'hidden_units' : [1000,1000,500],
    'lr': 1e-5,
    'batch_size': 16,
    'epoch':1200,
    'lr_decay_step':30,
    "lr_decay_rate":0.95,
    'weight_decay':0.3,
}

# Bagging Method
Net = [];train_score, test_score = 0,0; step = 20; begin_random_num = 254
for seed in np.arange(begin_random_num, begin_random_num + step, 1, dtype = np.int32):
    print(f'start of seed {seed}')
    # Bootstrapping method to divide the dataset
    x_train_set, x_test_set, y_train_set, y_test_set = bootstrap_with_dup(data_x, data_y, seed = seed)
    np.random.seed(seed)
    # Give disturbance on the learning rate 
    args['lr'] = args['lr'] + np.random.randn() * 1e-6
    # Training
    net, dnn_train_sc, dnn_test_sc  = DNN_train(x_train_set, y_train_set, x_test_set, y_test_set, args = args, 
                        model_name = f'DNN_model_20220816_seed={seed}', need_fig= False)
    Net.append(net)
    pred_train = net(torch.tensor(x_train_set, dtype = torch.float32)).detach().numpy()
    pred_test = net(torch.tensor(x_test_set, dtype = torch.float32)).detach().numpy()
    train_score += r2_score(y_train_set, pred_train)
    test_score += r2_score(y_test_set, pred_test)
    print(f'train score:{r2_score(y_train_set, pred_train)}')
    print(f'train score:{r2_score(y_test_set, pred_test)}')
# Record the loss value
train_score = train_score/step; test_score = test_score/step


# Plot the results
pred = sum(net(torch.tensor(data_x, dtype = torch.float32)) for net in Net).detach().numpy()
scatter_plot_TM_CW(data, col_x, pred = pred/step, backend = 'Temperature')

# Save the bagging model
state = {}
for k in range(len(Net)):
    state.update({f"net_{k}": Net[k].state_dict()})
state.update({'ini_args': args})
state.update({'train_score':train_score})
state.update({'test_score': test_score})
torch.save(state, f"./model/Bagging_DNN_model.pth")