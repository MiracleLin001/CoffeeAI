import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import KFold

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

# Using bootstapping method to divide the dataset into train and testing
x_train_set, x_test_set, y_train_set, y_test_set = bootstrap_with_dup(data_x, data_y, seed = 4458)
# Train DNN args
args = {
    'hidden_units' : [1000,1000,500],
    'lr': 1e-5,
    'batch_size': 16,
    'epoch':1500,
    'lr_decay_step':30,
    "lr_decay_rate":0.95,
    'weight_decay':0.4,
}
net, dnn_train_sc, dnn_test_sc  = DNN_train(x_train_set, y_train_set, x_test_set, y_test_set, args = args, model_name = 'DNN_model')

# Plot the results
pred = net(torch.tensor(data_x, dtype = torch.float32)).detach().numpy()
scatter_plot_TM_CW(data, col_x, pred = pred, backend = 'Temperature')

