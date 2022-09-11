import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.ensemble import RandomForestRegressor

from matplotlib.font_manager import FontProperties  # 导入FontProperties
from base_func import *
from base_class import *

# Load the dataset
data_x = pd.read_csv('data\data_x.csv', index_col=0)
data_y = pd.read_csv('data\data_y.csv', index_col=0)
data = pd.concat([data_y, data_x],axis=1)
col_x = data_x.columns
data_x = data_x.to_numpy()
data_y = data_y.to_numpy()

# Implement the Random Forest Module and Plot the Result
forest_reg, rf_train_sc, rf_test_sc = RF(data_x, data_y, method = 'bootstrap')
pred = forest_reg.predict(data_x)
scatter_plot_TM_CW(data, col_x, pred = pred, backend = 'Temperature')
scatter_plot_TM_CW(data, col_x, pred = pred, backend = 'Coffe/WaterRatio')
