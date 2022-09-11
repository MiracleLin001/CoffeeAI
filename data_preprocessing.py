import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import KFold

from matplotlib.font_manager import FontProperties  # FontProperties
from base_func import *


font = FontProperties(fname="SimHei.ttf", size=14)  # Set the Font

data = pd.read_csv('data/TAV_result.csv')

# Delete the deviation value
data = data.loc[data['Cinnamic acid'] < 0.03,:]
data = data.loc[data['Quinic acid'] > 0.4,:]
data = data.loc[data['Fructose'] < 0.005,:]
data = data.loc[data['Glucose'] < 0.0004,:]
data = data.loc[data['Ferulic acid'] < 0.08,:]
data = data.loc[data['Sucrose'] < 0.15,:]
data = data.loc[data['Lactose'] < 0.6,:]
data = data.loc[data['Maltose'] < 0.0006,:]
data = data.loc[data['Caffeine'] < 3000,:]

data_y = data[['Coffee/WaterRatio','Temperature']]
data_x = data.drop(['Coffee/WaterRatio','Temperature'], axis=1)
data_x, mean_data_x, std_data_x = zscore(data_x) # Use Zscore to standardize the data
data_x.to_csv('./data/data_x.csv'); data_y.to_csv('./data/data_y.csv'); data.iloc[:,2:] = data_x.copy()
np.savez('./data/mean_std.npz', mean = mean_data_x, std = std_data_x) 

# Overview of the dataset
sns.set_theme(style="darkgrid")
col_x = data_x.columns
col_y = data_y.columns
data.to_csv('data/final_results_processed.csv') # save data
scatter_plot_TM_CW(data, col_x, backend = 'Temperature')

