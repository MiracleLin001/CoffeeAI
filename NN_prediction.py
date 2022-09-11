from base_class import *
from base_func import *
from matplotlib import ticker

plt.rcParams["font.family"] = "Times New Roman"

# Load dataset and apply zscore on it
data_x = pd.read_csv('data\data_x.csv', index_col=0)
data_y = pd.read_csv('data\data_y.csv', index_col=0)

data = pd.concat([data_y, data_x],axis=1)
# We only focus on the six targets
col_x = [ 'Citric acid', 'Quinic acid','Caffeic acid','Caffeine','CGA(Bitterness)', 'CGA(Acidity)']
data_x_slice = pd.read_csv(r'data\TAV_result.csv')[col_x]
_, mean_data_x, std_data_x = zscore(data_x_slice)

# Generate the test data range
mean = np.mean(data_x, axis = 0); basic_data = np.tile(mean, (200,1))
std = np.std(data_x, axis=0)
generate_data_x = []
for i in range(len(mean)):
    tmp_basic_data = basic_data.copy(); tmp_basic_data[:,i] = mean[i] +0.5*std[i]*np.linspace(-1,1,200)
    generate_data_x.append(tmp_basic_data)

data_x = data_x.to_numpy()
data_y = data_y.to_numpy()
# Load the Net/Bagging model
Net_dict = torch.load('model\Bagging_DNN_model.pth')
input_dim = len(data_x[0,:])
output_dim = len(data_y[0,:])
args = {
    'hidden_units' : [1000,1000,500],
    'lr': 1e-5,
    'batch_size': 16,
    'epoch':1500,
    'lr_decay_step':30,
    "lr_decay_rate":0.95,
    'weight_decay':0.4,
}
Net = []
for k in range(len(Net_dict)-3):
    net = CoffeeNet(input_dim, output_dim, args['hidden_units'],limit_interval = [[10, 80],[20, 100]],)
    net.load_state_dict(Net_dict[f'net_{k}']())
    Net.append(net)


# Use random forest as a reference group
forest_reg, rf_train_sc, rf_test_sc = RF(data_x, data_y, method = 'bootstrap')


# Canvas for the Temperature
fig, ax = plt.subplots(len(col_x),2, figsize = (10,20), dpi = 400)
# NN prediction
for k in range(len(col_x)):
    generate_data_y = sum([net(torch.tensor(generate_data_x[k], dtype=torch.float32)) for net in Net])/len(Net)
    generate_data_y = generate_data_y.detach().numpy()
    chem = col_x[k]
    sns.scatterplot(x = generate_data_y[:,1], y = generate_data_x[k][:,k] * std_data_x[k] + mean_data_x[k], ax = ax[k,0], s=6, edgecolors='black')
    ax[k,0].xaxis.set_tick_params(labelbottom=True)
    ax[k,0].xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[k,0].set_title(chem)
# Random forest prediction
for k in range(len(col_x)):
    generate_data_y = forest_reg.predict(generate_data_x[k])
    chem = col_x[k]
    sns.scatterplot(x = generate_data_y[:,1], y = generate_data_x[k][:,k] * std_data_x[k] + mean_data_x[k], ax = ax[k,1], s=6, edgecolors='black')
    ax[k,1].xaxis.set_tick_params(labelbottom=True)
    ax[k,1].xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[k,1].set_title(chem)
ax[-1,0].set_xlabel('Temperature[$^\circ C$]'); ax[-1,1].set_xlabel('Temperature[$^\circ C$]')
fig.supylabel('TAV', fontsize = 16)
plt.tight_layout()
plt.savefig("temperature+nn+rf_all.png")

# Canvas for the Water/Coffee ratio
fig, ax = plt.subplots(len(col_x),2, figsize = (10,20), dpi = 400)

# NN prediction
for k in range(len(col_x)):
    generate_data_y = sum([net(torch.tensor(generate_data_x[k], dtype=torch.float32)) for net in Net])/len(Net)
    generate_data_y = generate_data_y.detach().numpy()
    chem = col_x[k]
    sns.scatterplot(x = generate_data_y[:,0], y = generate_data_x[k][:,k] * std_data_x[k] + mean_data_x[k], ax = ax[k,0], s=6, edgecolors='black')
    ax[k,0].xaxis.set_tick_params(labelbottom=True)
    if chem in ['Caffeic acid', 'Caffeine', 'CGA(Bitterness)', 'CGA(Acidity)']:
        ax[k,0].set_yscale('log')
    ax[k,0].xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[k,0].set_title(chem)

# Random forest prediction
for k in range(len(col_x)):
    generate_data_y = forest_reg.predict(generate_data_x[k])
    chem = col_x[k]
    sns.scatterplot(x = generate_data_y[:,0], y = generate_data_x[k][:,k] * std_data_x[k] + mean_data_x[k], ax = ax[k,1], s=6, edgecolors='black')
    ax[k,1].xaxis.set_tick_params(labelbottom=True)
    if chem in ['Caffeic acid', 'Caffeine', 'CGA(Bitterness)', 'CGA(Acidity)']:
        ax[k,1].set_yscale('log')
    ax[k,1].xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax[k,1].set_title(chem)
ax[-1,0].set_xlabel('Water/Coffee Ratio'); ax[-1,1].set_xlabel('Water/Coffee Ratio')
fig.supylabel('TAV', fontsize = 16)
plt.tight_layout()
plt.savefig("cw+nn+rf_all.png")