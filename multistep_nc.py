import numpy as np
import pandas as pd
import os
import torch
from hkstorch import *
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn_quantile import RandomForestQuantileRegressor
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

geo = np.load('C:/Users/scott/Desktop/anl-outage-data/data/nc_geolocation_115.npy')[:,:-1]
total_cus = np.load('C:/Users/scott/Desktop/anl-outage-data/data/ncustomer_ncsc.npy')[:-1]
outage_all = np.load('C:/Users/scott/Desktop/anl-outage-data/data/ncoutage_202005-09.npy').T
loc = pd.read_csv('C:/Users/scott/Desktop/anl-outage-data/Argonne Data/NCSC_data/gis_state_nc_duke.csv')['location']

# select 2020-07-31 to 2020-08-10
alpha = 0.1
all_t = 1056
outage = outage_all[:,-(3360+all_t+800):-3360]
data = np.concatenate([geo, total_cus.reshape((-1,1)),outage],axis=1)
df = pd.DataFrame(data)

multistep = 10
num_step = 100
train_t = 300
window_size = 100
res_all = [[] for _ in range(multistep)]
res_train = []
for i in range(num_step):
    print(i)
    ## train model
    start = 3+i*multistep
    end = start+train_t
    train_data = data[:,start:end]
    model = TorchHawkes(obs=train_data)
    if os.path.exists(f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt"):
        model.load_state_dict(torch.load(f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt"))
    else:
        train(model, locs=geo, k=50, niter=100, lr=1., log_interval=10)
        torch.save(model.state_dict(),f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt")
    _, mus, triggs = model()

    ## prediction and residual on training data
    pred = []
    for j in range(window_size):
        pred.append(model._lambda(end-window_size+j).detach().cpu().numpy())
    res_train.append((data[:,end-window_size:end]- np.array(pred).T)/df.iloc[:,2].values.reshape((-1,1)))

    ## prediction and residual on test data
    pred_t = []
    for j in range(multistep):
        pred_t.append(model._lambda(end-start+j).detach().cpu().numpy())
    res = (data[:,end:end+multistep]- np.array(pred_t).T)/df.iloc[:,2].values.reshape((-1,1))
    for j in range(multistep):
        res_all[j].append(res[:,j])
res_train = np.concatenate(res_train, axis=0)

## test procedure
test_step = 50
for i in range(test_step):
    start = 3+num_step*multistep
    end = start+train_t
    train_data = data[:,start:end]
    model = TorchHawkes(obs=train_data)
    if os.path.exists(f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt"):
        model.load_state_dict(torch.load(f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt"))
    else:
        train(model, locs=geo, k=50, niter=100, lr=1., log_interval=10)
        torch.save(model.state_dict(),f"saved_model/nc/hawkes_nc_202005_{start}_{end}_step{i}.pt")
    _, mus, triggs = model()


    pred = []
    for j in range(window_size):
        pred.append(model._lambda(end-window_size+j).detach().cpu().numpy())
    res_now=(data[:,end-window_size:end]- np.array(pred).T)/df.iloc[:,2].values.reshape((-1,1))

    ## prediction
    test_data = data[:,end:end+multistep]
    pred_t = []
    for j in range(multistep):
        pred_t.append(model._lambda(end-start+j).detach().cpu().numpy())
    
    ## Poisson coverage
    coverage_p, coverage_hp, size, lower_p, upper_p = Poisson_cal(test_data, pred_t, alpha)

    ## SPCI
    coverage_cp, coverage_hcp, q_cp, quantiles_5, quantiles_95 = SPCI(test_data, res_train, res_all, res_now, pred_t, alpha)

    ## GraphCP
    coverage_gcp, coverage_hgcp, q_gcp, quantiles_5g, quantiles_95g = Graph_CP(test_data, res_train, res_all, res_now, pred_t, alpha)


# Create the figure and axis
print("Poisson coverage:", np.mean(coverage_p), np.mean(coverage_hp), " and Length:", np.mean(size))
print("Split CP coverage:", np.mean(coverage_cp), np.mean(coverage_hcp), " and Length:", np.mean(q_cp))
print("Graph CP coverage:", np.mean(coverage_gcp), np.mean(coverage_hgcp), " and Length:", np.mean(q_gcp))


## plot the moving average[
T=multistep
fig, ax = plt.subplots(figsize=(12, 6))
# Plot the mean residuals
ax.plot([i for i in range(T)],test_data[j,:], label='Power outage number', color='blue')
ax.plot([i for i in range(T)], [pred_t[i][j] for i in range(T)], label='Prediction', color='red')
# Plot the quantile range as a shaded region
#ax.fill_between([i for i in range(T)],[lower_p[j,i] for i in range(T)], [upper_p[j,i] for i in range(T)], color='blue', alpha=0.2, label='Parametric Prediction Interval')
ax.fill_between([i for i in range(T)],[(pred[i][j]+quantiles_5[j][i]*df.iloc[:,2].values[j]) for i in range(T)],  [(pred[i][j]+quantiles_95[j][i]*df.iloc[:,2].values[j]) for i in range(T)], color='brown', alpha=0.2, label='Split Conformal Prediction Interval')
#ax.fill_between([i for i in range(T)],[(pred[i][j]+quantiles_5g[j][i]*df.iloc[:,2].values[j]) for i in range(T)],  [(pred[i][j]+quantiles_95g[j][i]*df.iloc[:,2].values[j]) for i in range(T)], color='green', alpha=0.2, label='Graph Conformal Prediction Interval')
ax.set_title(f"Prediction Region for County {loc.iloc[j]}")
ax.set_xlabel("Date")
ax.set_ylabel("Outage Number")
ax.legend()

plt.plot(range(T),[np.mean(coverage_p[:,i]) for i in range(T)])
plt.plot(range(T),[np.mean(coverage_cp[:,i]) for i in range(T)])
plt.plot(range(T),[np.mean(coverage_gcp[:,i]) for i in range(T)])


average_outage = [np.mean(df.iloc[j,end+t:end+t+T]) for j in range(df.shape[0])]
s1 = [np.mean(size[j,:]) for j in range(df.shape[0])]
s2 = [np.mean(q_cp[j,:]) for j in range(df.shape[0])]
s3 = [np.mean(q_gcp[j,:]) for j in range(df.shape[0])]
c1=[np.mean(coverage_p[i,:]) for i in range(df.shape[0])]
c2=[np.mean(coverage_cp[i,:]) for i in range(df.shape[0])]
c3=[np.mean(coverage_gcp[i,:]) for i in range(df.shape[0])]
win_large = np.zeros(3)
count = 0
for i in range(df.shape[0]):
    if average_outage[i]>50:
        count += 1
        c=np.array([c1[i],c2[i],c3[i]])
        sz=np.array([s1[i],s2[i],s3[i]])
        win_ind = np.where(c>0.9)[0]
        if len(win_ind) == 0:
            win_ind = np.where(c==np.max(c))[0]
        win_large[np.where(sz==np.min(sz[win_ind]))[0]]+=1

print(win_large/count)

#### draw violin plot for each method coverage
## draw for all the data
data = [c1, c2, c3]
draw_violin(data)