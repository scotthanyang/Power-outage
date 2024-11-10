# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:00:23 2024

@author: scott
"""

import numpy as np
import pandas as pd
import os
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
all_t = 1056
outage = outage_all[:,-(3360+all_t+800):-3360]
data = np.concatenate([geo, total_cus.reshape((-1,1)),outage],axis=1)
df = pd.DataFrame(data)

start = 3
end = 303
data = data[:,start:end]
model = TorchHawkes(obs=data)
if os.path.exists(f"hawkes_nc_202005_{start}_{end}.pt"):
    model.load_state_dict(torch.load(f"hawkes_nc_202005_{start}_{end}.pt"))
else:
    train(model, locs=geo, k=50, niter=500, lr=1., log_interval=10)
    torch.save(model.state_dict(),f"hawkes_nc_202005_{start}_{end}.pt")
_, mus, triggs = model()


## calibration data
t = 800
T = 100
fit_pred = mus.detach().cpu().numpy() + triggs.detach().cpu().numpy()
pred_t = []
for i in range(t+T):
    pred_t.append(model._lambda(end-start+i).detach().cpu().numpy())
res = (df.iloc[:,end:end+t+T].values- np.array(pred_t).T)/df.iloc[:,2].values.reshape((-1,1))

pred = []
for i in range(T):
    pred.append(model._lambda(end-start+t+i).detach().cpu().numpy())
    
    
## parametric method (Poisson)
test = df.iloc[:,(end+t+1):(end+t+T+1)]
alpha = 0.1
z = norm.ppf(1 - alpha / 2)
hist_lower_p = fit_pred - z * np.sqrt(fit_pred)
hist_upper_p = fit_pred + z * np.sqrt(fit_pred)
lower_p = np.zeros((data.shape[0],T))
upper_p = np.zeros((data.shape[0],T))

coverage_p = np.zeros((data.shape[0],T))
coverage_hp = np.zeros((data.shape[0],T))
for i in range(T):
    lower_p[:,i] = pred[i] - z * np.sqrt(pred[i])
    upper_p[:,i] = pred[i] + z * np.sqrt(pred[i])
    for j in range(data.shape[0]):
        if lower_p[j,i] <= test.iloc[j,i] and test.iloc[j,i] <= upper_p[j,i]:
            coverage_p[j,i] += 1
            if test.iloc[j,i] >= 1:
                coverage_hp[j,i] += 1 
size = upper_p-lower_p

## quantile regression
window_size =100
coverage_cp = np.zeros((data.shape[0],T))
coverage_hcp = np.zeros((data.shape[0],T))
quantiles_5, quantiles_95 = np.zeros((data.shape[0],T)), np.zeros((data.shape[0],T))
n_rows = t - window_size + 1

for j in range(data.shape[0]):
    if j % 100 == 0:
        print(j)
    matrix = np.array([res[j,i:i + window_size] for i in range(n_rows)])
    qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05,0.95])
    qrf.fit(matrix[:,:-1], matrix[:,-1])
    for i in range(T):
        q5,  q95 = qrf.predict(res[j,t+i-window_size+1:t+i].reshape((1,-1)))
        quantiles_5[j,i] = q5[0]-1e-5
        quantiles_95[j,i] = q95[0]+1e-5
        if pred[i][j] + quantiles_5[j,i]*df.iloc[:,2].values[j] <= test.iloc[j,i] and pred[i][j] + quantiles_95[j,i]*df.iloc[:,2].values[j]>= test.iloc[j,i]:
            coverage_cp[j,i] += 1
            if test.iloc[j,i] >= 1:
                coverage_hcp[j,i] += 1 
            
q_cp = (quantiles_95-quantiles_5)*df.iloc[:,2].values.reshape((-1,1))

## graph conformal prediction method
# Create a NearestNeighbors model
knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
knn.fit(df.iloc[:,[0,1]])
distances, indices = knn.kneighbors(df.iloc[:,[0,1]])
combined_res = res[indices.flatten()].reshape(data.shape[0], -1)

qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05,0.95])
coverage_gcp = np.zeros((data.shape[0],T))
coverage_hgcp = np.zeros((data.shape[0],T))
quantiles_5g, quantiles_95g = np.zeros((data.shape[0],T)), np.zeros((data.shape[0],T))

for j in range(data.shape[0]):
    if j % 100 == 0:
        print(j)
    matrix = np.array([res[indices[j],i:i + window_size] for i in range(n_rows)]).reshape((-1,window_size))
    qrf.fit(matrix[:,:-1], matrix[:,-1])
    for i in range(T):
        q5,  q95 = qrf.predict(res[j,t+i-window_size+1:t+i].reshape((1,-1)))
        quantiles_5g[j,i]=q5[0]-1e-5
        quantiles_95g[j,i]=q95[0]+1e-5
        if pred[i][j] + quantiles_5g[j,i]*df.iloc[:,2].values[j] <= test.iloc[j,i] and pred[i][j] + quantiles_95g[j,i]*df.iloc[:,2].values[j] >= test.iloc[j,i]:
            coverage_gcp[j,i] += 1
            if test.iloc[j,i] >= 1:
                coverage_hgcp[j,i] += 1 

# Calculate the length of the prediction interval
q_gcp = (quantiles_95g-quantiles_5g)*df.iloc[:,2].values.reshape((-1,1))


# Create the figure and axis
print("Poisson coverage:", np.mean(coverage_p), np.mean(coverage_hp), " and Length:", np.mean(size))
print("Split CP coverage:", np.mean(coverage_cp), np.mean(coverage_hcp), " and Length:", np.mean(q_cp))
print("Graph CP coverage:", np.mean(coverage_gcp), np.mean(coverage_hgcp), " and Length:", np.mean(q_gcp))


## plot the moving average
fig, ax = plt.subplots(figsize=(12, 6))
# Plot the mean residuals
ax.plot([i for i in range(T)],df.iloc[j,end+t:end+t+T], label='Power outage number', color='blue')
# ax.plot(date[end+t:], fit_pred[i], label='Prediction', color='red')
# Plot the quantile range as a shaded region
ax.fill_between([i for i in range(T)],[lower_p[j,i] for i in range(T)], [upper_p[j,i] for i in range(T)], color='blue', alpha=0.2, label='Parametric Prediction Interval')
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