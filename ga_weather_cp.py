# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 12:04:37 2024

@author: scott
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hkstorch import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

config = {
"GA Oct 2018": {
        # outage configurations
        "outage_path":    "gaoutage_201809-11.npy",
        "outage_geo":     "ga_geolocation_665.npy",
        "outage_startt":  "2018-09-13 00:00:00",
        "outage_endt":    "2018-11-30 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_gaweather-20180911",
        "weather_geo":    "ga_geolocation_665.npy",
        "weather_startt": "2018-09-12 00:00:00",
        "weather_endt":   "2018-11-30 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        "_startt":        "2018-09-13 00:00:00",
        "_endt":          "2018-11-30 00:00:00"
    }}


loc_name = pd.read_csv('C:/Users/scott/Desktop/anl-outage-data/data/location_with_totalcustomers/gis_state_ga_gp.csv').iloc[:,[0]].values
obs_outage, obs_weather, locs, _ = dataloader(config["GA Oct 2018"], outageN=2, weatherN=2, isproj=False)
total_customer = np.load('C:/Users/scott/Desktop/anl-outage-data/data/ncustomer_ga.npy')[:obs_outage.shape[0]]
s = obs_outage.shape[0]
c_ind = [i for i in range(len(total_customer)) if total_customer[i]>=50]
obs_outage = obs_outage[c_ind]
obs_weather = obs_weather[c_ind]
locs = locs[c_ind]
loc_name = loc_name[c_ind]
total_customer = total_customer[c_ind]
loc_name = loc_name[:locs.shape[0],:]
loc_ids = locs[:,2]
total_customer = total_customer[:obs_outage.shape[0]]

## train the model with training data
startt     = arrow.get(config["GA Oct 2018"]["_startt"], "YYYY-MM-DD HH:mm:ss")
endt       = arrow.get(config["GA Oct 2018"]["_endt"], "YYYY-MM-DD HH:mm:ss")
train_start = arrow.get('2018-09-13 00:00:00', "YYYY-MM-DD HH:mm:ss")
train_end = arrow.get('2018-10-01 00:00:00', "YYYY-MM-DD HH:mm:ss")
startind_train   = int((train_start.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
endind_train     = int((train_end.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
train_outage = obs_outage[:,startind_train:endind_train+1] 
train_weather = obs_weather[:,startind_train:endind_train+1,:] 


print("Start Training")
model = TorchHawkesNNCovariates(d=24, obs=train_outage, covariates=train_weather)
if os.path.exists("hawkes_ga_20180913_1001.pt"):
    model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    model.load_state_dict(torch.load("hawkes_ga_20180913_1001.pt"))
else:
    train(model, locs=locs[:,:2], k=50, niter=300, lr=1., log_interval=10)
    torch.save(model.state_dict(),"hawkes_ga_20180913_1001.pt")
    model = TorchHawkesNNCovariates(d=24, obs=obs_outage, covariates=obs_weather)
    model.load_state_dict(torch.load("hawkes_ga_20180913_1001.pt"))
print("Training Completed")

## evaluation
_, mus, lams = model()
lams = lams.detach().cpu().numpy()
mus  = mus.detach().cpu().numpy()
lams = lams + mus

## plot prediction
T = obs_outage.shape[1]
dates = [startt.shift(seconds=int((endt - startt).total_seconds() * i / T)).datetime for i in range(T)]
plt.figure(figsize=(12, 6))
plt.plot(dates, np.average(obs_outage, axis=0), label='True outage number', color='blue')
plt.plot(dates, np.average(lams, axis=0), label='Poisson model prediction', color='red')
num_dates_to_show = 5
indices = np.linspace(0, T-1, num_dates_to_show, dtype=int)
plt.xticks([dates[i] for i in indices], [dates[i].strftime('%Y-%m-%d') for i in indices])
plt.xlabel('Date')
plt.ylabel('Outage Number')
plt.title('Prediction for GA')
plt.legend()
plt.tight_layout()
plt.show()

test_start = arrow.get('2018-10-12 00:00:00', "YYYY-MM-DD HH:mm:ss")
test_end = arrow.get('2018-10-20 00:00:00', "YYYY-MM-DD HH:mm:ss")
startind_test   = int((test_start.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
endind_test     = int((test_end.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
test_data = obs_outage[:,startind_test:endind_test+1] 
test_pred = lams[:,startind_test:endind_test+1] 


## poisson coverage
print("Poisson start")
alpha = 0.1
coverage_p, coverage_hp, size, lower_p, upper_p = Poisson_cal(test_data, test_pred, alpha)
print("Poisson Completed")
print("Poisson coverage:", np.mean(coverage_p), np.mean(coverage_hp), " and Length:", np.mean(size))


# calibration set
window_size = 50
cal_start = arrow.get('2018-10-05 00:00:00', "YYYY-MM-DD HH:mm:ss")
cal_end = arrow.get('2018-10-12 00:00:00', "YYYY-MM-DD HH:mm:ss")
startind_cal   = int((cal_start.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
endind_cal    = int((cal_end.timestamp() - startt.timestamp()) / (endt.timestamp()-startt.timestamp())*(obs_outage.shape[1]-1))
cal_data = obs_outage[:,startind_cal:endind_cal+1] 
cal_pred = lams[:,startind_cal:endind_cal+1] 
res_cal = (cal_data - cal_pred)/total_customer.reshape((-1,1))
res_all = (obs_outage[:,(startind_test-window_size):endind_test] - lams[:,(startind_test-window_size):endind_test])/total_customer.reshape((-1,1))

## SPCI
print("SPCI start")
coverage_cp, coverage_hcp, q_cp, quantiles_5, quantiles_95 = SPCI(test_data, res_cal, res_all, 
                                                                  test_pred, total_customer, alpha=0.1, window_size=window_size)
print("SPCI complete")

## Graph CP
print("Graph CP start")
coverage_gcp, coverage_ghcp, q_gcp, quantiles_g5, quantiles_g95 = Graph_CP(test_data, locs[:,[0,1]], 
                    res_cal, res_all, test_pred, total_customer, 0.1, dep=5, window_size=window_size, K=5)
print("Graph CP end")

for j in range(test_data.shape[0]):
     for i in range(test_data.shape[1]):
         quantiles_5[j,i] = np.max([quantiles_5[j,i],-test_pred[j,i]/total_customer[j]])
         quantiles_g5[j,i] = np.max([quantiles_g5[j,i],-test_pred[j,i]/total_customer[j]])
         quantiles_95[j,i] = np.max([quantiles_95[j,i],-test_pred[j,i]/total_customer[j]])
         quantiles_g95[j,i] = np.max([quantiles_g95[j,i],-test_pred[j,i]/total_customer[j]])
         q_cp[j,i] = (quantiles_95-quantiles_5)[j,i]*total_customer[j]
         q_gcp[j,i] = (quantiles_g95-quantiles_g5)[j,i]*total_customer[j]

             
print("Poisson coverage:", np.mean(coverage_p), np.mean(coverage_hp), " and Length:", np.mean(size))
print("Split CP coverage:", np.mean(coverage_cp), np.mean(coverage_hcp), " and Length:", np.mean(q_cp))
print("Graph CP coverage:", np.mean(coverage_gcp), np.mean(coverage_ghcp), " and Length:", np.mean(q_gcp))


# visualization
# ga  j=[5,7,110,150]
plt_start = arrow.get('2018-10-12 00:00:00', "YYYY-MM-DD HH:mm:ss")
plt_end = arrow.get('2018-10-20 00:00:00', "YYYY-MM-DD HH:mm:ss")
startind_plt = int((plt_start.timestamp() - test_start.timestamp()) / (test_end.timestamp()-test_start.timestamp())*(test_data.shape[1]-1))
endind_plt = int((plt_end.timestamp() - test_start.timestamp()) / (test_end.timestamp()-test_start.timestamp())*(test_data.shape[1]-1))
plt_ind = [i for i in range(startind_plt,endind_plt)]
plt_data = test_data[:,startind_plt:endind_plt]
T = plt_data.shape[1]
dates = [test_start.shift(seconds=int((plt_end - plt_start).total_seconds() * i / T)).datetime for i in range(T)]
fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
# Plot the mean residuals
# Plot for Parametric Prediction Interval
axs[0].plot(dates,plt_data[j,:], label='Power outage number', color='blue')
axs[0].fill_between(dates, [lower_p[j,i] for i in plt_ind], [upper_p[j,i] for i in plt_ind], color='red', alpha=0.2, label='Poisson')
axs[0].set_ylabel("Outage Number")
axs[0].legend()
axs[0].set_title(f"Prediction Region for Zipcode {loc_name[j][0]}")

# Plot for Split Conformal Prediction Interval
axs[1].plot(dates,plt_data[j,:], label='Power outage number', color='blue')
axs[1].fill_between(dates,[(test_pred[j,i]+quantiles_5[j,i]*total_customer[j]) for i in plt_ind],  [(test_pred[j,i]+quantiles_95[j,i]*total_customer[j]) for i in plt_ind], color='brown', alpha=0.2, label='SPCI')
axs[1].set_ylabel("Outage Number")
axs[1].legend()

# Plot for Graph Conformal Prediction Interval
axs[2].plot(dates,plt_data[j,:], label='Power outage number', color='blue')
axs[2].fill_between(dates,[(test_pred[j,i]+quantiles_g5[j,i]*total_customer[j]) for i in plt_ind],  [(test_pred[j,i]+quantiles_g95[j,i]*total_customer[j]) for i in plt_ind], color='green', alpha=0.2, label='Graph CP')
axs[2].set_ylabel("Outage Number")
axs[2].set_xlabel("Date")
axs[2].legend()


## compare win rate
s1 = [np.mean(size[j,:]) for j in range(obs_outage.shape[0])]
s2 = [np.mean(q_cp[j,:]) for j in range(obs_outage.shape[0])]
s3 = [np.mean(q_gcp[j,:]) for j in range(obs_outage.shape[0])]
c1=[np.mean(coverage_p[i,:]) for i in range(obs_outage.shape[0])]
c2=[np.mean(coverage_cp[i,:]) for i in range(obs_outage.shape[0])]
c3=[np.mean(coverage_gcp[i,:]) for i in range(obs_outage.shape[0])]
win_large = np.zeros(3)
count = 0
for i in range(obs_outage.shape[0]):
    if np.average(obs_outage,axis=1)[i]>50:
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
