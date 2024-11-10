# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 17:04:30 2024

@author: hanyang.jiang
"""

import torch
import torch.optim as optim
import random
import arrow
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.collections as mcoll
import matplotlib.path as mpath
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from scipy.stats import norm, poisson
from sklearn_quantile import RandomForestQuantileRegressor
from sklearn.neighbors import NearestNeighbors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

concise_new_feat_list = [
    "001", "002", "003", "004", "005", "006", "007", "008", "028", "029", 
    "030", "031", "032", "033", "034", "035", "038", "039", "040", "041", 
    "055", "056", "057", "059", "061", "064", "065", "066", "068", "071", 
    "072", "073", "094", "117", "118"
]
concise_old_feat_list = [
    "001", "002", "003", "004", "005", "006", "007", "008", "026", "027", 
    "028", "029", "030", "031", "032", "033", "036", "037", "038", "039", 
    "043", "044", "045", "047", "049", "052", "053", "054", "056", "059", 
    "060", "061", "079", "101", "102"
]

config = {
"NCSC May 2020": {
        # outage configurations
        "outage_path":    "ncoutage_202005-09.npy",
        "outage_geo":     "nc_geolocation_115.npy",
        "outage_startt":  "2020-05-01 00:00:00",
        "outage_endt":    "2020-09-14 23:45:00",
        "outage_freq":    15 * 60,                 # seconds per recording
        # weather configuration
        "weather_path":   "conv_ncscweather-202005",
        "weather_geo":    'nc_weathergeolocations.npy',
        "weather_startt": "2020-05-01 00:00:00",
        "weather_endt":   "2020-09-14 23:00:00",
        "weather_freq":   60 * 60,                 # seconds per recording
        "feat_list":      concise_new_feat_list,
        # time window
        # Tropical Storm Arthur
        "_startt":        "2020-05-15 00:00:00",
        "_endt":          "2020-06-01 00:00:00"
    }}

## parametric method (Poisson)
# Poisson_cal(test_data, pred_test, alpha)
def Poisson_cal(data, pred, alpha):
    z = norm.ppf(1 - alpha / 2)
    T = pred.shape[1]
    lower_p = np.zeros((data.shape[0],T))
    upper_p = np.zeros((data.shape[0],T))

    coverage_p = np.zeros((data.shape[0],T))
    coverage_hp = np.zeros((data.shape[0],T))
    for i in range(T):
        lower_p[:,i] = pred[:,i] - z * np.sqrt(pred[:,i])
        upper_p[:,i] = pred[:,i] + z * np.sqrt(pred[:,i])
        for j in range(data.shape[0]):
            if lower_p[j,i] <= data[j,i] and data[j,i] <= upper_p[j,i]:
                coverage_p[j,i] += 1
                if data[j,i] >= 1:
                    coverage_hp[j,i] += 1 
    size = upper_p-lower_p
    return coverage_p, coverage_hp, size, lower_p, upper_p

def SPCI(data, res_cal, res_all, pred_t, total_customer, alpha, dep=5, window_size=100):
    coverage_cp = np.zeros((data.shape[0],data.shape[1]))
    coverage_hcp = np.zeros((data.shape[0],data.shape[1]))
    quantiles_5, quantiles_95 = np.zeros((data.shape[0],data.shape[1])), np.zeros((data.shape[0],data.shape[1]))
    n_rows = res_cal.shape[1] - window_size - 1
    for j in range(data.shape[0]):
        if j % 50 == 0:
            print(j)
        matrix = np.array([res_cal[j,i:i + window_size] for i in range(n_rows)])
        qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05,0.95])
        qrf.fit(matrix[:,:-1], matrix[:,-1])
        for i in range(data.shape[1]):
            q5,  q95 = qrf.predict(res_all[j,i:(i+window_size-1)].reshape((1,-1)))
            quantiles_5[j,i] = q5[0]-1e-5
            quantiles_95[j,i] = q95[0]+1e-5
            if pred_t[j,i] + quantiles_5[j,i]*total_customer[j] <= data[j,i] and pred_t[j,i] + quantiles_95[j,i]*total_customer[j]>= data[j,i]:
                coverage_cp[j,i] += 1
                if data[j,i] >= 1:
                    coverage_hcp[j,i] += 1 
            
    q_cp = (quantiles_95-quantiles_5)*total_customer.reshape((-1,1))
    return coverage_cp, coverage_hcp, q_cp, quantiles_5, quantiles_95

def Graph_CP(data, locs, res_cal, res_all, pred_t, total_customer, alpha, dep=5, window_size=100, K=5):
    knn = NearestNeighbors(n_neighbors=K, algorithm='ball_tree')
    knn.fit(locs)
    distances, indices = knn.kneighbors(locs)
    combined_res = res_cal[indices.flatten()].reshape(data.shape[0], -1)
    coverage_cp = np.zeros((data.shape[0],data.shape[1]))
    coverage_hcp = np.zeros((data.shape[0],data.shape[1]))
    quantiles_5, quantiles_95 = np.zeros((data.shape[0],data.shape[1])), np.zeros((data.shape[0],data.shape[1]))
    n_rows = res_cal.shape[1] - window_size - 1
    for j in range(data.shape[0]):
        if j % 50 == 0:
            print(j)
        #matrix = np.array([np.concatenate([res_cal[indices[j],i:i + window_size -1].reshape((-1)),
         #     [res_cal[j,i+window_size]]],axis=0) for i in range(n_rows)])
        matrix = np.array([res_cal[indices[j],i:i + window_size] for i in range(n_rows)]).reshape((-1,window_size))
        qrf = RandomForestQuantileRegressor(n_estimators=30, max_depth=5, q=[0.05,0.95])
        qrf.fit(matrix[:,:-1], matrix[:,-1])
        for i in range(data.shape[1]):
            #q5,  q95 = qrf.predict(res_all[indices[j],i:(i+window_size-1)].reshape((1,-1)))
            q5,  q95 = qrf.predict(res_all[j,i:(i+window_size-1)].reshape((1,-1)))
            quantiles_5[j,i] = q5[0]-1e-5
            quantiles_95[j,i] = q95[0]+1e-5
            if pred_t[j,i] + quantiles_5[j,i]*total_customer[j] <= data[j,i] and pred_t[j,i] + quantiles_95[j,i]*total_customer[j]>= data[j,i]:
                coverage_cp[j,i] += 1
                if data[j,i] >= 1:
                    coverage_hcp[j,i] += 1 
            
    q_cp = (quantiles_95-quantiles_5)*total_customer.reshape((-1,1))
    return coverage_cp, coverage_hcp, q_cp, quantiles_5, quantiles_95



def percentile_excluding_zeros(data, percentile):
    # Create a masked array that excludes zeros
    masked_data = np.ma.masked_equal(data, 0)
    
    # Calculate the percentile for each row
    result = np.ma.apply_along_axis(
        lambda x: np.percentile(x.compressed(), percentile) if x.compressed().size > 0 else 0,
        axis=1, arr=masked_data
    )
    
    return result


def draw_violin(data):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    violin_parts = ax.violinplot(data,showmeans=True, showmedians=False, showextrema=False)

    # Manually set the color for each violin plot
    for i, pc in enumerate(violin_parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')  # Optional: Set edge color
        pc.set_alpha(0.7)  # Optional: Set transparency

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Poisson', 'SPCI', 'Graph CP'])
    ax.set_xlabel('Method')
    ax.set_ylabel('Coverage')
    plt.show()

def train(model, locs, k=100, niter=1000, lr=1e-1, log_interval=50):
    """training procedure for one epoch"""
    # coordinates of K locations
    model.to(device)
    coords    = locs
    # define model clipper to enforce inequality constraints
    clipper1  = NonNegativeClipper()
    clipper2  = ProximityClipper(coords, k=k)
    # NOTE: gradient for loss is expected to be None, 
    #       since it is not leaf node. (it's root node)
    logliks = []
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    for _iter in range(niter):
        try:
            model.train()
            optimizer.zero_grad()           # init optimizer (set gradient to be zero)
            loglik, _, _ = model()
            # objective function
            loss         = - loglik
            loss.backward()                 # gradient descent
            optimizer.step()                # update optimizer
            model.apply(clipper1)
            model.apply(clipper2)
            # log training output
            logliks.append(loglik.item())
            if _iter % log_interval == 0 and _iter != 0:
                print("[%s] Train batch: %d\tLoglik: %.3e" % (arrow.now(), 
                    _iter / log_interval, 
                    sum(logliks) / log_interval))
                logliks = []
        except KeyboardInterrupt:
            break


class NonNegativeClipper(object):
    """
    References:
    https://discuss.pytorch.org/t/restrict-range-of-variable-during-gradient-descent/1933
    https://discuss.pytorch.org/t/set-constraints-on-parameters-or-layers/23620/3
    """

    def __init__(self):
        pass

    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Alpha'):
            Alpha = module.Alpha.data
            module.Alpha.data = torch.clamp(Alpha, min=0.)
        if hasattr(module, 'Beta'):
            Beta  = module.Beta.data
            module.Beta.data  = torch.clamp(Beta, min=0.)
        # TorchHawkesNNCovariates
        if hasattr(module, 'Gamma'):
            Gamma  = module.Gamma.data
            module.Gamma.data = torch.clamp(Gamma, min=0.)
        if hasattr(module, 'Omega'):
            Omega  = module.Omega.data
            module.Omega.data = torch.clamp(Omega, min=0.)



class ProximityClipper(object):
    """
    """

    def __init__(self, coords, k):
        """
        Args:
        - coords: a list of coordinates for K locations [ K, 2 ]
        """
        distmat      = euclidean_distances(coords)        # [K, K]
        proxmat      = self._k_nearest_mask(distmat, k=k) # [K, K]
        self.proxmat = torch.FloatTensor(proxmat).to(device)         # [K, K]
        
    def __call__(self, module):
        """enforce non-negative constraints"""
        # TorchHawkes
        if hasattr(module, 'Alpha'):
            alpha = module.Alpha.data
            module.Alpha.data = alpha * self.proxmat
    
    @staticmethod
    def _k_nearest_mask(distmat, k):
        """binary matrix indicating the k nearest locations in each row"""
        
        # return a binary (0, 1) vector where value 1 indicates whether the entry is 
        # its k nearest neighbors. 
        def _k_nearest_neighbors(arr, k=k):
            idx  = arr.argsort()[:k]  # [K]
            barr = np.zeros(len(arr)) # [K]
            barr[idx] = 1         
            return barr

        # calculate k nearest mask where the k nearest neighbors are indicated by 1 in each row 
        mask = np.apply_along_axis(_k_nearest_neighbors, 1, distmat) # [K, K]
        return mask



class TorchHawkes(torch.nn.Module):
    """
    PyTorch Module for Hawkes Processes
    """

    def __init__(self, obs):
        """
        Denote the number of time units as N, the number of locations as K

        Args:
        - obs:    event observations    [ N, K ]
        """
        torch.nn.Module.__init__(self)
        # data
        self.obs    = torch.Tensor(obs).to(device) # [ K, N ]
        # configurations
        self.K, self.N = obs.shape
        # parameters
        self.Mu0   = self.obs.mean(1) / 10 + 1e-2                                      # [ K ]
        self.Beta  = torch.nn.Parameter(torch.Tensor(self.K).uniform_(1, 3).to(device))      # [ K ]
        self.Alpha = torch.nn.Parameter(torch.Tensor(self.K, self.K).uniform_(0, .01).to(device)) # [ K, K ]
    
    def _mu(self, _t):
        """
        Background rate at time `t`
        """
        return self.Mu0

    def _lambda(self, _t):
        """
        Conditional intensity function at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        if _t > 0:
            # current time and the past 
            t      = torch.ones(np.min([_t,self.obs.shape[1]]), dtype=torch.int32).to(device) * _t      # [ t ]
            tp     = torch.arange(np.min([_t,self.obs.shape[1]])).to(device)                            # [ t ]
            # self-exciting effect
            kernel = self.__exp_kernel(self.Beta, t, tp, self.K) # [ K, t ]
            Nt     = self.obs[:, :_t].clone()                    # [ K, t ]
            lam    = torch.mm(self.Alpha, Nt * kernel).sum(1)    # [ K ]
            lam    = torch.nn.functional.softplus(lam)           # [ K ]
        else:
            lam    = torch.zeros(self.K).to(device)
        return lam
        
    def _log_likelihood(self):
        """
        Log likelihood function at time `T`
        
        Args:
        - tau:    index of start time, e.g., 0, 1, ..., N (integer)
        - t:      index of end time, e.g., 0, 1, ..., N (integer)

        Return:
        - loglik: a vector of log likelihood value at location k = 0, 1, ..., K [ K ]
        - lams:   a list of historical conditional intensity values at time t = tau, ..., t
        """
        # lambda values from 0 to N
        lams0    = [ self._mu(t) for t in np.arange(self.N) ]     # ( N, [ K ] )
        lams1    = [ self._lambda(t) for t in np.arange(self.N) ] # ( N, [ K ] )
        lams0    = torch.stack(lams0, dim=1)                      # [ K, N ]
        lams1    = torch.stack(lams1, dim=1)                      # [ K, N ]
        Nloglams = self.obs * torch.log(lams0 + lams1 + 1e-5)     # [ K, N ]
        # log-likelihood function
        loglik   = (Nloglams - lams0 - lams1).sum()
        return loglik, lams0, lams1

    def forward(self):
        """
        customized forward function
        """
        # calculate data log-likelihood
        return self._log_likelihood()

    @staticmethod
    def __exp_kernel(Beta, t, tp, K):
        """
        Args:
        - Beta:  decaying rate [ K ]
        - t, tp: time index    [ t ]
        """
        delta_t = t - tp                              # [ t ]
        delta_t = delta_t.unsqueeze(0).repeat([K, 1]) # [ K, t ]
        Beta    = Beta.unsqueeze(1)                   # [ K, 1 ]
        return Beta * torch.exp(- delta_t * Beta)
    
class TorchHawkesNNCovariates(TorchHawkes):
    """
    PyTorch Module for Hawkes Processes with Externel Observation
    """

    def __init__(self, d, obs, covariates):
        """
        Denote the number of time units as N, the number of locations as K, and 
        the number of externel features as M.

        Args:
        - d:      memory depth
        - obs:    event observations    [ N, K ]
        - extobs: externel observations [ N, K, M ]
        """
        TorchHawkes.__init__(self, obs)
        # configuration
        self.d       = d                        # d: memory depth
        K, N, self.M = covariates.shape         # M: number of covariates
        assert N == self.N and K == self.K, \
            "invalid dimension (%d, %d, %d) of covariates, where N is not %d or K is not %d." % \
            (N, K, self.M, self.N, self.K)
        # data
        self.covs  = torch.Tensor(covariates).to(device)   # [ K, N, M ]
        # parameters
        self.Gamma = torch.nn.Parameter(torch.Tensor(self.K).uniform_(0, .01).to(device)) # [ K ]
        self.Omega = torch.nn.Parameter(torch.Tensor(self.M).uniform_(0, .5).to(device))  # [ M ]
        # network
        self.nn    = torch.nn.Sequential(
            torch.nn.Linear(self.M, 200),       # [ M, 20 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(200, 200),          # [ 20, 1 ]
            torch.nn.Softplus(), 
            torch.nn.Linear(200, 1),            # [ 20, 1 ]
            torch.nn.Softplus()).to(device)
        self.hmu   = 0

    def _mu(self, _t):
        """
        Background rate at time `t`

        Args:
        - _t:  index of time, e.g., 0, 1, ..., N (integer)
        Return:
        - lam: a vector of lambda value at time t and location k = 0, 1, ..., K [ K ]
        """
        # get covariates in the past d time slots
        if _t < self.d:
            X     = self.covs[:, :_t, :].clone()                            # [ K, t, M ]
            X_pad = self.covs[:, :1, :].clone().repeat([1, self.d - _t, 1]) # [ K, d - t, M ]
            X     = torch.cat([X_pad, X], dim=1)                            # [ K, d, M ]
        else:
            X     = self.covs[:, _t-self.d:_t, :].clone()                   # [ K, d, M ]
        # convolution with an exponential decaying kernel
        conv_X    = self.conv_exp_decay_kernel(X)                           # [ K, M ]
        # calculate base intensity
        mu  = self.nn(conv_X)                                               # [ K, 1 ]
        mu  = self.Gamma * mu.clone().squeeze_()                            # [ K ]
        return mu

    def conv_exp_decay_kernel(self, X):
        """
        Compute convolution of covariates with an exponential decaying kernel.

        Arg:
        - X: observed covariates in the past d time slots [ K, d, M ]
        """
        # exponential decaying kernel
        delta_t = torch.arange(self.d).to(device)                       # [ d ]
        delta_t = delta_t.unsqueeze(1).repeat([1, self.M])   # [ d, M ]
        Omega   = self.Omega.unsqueeze(0)                    # [ 1, M ]
        kernel  = torch.exp(- delta_t * Omega)               # [ d, M ]
        kernel  = kernel.unsqueeze(0).repeat([self.K, 1, 1]) # [ K, d, M ]
        # convolution 
        conv_X  = (X * kernel).sum(1)                        # [ K, M ]
        return conv_X
    
    
def simulation(model):
    # evaluate model
    _, mus, triggs = model()
    # compute recovered outages
    triggs = triggs.detach().cpu().numpy().sum(0)
    mus    = mus.detach().cpu().numpy().sum(0)
    lams   = triggs + mus
    return lams.max()



def avg(mat, N=2):
    """
    calculate sample average for every N steps. 

    reference:
    https://stackoverflow.com/questions/30379311/fast-way-to-take-average-of-every-n-rows-in-a-npy-array
    """
    cum = np.cumsum(mat,0)
    result = cum[N-1::N]/float(N)
    result[1:] = result[1:] - result[:-1]

    remainder = mat.shape[0] % N
    if remainder != 0:
        if remainder < mat.shape[0]:
            lastAvg = (cum[-1]-cum[-1-remainder])/float(remainder)
        else:
            lastAvg = cum[-1]/float(remainder)
        result = np.vstack([result, lastAvg])

    return result

def proj(mat, coord, proj_coord, k=10):
    """
    project data defined by mat from coordinate system 1 to coordinate system 2.

    Args:
    - mat:        2D data matrix         [ n_days, n_from_locations ]
    - coord:      from coordinate system [ n_from_locations, 2 ]
    - proj_coord: to coordinate system   [ n_to_locations, 2 ]
    - k:          find the nearest k points
    """
    dist      = euclidean_distances(proj_coord, coord) # [ n_to_locations, n_from_locations ]
    argdist   = np.argsort(dist, axis=1)               # [ n_to_locations, n_from_locations ]
    neighbors = argdist[:, :k]                         # [ n_to_locations, k ]
    # projection
    N, K      = mat.shape
    proj_K    = proj_coord.shape[0]
    proj_mat  = np.zeros((N, proj_K)) 
    for t in range(N):
        for loc in range(proj_K):
            proj_mat[t, loc] = mat[t, neighbors[loc]].mean()

    return proj_mat

def load_weather(config):
    # load geo locations appeared in weather data
    geo_weather = np.load("C:/Users/scott/Desktop/anl-outage-data/data/%s" % config["weather_geo"])
    # load outage data
    print("[%s] reading weather data from data/%s ..." % (arrow.now(), config["weather_path"]))
    obs_feats  = [ np.load("C:/Users/scott/Desktop/anl-outage-data/data/%s/%s-feat%s.npy" % (config["weather_path"], config["weather_path"], feat)) for feat in config["feat_list"] ]
    obs_feats  = np.stack(obs_feats, 0)
    print("[%s] weather data with shape %s are loaded." % (arrow.now(), obs_feats.shape))

    # check if the start date and end date of weather data
    freq       = config["weather_freq"]
    startt     = arrow.get(config["weather_startt"], "YYYY-MM-DD HH:mm:ss")
    endt       = arrow.get(config["weather_endt"], "YYYY-MM-DD HH:mm:ss")
    assert int((endt.timestamp() - startt.timestamp()) / freq + 1) == obs_feats.shape[1], "incorrect number of recordings or incorrect dates."

    # select data in the time window
    start_date = arrow.get(config["_startt"], "YYYY-MM-DD HH:mm:ss")
    end_date   = arrow.get(config["_endt"], "YYYY-MM-DD HH:mm:ss")
    startind   = int((start_date.timestamp() - startt.timestamp()) / freq)
    endind     = int((end_date.timestamp() - startt.timestamp()) / freq)
    obs_feats  = obs_feats[:, startind:endind+1, :] # [ n_feats, n_times, n_locations ]
    print("[%s] weather data with shape %s are extracted, from %s (ind: %d) to %s (ind: %d)" % \
        (arrow.now(), obs_feats.shape, start_date, startind, end_date, endind))

    return obs_feats, geo_weather

def load_outage(config, N=4):
    # load geo locations appeared in outage data
    geo_outage = np.load("C:/Users/scott/Desktop/anl-outage-data/data/%s" % config["outage_geo"])
    # load outage data
    print("[%s] reading outage data from data/%s ..." % (arrow.now(), config["outage_path"]))
    obs_outage = np.load("C:/Users/scott/Desktop/anl-outage-data/data/%s" % config["outage_path"])
    print("[%s] outage data with shape %s are loaded." % (arrow.now(), obs_outage.shape))

    # check if the start date and end date of outage data
    freq       = config["outage_freq"]
    startt     = arrow.get(config["outage_startt"], "YYYY-MM-DD HH:mm:ss")
    endt       = arrow.get(config["outage_endt"], "YYYY-MM-DD HH:mm:ss")
    assert int((endt.timestamp() - startt.timestamp()) / freq + 1) == obs_outage.shape[0], "incorrect number of recordings or incorrect dates."

    # select data in the time window
    start_date = arrow.get(config["_startt"], "YYYY-MM-DD HH:mm:ss")
    end_date   = arrow.get(config["_endt"], "YYYY-MM-DD HH:mm:ss")
    startind   = int((start_date.timestamp() - startt.timestamp()) / freq)
    endind     = int((end_date.timestamp() - startt.timestamp()) / freq)
    obs_outage = obs_outage[startind:endind+1, :] # [ n_times, n_locations ]
    print("[%s] outage data with shape %s are extracted, from %s (ind: %d) to %s (ind: %d)" % \
        (arrow.now(), obs_outage.shape, start_date, startind, end_date, endind))

    # rescale outage data
    obs_outage = avg(obs_outage, N=N)

    return obs_outage, geo_outage

def dataloader(config, standardization=True, outageN=1, weatherN=1, isproj=True):
    """
    data loader for MA data sets including outage sub data set and weather sub data set

    - season: summer or winter
    """
    obs_outage, geo_outage = load_outage(config)
    obs_feats, geo_weather = load_weather(config)

    # # NOTE: FOR NCSC DATA
    n_locs    = obs_outage.shape[1]
    obs_feats = obs_feats[:, :, :n_locs]

    # data standardization
    print("[%s] weather data standardization ..." % arrow.now())
    if standardization:
        _obs_feats = []
        for obs in obs_feats:
            scl = StandardScaler()
            scl.fit(obs)
            obs = scl.transform(obs)
            _obs_feats.append(obs)
        obs_feats = _obs_feats

    # project weather data to the coordinate system that outage data is using
    print("[%s] weather data projection ..." % arrow.now())
    if isproj:
        obs_feats = [ proj(obs, coord=geo_weather, proj_coord=geo_outage[:, :2], k=10) for obs in obs_feats ]

    obs_outage  = avg(obs_outage, N=outageN).transpose()                    # [ n_locations, n_times ]
    obs_feats   = [ avg(obs, N=weatherN).transpose() for obs in obs_feats ] # ( n_feats, [ n_locations, n_times ] )
    obs_weather = np.stack(obs_feats, 2)                                    # [ n_locations, n_times, n_feats ]

    return obs_outage, obs_weather, geo_outage, geo_weather