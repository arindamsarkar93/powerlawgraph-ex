#modified erdos-reyni model
from __future__ import division
import numpy as np
import statsmodels.api as sm
from scipy.stats import uniform, norm
import pickle
from math import ceil, floor

import pystan
from pystan import StanModel

#data preparation
DATA_ROOT = './data/';

dataset = '500Air.pickle';

filename = DATA_ROOT+dataset
with open(filename, 'rb') as f:
    g = pickle.load(f)

N = g['N']
row = g['row']
col = g['col']

#row-column makes an edge

#model
stan_code="""

data{
	int<lower=0> N;
	int X_tr [N,N];
	int X_ts [N,N];
	int K; //embedding size
}

parameters{
	matrix[N,N] p; //conn. prob.
}

model{
	for(i in 1:N){
		for(j in 1:N){
			p[i][j] ~ beta(1,1);
		}
	}

	for(i in 1:N){
		for(j in 1:N){
			X_tr[i][j] ~ bernoulli(p[i][j]);
		}
	}
}

generated quantities{
	real tr_log_lik=0.0;
	real ts_log_lik=0.0;
	//likelihood eval

	for(i in 1:N){
		for(j in 1:N){
			tr_log_lik += bernoulli_lpmf(X_tr[i][j]|p[i][j]);
			ts_log_lik += bernoulli_lpmf(X_ts[i][j]|p[i][j]);
		}
	}
}
""";

#Inference
#fit = pystan.stan(model_code = stan_code, data = data, iter=1000, chains = 4, n_jobs=1, verbose = False);

#log_lik = fit.extract('log_lik')['log_lik'];

#########-----------------------START-----------------------------#########
def run_inference(tr_split=0.8):
	#train-test split
	E = int(len(row));

	E_tr = int(floor(tr_split*E));
	E_ts = E - E_tr;

	idx = range(E);
	tr_idx = set(np.random.choice(idx,size = E_tr, replace = False));

	data = {};
	data['N'] = N;
	X_tr = np.zeros([N,N]); #adjacency matrix train
	X_ts = np.zeros([N,N]); #adjacency matrix test

	X_tr = X_tr.astype(np.int64);
	X_ts = X_ts.astype(np.int64);

	curr_idx = 0;

	for (r,c) in zip(row,col):
		if(curr_idx in tr_idx):
			X_tr[r][c]=1;
			X_tr[c][r]=1;

		else:
			X_ts[r][c] = 1;
			X_ts[c][r] = 1;

		curr_idx = curr_idx + 1;

	data['X_tr'] = X_tr;
	data['X_ts'] = X_ts;

	beta = 1.0;
	C_n = N**beta;

	#data['beta'] = beta;
	#data['C_n'] = C_n;
	data['K'] = K;

	fit = m.vb(data = data);
	#print fit.keys();

	tr_ll = fit['mean_pars'][-2];
	ts_ll = fit['mean_pars'][-1];

	print "Inference Results: ";
	print "alpha: ", fit['mean_pars'][0];
	print "Train log-likelihood:", tr_ll;
	print "Test log-likelihood:", ts_ll;

	return [tr_ll, ts_ll];
#########-----------------------END-----------------------------#########


m = StanModel(model_code = stan_code);

#embedding size:
K = 10;

num_rounds = 2;
tr_ll = 0;
ts_ll = 0;

#run multiple inference rounds--and average over runs
for i in range(num_rounds):
	res_ll = run_inference(0.8); #returns [tr_likelihood, ts_likelihood]
	tr_ll += res_ll[0];
	ts_ll += res_ll[1];


print "Average log-likelihoods:";
print "Train: ", tr_ll/num_rounds;
print "Test: ", ts_ll/num_rounds;

#log_lik = functions['log_lik'];
#print "log-likelihood:", np.mean(log_lik);

#print(fit['args']['sample_file'])

#print "log-likelihood:", np.mean(log_lik);
