#stan implementation of "power law simple graphs paper"
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
functions{
	real BFRY_lpdf(real w, real alpha){
		real eps = 1e-20; //small constant to avoid log(0)
		//By def of BFRY
		real lpdf = log(alpha + eps) - lgamma(1.0 - alpha) - (alpha + 1.0)*log(w + eps) + log(1 - exp(-w) + eps );

		return lpdf;
	}

	real tBFRY_lpdf(real w, real alpha, real C_n){
		//truncation
		real eps = 1e-20; //small constant to avoid log(0)
		if(w > C_n)
			return log(eps); //almost 0 prob.

		else
			return BFRY_lpdf(w|alpha);
	}

	//lpdf of generalized random graph as outlined in the paper
	real grg_lpdf(matrix X, matrix r, int N){
		real log_Gr = 0.0;
		real log_x_r = 0.0;
		real temp = r[1,1];

		for(i in 1:N){
			for(j in 1:N){
				log_Gr += log(1+r[i,j]);
				log_x_r += X[i,j] * log(r[i,j]);
			}
		}

		return (-log_Gr + log_x_r);
	}
}

data{
	int<lower=0> N;
	matrix[N,N] X_tr;
	matrix[N,N] X_ts;
	real C_n;
	real beta;
}

parameters{
	real<lower=0, upper=0.9999> alpha; //parameter of distribution
	vector<lower=0>[N] w; //scalar embedding of graph rep. by X
}

transformed parameters{
	vector[N] u;
	matrix[N,N] r;
	real L;

	L = sum(w);
	u = w/sqrt(L);

	r = u * u'; //uu^T
}

model{
	for(i in 1:N){
		w[i] ~ tBFRY(alpha,C_n);
	}

	X_tr ~ grg(r,N);
}

generated quantities{
	real tr_log_lik;
	real ts_log_lik;
	//likelihood eval

	tr_log_lik = grg_lpdf(X_tr|r,N);
	ts_log_lik = grg_lpdf(X_ts|r,N);
}
""";

#Inference
#fit = pystan.stan(model_code = stan_code, data = data, iter=1000, chains = 4, n_jobs=1, verbose = False);

#log_lik = fit.extract('log_lik')['log_lik'];

m = StanModel(model_code = stan_code);

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

	data['beta'] = beta;
	data['C_n'] = C_n;

	fit = m.vb(data = data);
	#print fit.keys();

	tr_ll = fit['mean_pars'][-2];
	ts_ll = fit['mean_pars'][-1];

	print "Inference Results: ";
	print "Train log-likelihood:", tr_ll;
	print "Test log-likelihood:", ts_ll;

	return [tr_ll, ts_ll];
#########-----------------------END-----------------------------#########

num_rounds = 10;
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

