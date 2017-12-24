#stan implementation of "power law simple graphs paper"
from __future__ import division
import numpy as np
import statsmodels.api as sm
import pystan
from scipy.stats import uniform, norm
import pickle

#data preparation
DATA_ROOT = './data/';

dataset = 'polblogs.pickle';

filename = DATA_ROOT+dataset
with open(filename, 'rb') as f:
    g = pickle.load(f)

N = g['N']
row = g['row']
col = g['col']

#row-column makes an edge

data = {};
data['N'] = N;
X = np.zeros([N,N]); #adjacency matrix

for (r,c) in zip(row,col):
	X[r][c] = 1;
	X[c][r] = 1;

data['X'] = X;

#model
stan_code="""
functions{
	real tBFRY_lpdf(real x, real alpha){
		real g = gamma_lpdf(x|1.0-alpha,1.0);
		real b = beta_lpdf(x|alpha,1.0);

		return g/b;
	}

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
	matrix[N,N] X;
}

parameters{
	real<lower=0> alpha; //parameter of distribution
	vector[N] w; //scalar embedding of graph rep. by X
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
		w[i] ~ tBFRY(alpha);
	}

	X ~ grg(r,N);
}

generated quantities{
	real log_lik;
	//likelihood eval

	log_lik = grg_lpdf(X|r,N);
}
""";

#Inference
fit = pystan.stan(model_code = stan_code, data = data, iter=1000, chains = 4, n_jobs=2, verbose = False);

log_lik = fit.extract('log_lik')['log_lik'];

print "log-likelihood:", np.mean(log_lik);

