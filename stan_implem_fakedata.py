#stan implementation of "power law simple graphs paper"
from __future__ import division
import numpy as np
import statsmodels.api as sm
from scipy.stats import uniform, norm
import pickle

import pystan
from pystan import StanModel

N = 100;

data = {};
data['N'] = N;
X = np.random.randint(2,size=(N,N)); #adjacency matrix
data['X'] = X;

beta = 1.0;
C_n = N**beta;

data['beta'] = beta;
data['C_n'] = C_n;

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
	matrix[N,N] X;
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

	X ~ grg(r,N);
}

generated quantities{
	real log_lik;
	//likelihood eval

	log_lik = grg_lpdf(X|r,N);
}
""";

#Inference
#fit = pystan.stan(model_code = stan_code, data = data, iter=1000, chains = 4, n_jobs=1, verbose = False);

#log_lik = fit.extract('log_lik')['log_lik'];

m = StanModel(model_code = stan_code);
fit = m.vb(data = data);
#print fit.keys();

print "log-likelihood:", fit['mean_pars'][-1];

#log_lik = functions['log_lik'];
#print "log-likelihood:", np.mean(log_lik);

#print(fit['args']['sample_file'])

#print "log-likelihood:", np.mean(log_lik);

