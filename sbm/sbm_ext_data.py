from __future__ import division

import cPickle
import pystan
from pystan import StanModel
import pickle

import numpy as np

#Generate data
#------------------START-------------------
K = 5; #no. of clusters
alpha = 0.2 * np.ones(K);

#block structure
np.random.seed(42); #reproducible results

DATA_ROOT = '../data/';

dataset = '500Air.pickle';

filename = DATA_ROOT+dataset
with open(filename, 'rb') as f:
    g = cPickle.load(f)

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

data['K'] = K;
data['alpha'] = alpha;
data['graph'] = X.astype(np.int64); #data consistency with stan

#-------------------------------------------------------------------------------
#Load model -- code from https://github.com/darthsuogles/mmsb/blob/master/mmsb.py
def load_stan_model( model_name ):
    """
    Load stan model from disk, 
    if not exist, compile the model from source code
    """
    try:
        stan_model = cPickle.load( open(model_name + ".model", 'rb') )
    except IOError:
        stan_model = pystan.StanModel( file = model_name + ".stan" )
        with open(model_name + ".model", 'wb') as fout:
            cPickle.dump(stan_model, fout)
        pass

    return stan_model
#-------------------------------------------------------------------------------

m = load_stan_model("sbm_vect");
fit = m.vb(data = data);

#phi_inf = fit['mean_pars'][0];
#pi_inf = fit['mean_pars'][1];

#print "phi (actual):";
#print phi_act;

#print "phi (inferred):";
#print phi_inf;

#print "pi (actual):";
#print pi_act;
#print "pi (inferred):";
#print pi_inf;

print "log_lik: ", fit['mean_pars'][-1];