from __future__ import division

import cPickle
import pystan
from pystan import StanModel

import numpy as np

#Generate data
#------------------START-------------------
N = 1000;
K = 5; #no. of clusters
alpha = 0.2 * np.ones(K);

#block structure
np.random.seed(42); #reproducible results
phi = np.random.rand(K,K);
phi = np.tril(phi) + np.tril(phi, -1).T; #symmetric

#cluster membership
clusters = np.random.choice(K, size = N, replace = True);

#sample data
graph = np.zeros([N,N]); #adjacency matrix rep.

for i in range(N):
	graph[i][i] = 1;
	for j in range(i+1,N):
		cluster_i = clusters[i];
		cluster_j = clusters[j];		
		conn = np.random.binomial(n=1,p=phi[cluster_i][cluster_j]);	

		#symmetrical connections
		graph[i][j] = conn;
		graph[j][i] = conn;

#-------------------END-------------------

pi_act = [list(clusters).count(x)/N for x in range(K)];
phi_act = phi;

#-----------------------------------------

data = {};
data['N'] = N;
data['K'] = K;
data['alpha'] = alpha;
data['graph'] = graph.astype(np.int64); #data consistency with stan

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

print "phi (actual):";
print phi_act;

#print "phi (inferred):";
#print phi_inf;

print "pi (actual):";
print pi_act;
#print "pi (inferred):";
#print pi_inf;

print fit;