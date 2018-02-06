#!/usr/bin/env python
"""Stochastic block model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

import matplotlib.pyplot as plt

ed.set_seed(42)

# DATA
"""
X_data, Z_true = karate("~/data")
N = X_data.shape[0]  # number of vertices
K = 2  # number of clusters
"""

#Generate data
#------------------START-------------------
N = 100;
K = 5; #no. of clusters
#alpha = 0.2 * np.ones(K);

#block structure
np.random.seed(42); #reproducible results

#fixed block structure
phi = [[0.5, 0.7, 0.8, 0.9, 0.8],
       [0.7, 0.5, 0.2, 0.1, 0.2],
       [0.8, 0.2, 0.5, 0.1, 0.1],
       [0.9, 0.1, 0.1, 0.5, 0.1],
       [0.8, 0.2, 0.1, 0.1, 0.5]];



#phi = np.random.rand(K,K);
#phi = np.tril(phi) + np.tril(phi, -1).T; #symmetric

#cluster membership
#cluster_pref = [0.75, 0.20, 0, 0, 0.05];
clusters = np.random.choice(K, size = N, replace = True);#,p = cluster_pref);

#sample data
graph = np.zeros([N,N]); #adjacency matrix rep.

#sparse?
sparsity = 0.5;

for i in range(N):
	graph[i][i] = 1;
	for j in range(i+1,N):
		cluster_i = clusters[i];
		cluster_j = clusters[j];		
		conn = np.random.binomial(n=1,p=phi[cluster_i][cluster_j] * sparsity);	

		#symmetrical connections
		graph[i][j] = conn;
		graph[j][i] = conn;

X_data = graph;
Z_true = clusters;
membership_act = [list(clusters).count(x)/N for x in range(K)];
#-------------------END-------------------



# MODEL
gamma = Dirichlet(concentration=tf.ones([K]))
Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K]))
Z = Multinomial(total_count=1.0, probs=gamma, sample_shape=N)
X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))

# INFERENCE (EM algorithm)
qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([K]))))
qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))

inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: X_data})

n_iter = 250
inference.initialize(n_iter=n_iter)

tf.global_variables_initializer().run()

for _ in range(inference.n_iter):
  info_dict = inference.update()
  inference.print_progress(info_dict)

inference.finalize()

# CRITICISM
Z_pred = qZ.mean().eval().argmax(axis=1)
pi_pred = qPi.mean().eval();

#print("Actual");
#print(np.asarray(phi));
#print(membership_act);

#print("predicted")
#print(pi_pred);
#print(qgamma.mean().eval());

X_pred = np.array(X.mean().eval() > 0.5, dtype=int);
cnt = N*N;
correct = np.sum(X_data == X_pred);

plt.subplot(211);
plt.imshow(X_data, cmap='Greys');

plt.subplot(212)
plt.imshow(X_pred, cmap='Greys');
plt.show();

print("Correctly predicted: ", correct);
print("Total entries: ", cnt);

print("Train Accuracy: ", correct/cnt);

print("Result (label flip can happen):")
print("Predicted")
print(Z_pred)
print("True")
print(Z_true)
print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))
