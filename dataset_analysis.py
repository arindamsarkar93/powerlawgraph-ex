import numpy as numpy
import pickle
import matplotlib.pyplot as plt

DATA_ROOT = '/home/arindam/code/powerlawgraph/data/';

dataset = 'polblogs.pickle';

filename = DATA_ROOT+dataset
with open(filename, 'rb') as f:
    graph = pickle.load(f)

N = graph['N']
row = graph['row']
col = graph['col']

print N,"nodes";
#print row;
#print col;

deg = [];
deg_dist = [];

for i in range(N):
	deg.append(0);
	deg_dist.append(0);


for i in range (len(row)):
	r = row[i];
	c = col[i];

	deg[r] = deg[r]+1;
	deg[c] = deg[c]+1;


#plot degree distribution
for i in range(N):
	deg_dist[deg[i]] = deg_dist[deg[i]]+1;


plt.scatter(range(N),deg_dist,s=5);
plt.ylim(1, max(deg_dist)+1)
plt.show()