from defs import *
import scipy.sparse
from itertools import combinations

def sample_graph(w):
    N = len(w)
    u = w/np.sqrt(w.sum())
    r = np.outer(u, u)
    p = r/(r + 1.)
    X = (np.random.rand(N, N) < p).astype(int)
    X = np.triu(X, k=1) #only upper triangular part

    graph = {}
    graph['N'] = N
    graph['row'], graph['col'], _ = scipy.sparse.find(X) #compact representation of edges of graph
    #form: {row,col} indicating two end points of each edge

    return graph

def get_pairs(graph):
    N = graph['N']
    row = graph['row']
    col = graph['col']

    #enumerate all edges
    pairs = np.array(list(combinations(range(N), 2)))
    pairs = np.column_stack((pairs, np.zeros(len(pairs), dtype=int))) 
    #^stacks a 0 with every possible edge pair formed in last step -- this gives 'connected?' info

    # fill in edges
    for (r, c) in zip(row, col):
        k = r*(2*N-r-1)/2-r + c-1
        pairs[k, 2] = 1 #connect

    return pairs

def get_degree(graph):
    N = graph['N']
    row = graph['row']
    col = graph['col']

    degree = np.zeros(N, dtype=int)
    for (r, c) in zip(row, col):
        degree[r] += 1
        degree[c] += 1
    return degree

def log_likel(pairs, w):
    if w.ndim == 1:
        w = w.reshape(1,-1) #W is a column vector
    [S,N] = w.shape #S samples of N w params
    u = w/np.sqrt(w.sum(axis=1).reshape(-1,1)) #normalize
    r = np.einsum('ij,ik->ijk', u, u) 
    #^^ for each row (sample of N latents), perform a U_i * U_j operation; This gives another 
    #dimension for each sample 'i'. Notation: for each row i, in both matrices, perform i_j * i_k, and index first by i, then j,k
    #so, no summation, only product. This is supposed to be an efficient operation internally!

    ll = -log(1 + r[:,pairs[:,0],pairs[:,1]]).sum(axis=1) #log_sum G(r)^-1 for all possible edges
    pos = pairs[:,2] == 1
    ll += log(r[:,pairs[pos,0],pairs[pos,1]]).sum(axis=1) #log_sum r_ij^x_ij -- only present edges
    #ll now has log likelihood for S samples

    return ll.mean() #sample mean

def log_likel_grad(pairs, w): #grad of LL wrt w latent variable
    N = len(w)
    wsum = w.sum()

    row = pairs[:,0]
    col = pairs[:,1]
    denom = 1./(wsum + np.outer(w, w)[row, col] + eps) #-ve term coming from differential of log G(r)^-1;
    #L in denom. will be removed as it is constant. Note: row-col => edge

    gw = -(denom.sum())*np.ones(N)

    gw -= np.bincount(row, weights=w[col]*denom, minlength=N)
    gw -= np.bincount(col, weights=w[row]*denom, minlength=N)
    #using bincount, for each node, multiply w_i * sum obtained in last step

    #next term is prod(U_i ^ D_i)
    iw = 1./(w + eps) #coming from log w_i
    pos = pairs[:,2] == 1 
    gw += (len(pairs) - pos.sum())/wsum #coming from derivative of L in deonominator
    prow = row[pos]
    pcol = col[pos]
    gw += np.bincount(prow, weights=iw[prow], minlength=N) #only for 'p'resent edges, same for next one
    gw += np.bincount(pcol, weights=iw[pcol], minlength=N)
    gw *= float(0.5*N*(N-1))/float(len(pairs)) #scaling

    return gw #gradient matrix
