/**
Stochastic Blockmodel STAN Model
**/

data{
  int<lower=1> N; //data points
  int<lower=1> K; //clusters
  vector[K] alpha; //dirichlet hyper param
  int graph[N,N]; //adjacency matrix
}

transformed data{
  real a;
  real b;
  a = 1.0;
  b = 1.0;
}

parameters{
  matrix<lower=0, upper=1>[K,K] phi; //block matrix
  simplex[K] pi; //cluster membership
}

model{
  for(i in 1:K){
    for(j in 1:K){
      phi[i][j] ~ beta(a,b); //prior on block matrix entries
    }
  }

  pi ~ dirichlet(alpha); //mixture distribution

  for(i in 1:N){
    for(j in i+1:N){ //symmetry and ignore diagonals

      //marginalize out clusters
      graph[i][j] ~ bernoulli(pi' * phi * pi); //sum over pairs <-- Doesn't seem correct! What about p(0|..) case?
    }
  }
}

generated quantities{
  //likelihood?
  real log_lik = 0.0;

  matrix[N,K] log_zprob; //cluster probability of each node
  vector[K] lps;
  int clusters_inf[N];

  //likelihood
  for(i in 1:N){
    for(j in i+1:N){

      //marginalize out clusters
      log_lik += bernoulli_lpmf(graph[i][j]|pi' * phi * pi);
    }
  }

  //clusters

  for(i in 1:N){

    clusters_inf[i] = 1; //initialization
    //prob. -- node i has cluster z_i
    for(z_i in 1:K){
      log_zprob[i][z_i] = log(pi[z_i]);

      for(j in 1:N){

        for(z_j in 1:K){
          lps[z_j] = log(pi[z_j]) + bernoulli_lpmf(graph[i][j] | phi[z_i][z_j]);
        }

        log_zprob[i][z_i] += log_sum_exp(lps);
      }

      if(log_zprob[i][z_i] > log_zprob[i][clusters_inf[i]]){
        clusters_inf[i] = z_i;
      }
    }
  }

  //convert to python indices
  for(i in 1:N){
    clusters_inf[i]-=1;
  }
}