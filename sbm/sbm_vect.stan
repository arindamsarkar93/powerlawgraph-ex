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

  for(i in 1:N){
    for(j in i+1:N){

      //marginalize out clusters
      log_lik += bernoulli_lpmf(graph[i][j]|pi' * phi * pi);
    }
  }
}