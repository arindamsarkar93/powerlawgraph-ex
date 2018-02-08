/**
Latent Space Models : STAN Model
**/

data{
  int<lower=1> N; //data points
  int<lower=1> D; //embedding size
  int graph[N,N]; //adjacency matrix
}

transformed data{
  real a;
  real b;
  a = 1.0; //try other values here
  b = 2.0;
}

parameters{
  matrix[N,N] Z; //per edge bias
  row_vector[D] X[N]; //node embeddings
  real<lower=machine_precision()> nu[D];   
}

transformed parameters{
  //multiplicative inverse gamma prior
  matrix[D][D] lambda; //relative embedding importance - xx[positive def.] scaling matrix
  //positiveness taken care by priors 

  lambda[1][1] = 1/nu[1];

  for(i in 1:D){
    for(j in 1:D){
      if(i==j && i>1){
        for(d in 1:i){
          lambda[i][i] = lambda[i][i] * (1/nu[d]);
        }
      }

      else{
        lambda[i][j] = 0;
      }
    }
  }

}

model{
  for(i in 1:N){
    for(j in 1:N){
      Z[i][j] ~ normal(0,1); //prior on Z --> over simplistic?
    }

    for(j in 1:D){
      X[i][j] ~ normal(0,1);
    }
  }

  //a,b being hyperparameters
  nu[1] ~ gamma(a,1);

  for(d in 1:D){
    nu[d] ~ gamma(b,1);
  }

  for(i in 1:N){
    for(j in 1:N){
      graph[i][j] ~ bernoulli(inv_logit(Z[i][j] + X[i] * lambda * X[j]'));
    }
  }
}

generated quantities{
  //likelihood?
  real log_lik = 0.0;

  for(i in 1:N){
    for(j in 1:N){
      log_lik += bernoulli_lpmf(graph[i][j]|inv_logit(Z[i][j] + X[i] * lambda * X[j]'));
    }
  }
}