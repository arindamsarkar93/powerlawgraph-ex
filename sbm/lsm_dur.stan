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
  //matrix<lower=-0.0001>[N,N] Z; //per edge bias --> not required for single network
  row_vector<lower=-0.0001>[D] X[N]; //node embeddings
  real<lower=0.001> nu[D];
}

transformed parameters{
  //multiplicative inverse gamma prior
  matrix<upper=100>[D,D] lambda; //relative embedding importance - xx[positive def.] scaling matrix
  //positiveness taken care by priors 
  //matrix[N,N] il_param; //inv logit param

  //inv. gamma instead of mult inv gamma -- to handle some issues
  for(i in 1:D){
    for(j in 1:D){
      if(i==j){
        lambda[i][i] = (1/nu[i]);
      }

      else{
        lambda[i][j] = 0;
      }
    }
  }

  /*
  for(i in 1:N){
    for(j in 1:N){
      il_param[i][j] = max(0,Z[i][j] + X[i] * lambda * X[j]');
    }
  }
  */

}

model{
  for(i in 1:N){
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
      graph[i][j] ~ bernoulli(inv_logit(X[i] * lambda * X[j]'));
    }
  }
}

generated quantities{
  //likelihood?
  real log_lik = 0.0;
  //real param;

  for(i in 1:N){
    for(j in 1:N){
      //param = max(0,Z[i][j] + X[i] * lambda * X[j]');
      log_lik += bernoulli_lpmf(graph[i][j]|inv_logit(X[i] * lambda * X[j]'));
    }
  }
}