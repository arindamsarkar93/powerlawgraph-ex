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
	matrix[K,K] lps;

	for(i in 1:K){
		for(j in 1:K){
			phi[i][j] ~ beta(a,b); //prior on block matrix entries
		}
	}

	pi ~ dirichlet(alpha); //mixture distribution

	for(i in 1:N){
		for(j in 1:N){

			//marginalize out clusters
			for(k_i in 1:K){
				for(k_j in 1:K){
					lps[k_i][k_j] = log(pi[k_i]) + log(pi[k_j]) + bernoulli_lpmf(graph[i][j]|phi[k_i][k_j]);
				}
			}

			target += log_sum_exp(lps); //for log likelihood update
		}
	}
}

generated quantities{
	//likelihood?
	real log_lik = 0.0;

	for(i in 1:N){
		for(j in 1:N){

			//marginalize out clusters
			for(k_i in 1:K){
				for(k_j in 1:K){
					log_lik += log(pi[k_i]) + log(pi[k_j]) + bernoulli_lpmf(graph[i][j]|phi[k_i][k_j]);
				}
			}
		}
	}

}