## What is MCMC?
Markov Chain Monte Carlo is a technique to solve the problem of sampling from a complicated distribution.

Definition (The sampling problem):  Let D be a distribution over a finite set X. You are given black-box access to the probability distribution function p(x) which outputs the probability of drawing x $\in$ X according to D. Design an efficient randomized algorithm A which outputs an element of X so that the probability of outputting x is approximately p(x). More generally, output a sample of elements from X drawn according to p(x).

## Why MCMC?

## Transition Matrix, Stationary Distribution, Detailed Balance Condition

```python

```

Given posterior as stationary distribution of a Markov Chain, we generate samples from this posterior distribution. (TBC)


Given the stationary distribution of our Markov Chain, the posterior integral from Bayesian inference is intractable. (TBC) => but how is MCMC linked to transition matrix and posterior bayesian inference ??

The main theorem we need to do anything useful with Markov chains is the stationary distribution theorem (sometimes called the “Fundamental Theorem of Markov Chains,” and for good reason). What it says intuitively is that for a very long random walk, the probability that you end at some vertex v is independent of where you started! All of these probabilities taken together is called the stationary distribution of the random walk, and it is uniquely determined by the Markov chain.

Graph is required to be strongly connected

## Intuition from Reject Sampling

Reject Sampling is an important building block in MCMC.

Reject Sampling provides an easy way to sample form a complex distribution whose integration is not immediately obvious.

The only requirement is to define a wrapper probability density function that wraps the real data distribution we try to approximate. It does not matter what the distribution of wrapper is, it could be uniform or normal or any other distribution.

Let's assume we use a normal as wrapper.
The intuition of approximation is:
* acceptance probability is the ratio between the height of wrapper and real data approximation, where height of wrapper represents the times we will draw such value and height of real data approximation represents the volume/amount of such real data's value
* when we draw from wrapper function from a normal distribution, value around the mean $\mu$ should have a higher presence (we could see this from the height of wrapper) i.e. we are likely to draw more of value around the mean $\mu$
* however, the uniform distribution acted as a gate/threshold to proportionally allow value to come through where higher value has a proportionally higher chance to be "accepted" as an sample
* real data is approximated through the percentage the height ratio is making through the acceptance gate, when drawing in normal distribution
* Think of it as 3 scenarios when draws more: 1. draws more samples + acceptance ratio high = more accepted samples 2.draws more samples + super low acceptance ratio = very low samples (despite more draws, more proportion did not make through) 3. normal in wrapper + flat line in real data = it is true that we draw more samples, but acceptance ratio is decreased so real data approximation line remains flat


```python

import scipy.stats as stat
import numpy as np


def scale_wrapper():
    x_axis = np.arange(-15, 15, 0.001) # x-axis between -10 and 10 with .001 steps.
    real_data_distribution = stat.norm.pdf(x_axis,3,1) + stat.norm.pdf(x_axis, -5, 2) # list of all value's probability
    wrapper_distribution = stat.norm.pdf(x_axis,0,4)
    c =  np.max(real_data_distribution/wrapper_distribution) # c is scaling down
    return c

def plot_real_data_distribution():
    """
    plot
    """
    x_axis = np.arange(-15, 15, 0.001) # x-axis between -10 and 10 with .001 steps.
    real_data_distribution = stat.norm.pdf(x_axis,3,1) + stat.norm.pdf(x_axis, -5, 2) # list of all value's probability
    plt.plot(x_axis, real_data_distribution, color = 'green', label = 'real data')

    # wrapper distribution
    wrapper_distribution = stat.norm.pdf(x_axis,0,4)
    c =  np.max(real_data_distribution/wrapper_distribution) # c is scaling down

    #PLOT SCALED PROPOSAL/ENVELOP DISTRIBUTION
    plt.plot(x_axis,c*wrapper_distribution, color = 'pink', label = 'wrapper');
    plt.legend()
    plt.show()
  
def real_data_distribution(x):
    """
    pdf() returns the height
    
    ?? why no c => c is used to scale down
    """
    return stat.norm.pdf(x, 3,1) + stat.norm.pdf(x,-5,2) 

def reject_sampling():
    """
    1. draw one sample from normal distribution
    2. use that sample to draw from real data approximation
    3. accept if acceptance > uniform
    """
    accepted_samples = []
    for i in xrange(10000):
        x_i = np.random.normal(0,4,1)[0] # x_i: random variable(row) not prob(height), 0: mean, 4:sd, 1:num of samples
        real_data_prob = real_data_distribution(x_i)
        scale_ratio = scale_wrapper()
        acceptance_prob = real_data_prob/ (scale_ratio*(stat.norm.pdf(x_i, loc=0, scale=2)))        
        uniform_prob = np.random.uniform(0,1,1)
        if acceptance_prob > uniform_prob:
            accepted_samples.append(x_i)
            
```

## Metropolis

```python

def metropolis_hastings():
    """
    1. sample joint distribution rv and x_k random variables
    2. combine into initial state/sample of such joint distribution
    3. calculate initial state's loglike (i.e. how likely is sample from this joint distribution)
    4. use initial state loglike as threshold
    5. accept if (loglike_new - loglike_old) > uniform
    6. if accepted => propose a new state conditioned on the accepted state
    7. resulted in distribution of samples(i.e. likelihood of samples drawn conditionally) 
    """

```



## MCMC

Transition matrix is used to update real data approximation.

Difference from reject sampling + Intuition:
* State/sample (i.e. random variable) accepted is used to draw new samples.
* 

?? is it because MCMC and metropolis is different 

?? also could one based on what we sample => proposed a new wrapper function => so we could effeciently draw from real data distribution??
?? calculating difference between samples drawn and proposed distirubtion is it called KL divergence/ variational inference 


## Application of MCMC
1. MCMC cypher
   code:

2. Knapsack Problem


## Gibbs sampling

## Visualizing dirichlet distribution

## Expectation Maximization
1. Running expectation maximization to find parameters for gaussian mixture models

2. EM to fit a diagonal covariance Gaussian mixture model to text data

## LDA

1. Code

2. Mathematical Proof

why sampling as opposed to just calculate real probability??
prior ??
preserve dirichlet distribution of word topics and topic documents??


## Slice Sampling




reference:
http://isaacslavitt.com/2013/12/30/metropolis-hastings-and-slice-sampling/
https://nbviewer.jupyter.org/github/yoyolin/mcmc-tutorial/blob/master/MCMC_for_Bayesian_Inference.ipynb
https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/
http://statweb.stanford.edu/~cgates/PERSI/papers/MCMCRev.pdf
http://mlwhiz.com/blog/2015/08/21/MCMC_Algorithms_Cryptography/
https://github.com/tuanavu/coursera-university-of-washington/blob/master/machine_learning/4_clustering_and_retrieval/assigment/week4/4_em-with-text-data_graphlab.ipynb
