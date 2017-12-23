# 

## Information theory


### Information
$$-log(P(X))$$
i.e. how rare is event X, higher probability, lower amount of information

### Entropy
Average information (Entropy) => expectation of log probabiliity i.e. P(X) * -log(P(X))

### Differential Entropy
Entropy of continuous variable

expectation of seeing useful information for certain continous distribution 

### KL Divergence == Relative Entropy

Motivation: approximating conditional probability p(z|x) using q(z) => KL is hard since conditional is hard to get (since you need to do integral), but lower is easy since joint is easy to get (if you have a probabilitic graphical model)

measures distance between 2 probability distribution

relative entropy = difference between entropys with respect to 1st

not symmetric => asking A how rich B is is different form asking B how rich A is.

KL divergence to measure quality of estimation. i.e. if kl divergence is small, estimation is close.

KL(q(z)||p(z|x)): measuring the quality of esitmation using q(z) to estimate p(z|x)

log P(X) = KL + lower bound where log P(X) is fixed and wont change. i.e. lower bound L controls KL divergence, less negative lower bound, lower kl

lower bound: 
$$\sum q(z) log \frac{p(x,z)}{q(z)}$$

## variational inference

NOTE: you could also use metropolis hasting (exact), in comparison, variational inference is deterministic, approximation, less accurate but less time to compute. Also Laplace Approximation

bypass the need to solve integral and get conditional directly from joint pdf

Now we are trying to maximize lower bound (less negative) so KL is lowest. => q(z) should be as much close to 1, whereas p(x,z) should be as close to q(z) and as close to 1, given P(x,z) is always smaller than q(z) 

assume z1, z2, z3 are independent