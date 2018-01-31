from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

def basic_1():
	""" http://www.cvxpy.org/en/latest/tutorial/intro/index.html 
	http://www.cvxpy.org/en/latest/tutorial/advanced/#solve-method-options """
	# Scalar optimization/decision variables
	x = Variable()
	y = Variable()

	# Create two constraints
	constraints = [x + y == 1,
	               x - y >= 1]

	# Objective function
	obj = Minimize(square(x - y))

	# Optimization computation
	prob = Problem(obj, constraints)
	prob.solve() 

	print "status:", prob.status
	print "optimal objective value", prob.value
	print "optimal decision variable", x.value, y.value

	# The optimal dual variable (Lagrange multiplier) for
	# a constraint is stored in constraint.dual_value.
	print "optimal (x + y == 1) dual variable", constraints[0].dual_value
	print "optimal (x - y >= 1) dual variable", constraints[1].dual_value
	print "x - y value:", (x - y).value


def basic_2():
	""" http://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/python_intro/notebooks/cvxprob.ipynb """	
	m = 10
	n = 7

	# Vector decision variables 
	x = Variable(n)

	# <=, >=, == are overloaded to construct CVXPY constraints
	constraints = [0 <= x, x <= 1]

	# Objective function
	np.random.seed(1)
	A = np.asmatrix(np.random.randn(m, n))
	b = np.asmatrix(np.random.randn(m, 1))
	# *, +, -, / are overloaded to construct CVXPY objective
	cost = sum_squares(A*x - b)
	objective = Minimize(cost)

	# Optimization computation
	prob = Problem(objective, constraints)	
	result = prob.solve()
	
	# The optimal value for x is stored in x.value.
	print x.value
	# The optimal Lagrange multiplier for a constraint
	# is stored in constraint.dual_value.
	print constraints[0].dual_value

def basic_3():
	""" binghamton simplex dual """
		# Scalar optimization/decision variables
	x = Variable()
	y = Variable()

	# Create two constraints
	constraints = [x + y <= 7,
	               4*x + 10*y <= 40,
	               -x <= -3,
	               x >= 0,
	               y >= 0 ]

	# Objective function
	obj = Maximize(30*x + 100*y)

	# Optimization computation
	prob = Problem(obj, constraints)
	prob.solve() 

	print "status:", prob.status
	print "optimal objective value", prob.value
	print "optimal decision variable", x.value, y.value

	# The optimal dual variable (Lagrange multiplier) for
	# a constraint is stored in constraint.dual_value.
	print "constraint 1 row operation", constraints[0].dual_value
	print "constraint 2 row operation", constraints[1].dual_value
	print "constraint 3 row operation", constraints[2].dual_value
	print "optimal (x - y >= 1) dual variable", constraints[3].dual_value
	print "optimal (x - y >= 1) dual variable", constraints[4].dual_value
	print "x - y value:", (x - y).value

def basic_4():
	""" binghamton Operations Research 04D: Simplex Method Entering & Leaving Variables, Pivoting """
		# Scalar optimization/decision variables
	x = Variable()
	y = Variable()

	# Create two constraints
	constraints = [2*x + y <= 9,
	               x + 2*y <= 9,	               
	               x >= 0,
	               y >= 0]

	# Objective function
	obj = Maximize(3*x + 2*y)

	# Optimization computation
	prob = Problem(obj, constraints)
	prob.solve() 

	print "status:", prob.status
	print "optimal objective value", prob.value
	print "optimal decision variable", x.value, y.value

	# The optimal dual variable (Lagrange multiplier) for
	# a constraint is stored in constraint.dual_value.
	print "constraint 1 row operation", constraints[0].dual_value
	print "constraint 2 row operation", constraints[1].dual_value
	print "constraint 3 row operation", constraints[2].dual_value
	print "optimal (x - y >= 1) dual variable", constraints[3].dual_value	
	print "x - y value:", (x - y).value

def click_ad():
	""" http://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/applications/optimal_ad.ipynb """
	np.random.seed(1)
	m = 5 # no. ad
	n = 24 # time slots: 24 hours
	SCALE = 10000
	
	## Budget
	B = np.random.lognormal(mean=8, size=(m,1)) + 10000 # [[ 25128.64678613],[ 11616.86373593],[ 11757.81747544],[ 11019.46308236],[ 17082.67988093]]))	
	B = 1000*np.round(B/1000) # [[25000], [12000], [12000], [11000], [17000]]
	
	## fraction of display that translates into click => depending on historical performance
	P_ad = np.random.uniform(size=(m,1)) # 
	print("P_ad: ", P_ad)
	P_time = np.random.uniform(size=(1,n))
	print("P time:", P_time)
	P = P_ad.dot(P_time)
	print("P: ", P)

	## actual total no. taffic: sum of a sine curve
	T = np.sin(np.linspace(-2*np.pi/2,2*np.pi  -2*np.pi/2,n))*SCALE	
	T += -np.min(T) + SCALE
	print("T: ",T)

	## actual no. of clicks
	c = np.random.uniform(size=(m,1)) # random ratio for 5 ads
	print(c)	
	c *= 0.6*T.sum()/c.sum() # traffic translates to clicks ratio amortized by each ad at random ratio: 0.6 * 479440.504606 / 3.55689437875
	print(c)	
	c = 1000*np.round(c/1000) # round up
	print(c)

	print(c.min())
	R = np.array([np.random.lognormal(c.min()/c[i]) for i in range(m)]) # revenue translate from click for each ad ratio
	print("R: ", R)

	# D = np.random.uniform(size=(m,n))
	# print([R[i]*P[i,:]*D[i,:].T for i in range(m)]) 

	# B is upper limit of revenue/getting paid => min_elemwise impose a upper bound on revenue
	D = Variable(m,n)
	## objective function
	Si = [min_elemwise(R[i]*P[i,:]*D[i,:].T, B[i]) for i in range(m)] # revenue is upper bounded by budget
	prob = Problem(Maximize(sum(Si)),
               [D >= 0,
                D.T*np.ones(m) <= T,
                D*np.ones(n) >= c])
	prob.solve()
	print("optimal display:", D.value.A)


def linear_regression():
	"""https://cs.stanford.edu/~tachim/optimization_code.html"""
	# add random noise to generated linear data
	x = np.arange(40)
	y = 0.3 * x + 5 + np.random.standard_normal(40)
	for i in xrange(40):
	    if np.random.random() < 0.1:
	        y[i] += 10
	plt.scatter(x, y)
	# plt.show()

	# decision variable
	w = Variable(); b = Variable()
	# objective function
	obj = 0
	for i in xrange(40):
	    obj += (w * x[i] + b - y[i]) ** 2
	Problem(Minimize(obj), []).solve()
	
	# optimal value
	w = w.value; b = b.value
	plt.scatter(x, y)
	plt.plot(x, w * x + b)
	plt.show()

def logistic_regression():
	""" http://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/applications/model_fitting.ipynb"""
	

def svm():
	""" https://cs.stanford.edu/~tachim/optimization_code.html """
	x1 = np.random.normal(2, 1, (2, 40))
	x2 = np.random.normal(-2, 1, (2, 40))
	plt.scatter(x1[0, :], x1[1, :], color='blue')
	plt.scatter(x2[0, :], x2[1, :], color='green')

	w = Variable(2); b = Variable()
	obj = 0
	for i in xrange(40):
	    obj += pos(1 - (w.T * x1[:, i] + b))
	    obj += pos(1 + (w.T * x2[:, i] + b))
	Problem(cvxpy.Minimize(obj), []).solve()
	x = np.arange(-6, 4)
	y = -(w.value[0, 0] * x + b.value) / w.value[1, 0]
	plt.plot(x, y, color='red')
	plt.scatter(x1[0, :], x1[1, :], color='blue')
	plt.scatter(x2[0, :], x2[1, :], color='green')
	plt.show()


def worst_case_risk():
	""" http://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/applications/worst_case_analysis.ipynb 
	NOT DONE!!!!!
	"""
	np.random.seed(2)
	n = 5
	mu = np.abs(np.random.randn(n, 1))/15
	print("mu", mu)
	Sigma = np.random.uniform(-.15, .8, size=(n, n))
	print("Sigma: ",Sigma)
	Sigma_nom = Sigma.T.dot(Sigma)
	print "Sigma_nom ="
	print np.round(Sigma_nom, decimals=2)




def main():
	# basic_1()
	# click_ad()
	linear_regression()



if __name__ == '__main__':
	main()