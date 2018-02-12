---
layout: post
comments: true
title:  "Introduction and real world application of Linear Programming(LP). Rethinking linear regression, logistic regression, support vector machine in Convex Optimization."
excerpt: " In progress... "
date:   2017-01-14 11:00:00
mathjax: true
---


## simplex (actual implemntation in algorithm form) n its graphical meaning

	Intuitively, ERO is looking for extreme points in constraints.
	Here is a classical example:


<div class="imgcap">
<img src="/assets/LP_1/simplex.jpg" height="400">
</div>


		1. Pivoting: determine entering & leaving variables + ratio test to determine pivot row

			=> simplex canonical form n canonical simplex tableau

			=> pivoting: trying out extreme point according to objective function, adjacent bfs by chaning 1 variable at 1 time

			=> pivot row + entering variable (pivot column) == pivot point (extreme point)		
				=> entering variable: decision variables that causes the biggest increase in objective function		
				=> slack variable in same pivot row becomes zero
				=> elimination row operation == found adjacent BFS (adjacent extreme point)
				=> get difference between 2 constraints changing 1 variable in BFS to adjacent BFS, given we have the same variable in BFS (i.e. when constraints meet how much difference in that variable in adjacent BFS)		
					=> convert to [1 0 1 0] that measure difference
				=> when minus contraint with objective function, you are left with SLOPE DIFFERENCE 
				=> when minus 1 constraint from another, looking for point that coincides, then see how that slope difference translate to value at that point
				=> slope difference: share area between objective line and constraint line difference i.e. potential for constraint line to go further for a higher value
				=> entering variable might not be pivot
						
			=> ratio test to decide pivot row 
				=> determine largest value entering variable (x or y) could be without violating constraints

				=> i.e. amount all constraints the smallest so it wont violate other constraints when we zero out all other decision variables
				=> rhs of constraint row / coef of entering var in constraint row				
			
			=> pivot row + entering variable (pivot column) == pivot point (extreme point)		
				=> entering variable: decision variables that causes the biggest increase in objective function		
				=> slack variable in same pivot row becomes zero
				=> elimination row operation == found adjacent BFS (adjacent extreme point)
				=> get difference between 2 constraints changing 1 variable in BFS to adjacent BFS, given we have the same variable in BFS (i.e. when constraints meet how much difference in that variable in adjacent BFS)		
					=> convert to [1 0 1 0] that measure difference
				=> when minus contraint with objective function, you are left with SLOPE DIFFERENCE 
				=> when minus 1 constraint from another, looking for point that coincides, then see how that slope difference translate to value at that point
				=> slope difference: share area between objective line and constraint line difference i.e. potential for constraint line to go further for a higher value
				=> entering variable might not be pivot

		2. ERO + updates obj value
			=> get rid of same variable in other rows by ERO 

Graphically:

<div class="imgcap">
<img src="/assets/LP_1/graphical.jpg" height="400">
</div>


		convert to stadard form: 
		add one slack variable for each inequality, which turn inequality into equality
			=> slack indicates room from constraint, allow such extreme point not on the constraint line i.e. sum of decision variable distance from constraint
			=> n excess variables 
		(should not exist for all constraint equations in max since constraint should be upper rather than lower bounded, however okay if 1 excess and 1 slack)
		

		=> 2. Basic feasible solution i.e. extreme point => given n variables and m constraints (n>m), basic feasible solution is obtained by setting n-m variables non-basic variables (NBV) to 0 to solve for m variables Basic Variables (BV) i.e. just pick 1 variable and set rest to 0 in each constraint
		if LP has optimal solution, there could be a set of points and at least 1 of these set of points is extreme point
		no. of basic feasible solution: number of variables in each constraints multiply by each other (only 1 variable in each constraint could take on non-zero value)
		adjacent fesible solution: change of 1 decision variable


	Formally, simplex in matrix form:

<div class="imgcap">
<img src="/assets/LP_1/simplex_matrix.jpg" height="400">
</div>

	I wrote some python code to demonstrate simplex solver from scratch.

```python

"""
Author: Steven Wong
This software has not been tested rigoriously and not meant to be distributed commercially.
"""

import numpy as np
from operator import sub

class LinearProgram(object):
	"""LinearProgram"""
	def __init__(self, A, b, c):		
		self.constraints = A
		self.rhs = b
		self.obj = c	
		self.updated_obj = [ -x for x in self.obj] 
		self.BV = []
		self.NBV = []

		self.canonical_simplex_tableau()
		
		
	def get(self, mat, col_list):
		"""get A_bv, A_nbv, c_bv, c_nbv i.e. constraints matrix n objective vector with only BV or NBV columns"""
		np_mat = np.matrix(mat)
		# print(b[:,col_list])
		return np_mat[:,col_list]

	def canonical_simplex_tableau(self):
		"""add slack variables + init BV, NBV"""

		self.num_constraints = len(self.constraints)
		num_decision_variables = len(self.constraints[0])
		self.NBV = [i for i in range(num_decision_variables)]

		zeros = [0] * self.num_constraints
		self.updated_obj += zeros
		augmented_matrix = [r + zeros for r in A]			
		col = 1
		for row, _ in reversed(list(enumerate(augmented_matrix))):			
		 	# print(augmented_matrix[row][-col])
		 	augmented_matrix[row][-col] +=1
		 	col += 1	
		 	self.BV.append(len(augmented_matrix[0])-row-1) ## init BV
		self.augmented_matrix = augmented_matrix	

					
	def simplex_solve(self):
		""" keep looping if still negative in objective functoin"""
		optimal = 1
		loop = 1
		while(optimal):
			print("\n")
			print("ITERATION:", loop)
			loop +=1
			self.entering_variable()
			self.update_obj()

			for i in range(len(self.constraints)):
				if self.updated_obj[i] < 0:
					break
				optimal = 0		
		self.calculate_optimal()
		print("\n")
		print("SOLUTION: ", self.b_hat, self.opt)

	def calculate_optimal(self):
		self.obj += [0] * len(self.constraints)
		self.A_bv = self.get(self.augmented_matrix, self.BV)
		self.b_hat = np.dot(self.A_bv.I, self.rhs).tolist()[0]
		self.x = [0] * len(self.augmented_matrix[0])
		for i, j in zip(self.BV, self.b_hat):
			self.x[i] = j
		
		self.x = np.matrix(self.x)
		self.obj = np.matrix(self.obj)
		self.opt = self.obj.dot(self.x.T)


	def update_obj(self):
		""" 
		r = C_n - A_n.T * A_b.-T * C_b
		find linear combination that turn BV into identity
		
		+ use that linear combination to update NBV objective value
		
		+ turn all C_bv into zeros"""
		
		print("C_nbv: ", self.get(self.updated_obj, self.NBV))
		print("C_bv: ", self.get(self.updated_obj, self.BV))
		print("A_bv inver, trans: ", self.get(self.augmented_matrix, self.BV).I.T)
		print("A_nbv trans: ", self.get(self.augmented_matrix, self.NBV).T)

		A_nbv_T = self.get(self.augmented_matrix, self.NBV).T
		A_bv_I_T = self.get(self.augmented_matrix, self.BV).I.T
		c_nbv = self.get(self.updated_obj, self.NBV)
		c_bv = self.get(self.updated_obj, self.BV).T

		temp = A_nbv_T.dot(A_bv_I_T).dot(c_bv)
		print(temp, c_nbv)
		# c_nbv = map(sub, c_nbv.tolist(), temp.tolist())
		c_nbv = [a - b for a, b in zip(c_nbv, temp.T)][0].tolist()[0]
		# c_nbv = c_nbv.tolist() - temp.tolist()

		print("c_nbv",c_nbv)
		
		self.updated_obj = [0] * len(self.augmented_matrix[0])		

		for i, j in zip(self.NBV, c_nbv):
			self.updated_obj[i] = j
		print("objective row", self.updated_obj)

		# self.get(self.obj, BV).zeros() # zero out c_bv


	def entering_variable(self):
		""" 
		Pivoting 
		1. pivot col/entering variable 
		2. ratio test to pivot row
		3. update NBV n BV 
		"""

		min_value = min(self.updated_obj)
		# max_index = [i for i, j in enumerate(self.obj) if j == max_value]
		self.pivot_col_index = self.updated_obj.index(min_value) 
		
		self.A_bv = self.get(self.augmented_matrix, self.BV) # constraints with only BV as column
		print("A_bv:", self.A_bv)
		self.b_hat = np.dot(self.A_bv.I, self.rhs)
		print("b_hat: ",self.b_hat)

		self.pivot_col = self.get(self.augmented_matrix, self.pivot_col_index)		
		self.d = np.dot((-self.A_bv.I), self.pivot_col)		
		self.ratio_test = (-self.b_hat/self.d.T).tolist()[0] # convert np back to array		
		
		# self.pivot_row_index = self.ratio_test.index(min(self.ratio_test))		
		self.pivot_row_index = self.ratio_test.index(min(i for i in self.ratio_test if i > 0))		
		print("BV, NBV before", self.BV, self.NBV)

		## update BV
		self.BV[self.pivot_row_index] =\
			self.pivot_col_index		
		## update NBV
		compare = [i for i in range(len(self.augmented_matrix[0]))]		
		self.NBV = list(set(self.BV)^set(compare))
		print("updated BV, NBV: ", self.BV, self.NBV)


if __name__ == "__main__":


	# constraint matrix
	A = [[-1,2],
	     [3,1]]	     
	# rhs
	b = [4,9]	
	# objective function
	c = [1,2]


	LP = LinearProgram(A, b, c)
	LP.simplex_solve()


```

	=> extreme point definition: 1 coordinate is 0 or interception
	=> convert LP into Standard Form
		=> all decision variables (variables in objective function) are non-negative
				


		=> EXAMPLE: patrict JMT 3d data, 2 contraints case


		=> 5. matrix form
			=> RGB math lec12a




## SVD post UPDATES
	=> covariance matrix be thought of as projection matrix
	=> eigenvector perspective

### SVM vs Ax=b vs linear regress vs logistic regression vs convex optimization vs svd
	=> linear regression minimize sum of squared vertical distance vs pca minimize sum of squared perpendicular distance
		=> cs168 lec 7
	=> least square 
		=> svd vs linear regression
		=> linear algebra vs calculas(convex optimization) perspectives 
			=> 18.06 lec 14-16 (projection matrices into col n null space)

	=> convex opt
		=> linear vs svm 
		https://stats.stackexchange.com/questions/95340/comparing-svm-and-logistic-regression
		http://www.cs.toronto.edu/~kswersky/wp-content/uploads/svm_vs_lr.pdf




convex optimization => LP + QP
LP 
	=> simplex + dual + kkt + max min (graph problem in general) + bi partite, image segmentation + compressive sesning, sparse recovery + l1 minimization + click ad revenue optimization + portfolio optimization(QP, linear program with random cost) + Worst-case risk analysis (semi defnite matrix) + Model fitting i.e. logistic regression + linear regression (Huber regression)

	=> another perspective: graph problem
		=> MAx flow can be encoded as LP, but all graph problems ?? LP solve pagerank ??
		=> TSP
			=> https://hal.archives-ouvertes.fr/inria-00504914/document
		=> operation research graph problems..??
	=> e.g. military transhippment, max flow, bi partite
	=> function convexity: DCP (http://www.cvxpy.org/en/latest/tutorial/dcp/) => tree visualization to show convexity
	=> http://www.cvxpy.org/en/latest/tutorial/advanced/
	=> http://www.cvxpy.org/en/latest/examples/index.html
	=> https://blog.quantopian.com/markowitz-portfolio-optimization-2/
	=> https://ocw.mit.edu/courses/mathematics/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/lecture-notes/MIT18_S096F13_lecnote14.pdf
	=> https://sites.math.washington.edu/~burke/crs/408/fin-proj/mark1.pdf
	=> constraint being semidefinite ??
	=> why dual == lagrangian ??
	=> dual/u meaning if objective not linear??
	=> determinat >0 == eigen value >0 == positive semi definte ??
	=> eignevalue decomposition == pagerank ??
	=> LP could be encoded as Semi definite ??
	=> lagrangian vs convex optimization ??

	=> QP is a subset of LP

QP
	=> least square: QP
	=> SVM
	=> QP vs LP


Proof of convexity



















## convex optimization
	Applications:
		=> assets amount in portfolio with expected return minimize risk, airplane control surface deflections, optimal sizing of circult design, power allocated to wireless transmitter, where to transmit signal, data center staff bandwidth memory to allocate, staff scheduling etc.
		=> engineering design: x represents a design e.g. circuit, device, structure, e.g. steer frame of a building, cross section area of beams of the building => optimize for a combination of cost, weight, power and other performance requirements
		=> x could also represents parameters in a model in machine learning model, constraints impose requirements on model parameters e.g. non negative if source of admission, parameters has to be positive-semidefinite if represents covariance


	Formal definition:
		=> stephen boyd https://www.youtube.com/watch?v=C7gZzhs6JMk
		=>  equality constraints have to be linear, affine function
		=> constraint function have to be convex i.e. satisfy jensen inequality

	very like information theory

	most optimization problem is intractable. convex optimization is tractable.

	convexity is in large part what's driving computational tractability of LP

	LP n duality vs convex optimization
		=> LP duality not going to work outside of linear constraints 
		=> transition from linear to quadratic 
		=> lagadrian as another to express dual in convex constraints
		=> soft margin primal svm, dual svm (Alexander Ihler, virginia tech)
		=> KKT dual, optimizing over lagrange multipler
		=> lagrangian optimization (how to transition from LP to lagrangian??)
		=> lagrangian quantifies misses (linear) instead of LP strictly prohibits miss (or lambda/cost is infinity)
		=> gives convexity even original problem is not convex
		=> parameterized lower bound on the optimal value of the problem (certificate proving optimal n quit)
		=> use dual form to solve for nu
		=> dual when data points smaller than num of features
		=> large margin princple
		=> trade off between margin and error


	Dual
		weak duality => difficult problem, non trivial lower bound
		=> if not convex, gap between primal and dual optimal 
		WHY ??
		=> simpler, faster, fewer constraints, for many variables few constraints case (R10M problem, 10 constraints => dual only R10, 10 constraints => as most variables have no constraints on them )
		=> dual is guaranteed to be concave even if primal is convex
		=> solution of lasso is stable
		=> solve both together, bound how far you are from the result, stopping criteria for algorithm
		=> dual function, everywhere is an underestimate
		
		TO GET DUAL:
		=> lagrangian, lower bound on constraint 
		=> set d lag/dx = 0, solve x
		=> dual: g(u,v) = sub x into orginal primal
		=> solve for dual(concave), project on constraint then projective gradient descent

		

	Lagrangian 
		=> multivariable cal: max obj with constraints, set lagrangian to zero is same as solving for pt w gradient vectors n hence max obj
		=> cost of violating constraints?? lamda * constraints
		=> lambda: lagrange multiplier associated w f(x), also called dual variable


	Lagrange dual function(not necessarily convex)
		=> concave even original function(lagrangian) not convex
		=> parameterized lower bound of optimal value
		=> dual function always concave
		=> standard function dual form
		lagrange: necessary but not sufficient by its own => second order condition





	KKT condition
		=> 1st order necessary condition
		=> lagrangian could only solve equality constraint
		=> test additional condition to ensure optimality
		=> feasible + no direction improves objective or feasible (lagrangian) + complimentary slackness(lagrange * constraints = 0, either 1 needs to be zero at optimal condition) + positive lagrange multiplier
			=> https://www.youtube.com/watch?v=eREvLgRJWrE
			=> https://www.youtube.com/watch?v=ws38Jon_-_E (BYU apmonitor.com)
			=> https://www.youtube.com/watch?v=JTTiELgMyuM
			=> https://www.youtube.com/watch?annotation_id=annotation_674267549&feature=iv&src_vid=JTTiELgMyuM&v=AQWy73cHoIU
		=> sequential quadratic programming (SQP) if inequality constraints or non linear objective function or constraints 
			=> modified version of KKT, form search direction, form quadratic approximation of the problem, sequential solve those by getting better n better direction i.e. constraint/KKT1 feasbility test likely violated, set constraint to be inactive, set lamda = 0
			=> intuition: see which constraint is active 

		=> inverse matrix == solve for x

	conjugate functions
		=> 
		=> 


	LP vs linear regression vs SVM vs lasso vs regression
		=> difference algorithm but all instances of same thing - stephen boyd
	

	Convex function ? why lazer treatment is convex problem ?
		=> convexity definition: jensen inequality
			=> intuitively meaning there is only 1 global optimal + closer n closer to optimal
		=> affine function is convex
		=> convex example: 
			=> linear least sqaure, lazer tumor cell
			=> minimize sum of norm of total variation of pixel matrix, guess pixel in corrupted parts of image
			=> SVM convexity is preserved
			=> norm (if vector) or trace (sum of inner product of matrix mulitplication)
			=> log of determinant

	Semi-definite
		=> ellipsoids NOT hyperbola
		=> convex if Hessian matrix is positive semi definite, concave is negative semi definite
			=> hessian matrix is a square matrix of second order partial derivatives of function of x parameters, with row being 1st order differentiate against parameter, col being 2nd order differentiate against parameter
			=> positive semi definite: z.T dot H dot z == positive

			=> Definiteness is a useful measure for optimization. Quadratic forms on positive definite matrices are always positive for non-zero xx and are convex. This is a very desirable property for optimization since it guarantees the existences of maxima and minima, allow you to use the Hessian matrix to optimize multivariate functions
				=> https://math.stackexchange.com/questions/217244/what-is-the-importance-of-definite-and-semidefinite-matrices
			=> 

		=> 

	Convexity proof ?
		=> case by case, not always easy to check whether or not a given function is convex, but there is mature analytic toolbox for this purpose taught in ee364=> common cases: LP, QP, QPQP, semi definite programming etc.

	

	SVM + derive dual form of SVM
		=> show SVM is optimizing contrainted system: quadratic programming with quadratic cost function of parameters n linear constraints
		=> lagar

		=> ref: uc irlvine => 1d example: max margin == max distance between positive and negative + w perpendicular to decision boundary
		=> https://jeremykun.com/2017/06/05/formulating-the-support-vector-machine-optimization-problem/
		=> MIT An Idiot’s guide to Support vector machines (SVMs) + MIT 6034
		=> andrew ng
		=> siraj

		=> proof of distance: https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/
		https://brilliant.org/wiki/dot-product-definition/
		https://mathinsight.org/distance_point_plane



	=> cs168 convex n linear programming + matrix completion, sparse recovery, compression image



## graphical

linear programming == algorithm to solve this problem not doing it mannually !!

second order approximation requires solving linear system at each step








## Max primal is same as min dual
	Motivation
		=> another way to compute the same question
		=> dual is easier becasue constraint tend to be easier
		=> faster if x vector(data dimension) dimension really big n u vector(no of constraints) is small 
	
	proof + intuition + graphical
		=> cmu proof
			=> 2 variables to demo minimizing tightest bound 
			=> general LP i.e. multiple variable
			feasible dual solutions correspond to bounds on the best-possible primal objective function value (derived from taking linear combinations of the constraints), and the optimal dual solution is the tightest-possible such bound. (cs261)

	=> upper bound(bigger than sign), lower bound (smaller than)
	=> 05b: constraints multiplied by u does not change the linear constraint graphically
		=> you cannot just flip the sign, graphically is not the same


	know optimal without graphical
	transpose + computing same
		=> u is ratio test n pivoting in ERO => contribution of rhs of that constraint on obj value (if objective function is linear)
		=> resultant after ERO(picking extreme point) being bigger than objective function but minimize such value
		=> extreme point we pick is outside the constraints
		=> all those u are ERO to pick extreme points (i.e. decision variable is 1 n rhs is that decision's value)
		=> add more portion of decision variable and hence rhs bigger
		=> in 2 d variable case, z (value) is 3rd dimension, if all decision variable more portion then must be higher value
		=> w indicates rhs after ERO (ratio test)

		One way to visualize maximizing primal is equivalent to minimizing dual:
		i.e. max z = min w @6:00

		Given a typical 2 constraints function 2 variable case in 3d space with x_1 being x-axis, x_2 being y-axis, value of objective function is the z-axis,  z-axis has the highest value at optimal extreme point.

		u_1, u_2, u_3 denotes all possible row operations to locate extreme points in original max case. By setting new constraints (u_1+4u_2 - u_3) to more than original objective function value (30) is same as locating extreme points multiplied by arbitrary constant. i.e. locating all points crossing or on original constraints lines. 

		w denotes new value of objective function after located new points (which is constrained to be across constraint lines), minimizing that is same as looking for the same constraint line as original primal problem.

		in short 
		=> new u constraints: different ERO to locate extreme points, that extreme point is more portion than original n hence higher value (if all decision variable is more portion, then value in objective function must be higher)
		=> min w: w is rhs after ERO, minimizing that is same as looking at the same constraint line

		=> cs261 proof: 


	strong law of duality
	complementary slackness
		=> su = ex == 0 if optimal 
		=> if slack == 0 then maximised in that constraint => u has been changed i.e. linear operation happened

	transportation simplex - 
		objective: change 4 variables to satisfy row and col contraints
		u n v are telling how much columns and rows have more cost than your cell
		i.e. motiviation to change such row and col's value => see chain effect on how change of 1 value affects other row n col
		entering variable: highest row and col cost reduction 

		i dont have full proof, this is just for high level intuition:
		To satisfy row and col constraints requirements, we need to change 4 variables.
		u and v are indicators of how much more cost you could reduce if you make a change in its row or col	
		i.e. motivation to make a change in such row or col
		and hence entering variable indicates which row and col to make a change you reduce the most cost
		looping is sanity check to keep both row and col within constraints






## biparttite matching

## l1 minimization





## reference:
https://www.youtube.com/watch?v=C7gZzhs6JMk
[here](http://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/10-dual-lps-scribed.pdf)

LP: lmu + australia math
svm: mit svm for idiots + coursera svm + jess noss + jeremy kunn
duality + kkt: operation research + cmu ryan + stephen














