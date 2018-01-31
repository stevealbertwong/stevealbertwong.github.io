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


