import os
import numpy
from numpy import linalg
import pickle

class Matrices():
	def __init__(self):
		self.clusters_vec_filenames = os.listdir("./vector/cluster")
		self.api_vec_filenames = os.listdir("./vector/api")

		self.m = len(self.clusters_vec_filenames)
		self.n = len(self.api_vec_filenames)

		self.api_dict = dict()
		for i in range(self.n):
			api = self.api_vec_filenames[i][:-4]
			self.api_dict[api] = i

		self.M = numpy.zeros((self.m, self.m))
		self.A = numpy.zeros((self.n, self.n))
		self.R = numpy.zeros((self.m, self.n), dtype = numpy.int)

		self.D_M = numpy.zeros((self.m, self.m))
		self.D_A = numpy.zeros((self.n, self.n))
		self.D_RM = numpy.zeros((self.m, self.m), dtype = numpy.int)
		self.D_RA = numpy.zeros((self.n, self.n), dtype = numpy.int)

		self.Y = numpy.zeros((self.m, self.m), dtype = numpy.int)
		self.Z = numpy.zeros((self.m, self.n), dtype = numpy.int)

	def cosine(self, vector1, vector2):
		numerator = float(numpy.dot(vector1, vector2.T))
		denominator = linalg.norm(vector1) * linalg.norm(vector2)
		return numerator / denominator

	def neg_half_power_of_diagonal_matrix(self, diagonal_matrix):
		rank = diagonal_matrix.shape[0]
		result = numpy.zeros((rank, rank))
		for i in range(rank):
			result[i][i] = 1 / numpy.power(diagonal_matrix[i][i], 0.5)
		return result

	def construct_M(self):
		vectors = []
		for filename in self.clusters_vec_filenames:
			with open("./vector/cluster/%s" % filename, "rb") as vec_file:
				vectors.append(pickle.load(vec_file))

		for i in range(self.m):
			for j in range(self.m):
				self.M[i][j] = self.cosine(vectors[i], vectors[j])

	def construct_A(self):
		vectors = []
		for filename in self.api_vec_filenames:
			with open("./vector/api/%s" % filename, "rb") as vec_file:
				vectors.append(pickle.load(vec_file))

		for i in range(self.n):
			for j in range(self.n):
				self.A[i][j] = self.cosine(vectors[i], vectors[j])

	def construct_R(self):
		for i in range(self.m):
			for j in range(self.n):
				cluster = self.clusters_vec_filenames[i][:-4]
				api = self.api_vec_filenames[j][:-4]
				count = 0

				with open("./data/cluster/%s.info" % cluster, "r", encoding = "utf-8") as cluster_info_file:
					mashup = cluster_info_file.readline()
					while mashup:
						mashup = mashup[:-1]
						with open("./data/mashup/%s.info" % mashup, "r", encoding = "utf-8") as mashup_info_file:
							apis = mashup_info_file.readlines()[2].split(",")
							if api in apis:
								count += 1
						mashup = cluster_info_file.readline()

				self.R[i][j] = count

	def construct_D_M(self):
		for i in range(self.m):
			self.D_M[i][i] = numpy.sum(self.M[i])

	def construct_D_A(self):
		for i in range(self.n):
			self.D_A[i][i] = numpy.sum(self.A[i])

	def construct_D_RM(self):
		for i in range(self.m):
			self.D_RM[i][i] = numpy.sum(self.R[i])

	def construct_D_RA(self):
		RT = self.R.T
		for i in range(self.n):
			self.D_RA[i][i] = numpy.sum(RT[i])

	def construct_Y(self):
		for i in range(self.m):
			self.Y[i][i] = 1

	def construct_Z(self):
		for i in range(self.m):
			with open("./data/cluster/%d.info" % i, "r", encoding = "utf-8") as cluster_info_file:
				mashup = cluster_info_file.readline()
				while mashup:
					mashup = mashup[:-1]
					with open("./data/mashup/%s.info" % mashup, "r", encoding = "utf-8") as mashup_info_file:
						apis = mashup_info_file.readlines()[2].split(",")
						for api in apis:
							self.Z[i][self.api_dict[api]] = 1
					mashup = cluster_info_file.readline()

	def gen_matrices(self):
		self.construct_M()
		self.construct_A()
		self.construct_R()
		self.construct_D_M()
		self.construct_D_A()
		self.construct_D_RM()
		self.construct_D_RA()
		self.construct_Y()
		self.construct_Z()

		S_M = numpy.dot(
			numpy.dot(
				self.neg_half_power_of_diagonal_matrix(self.D_M), 
				self.M
				), 
			self.neg_half_power_of_diagonal_matrix(self.D_M)
			)
		with open("./matrix/S_M.mat", "wb") as S_M_file:
			pickle.dump(S_M, S_M_file)

		S_A = numpy.dot(
			numpy.dot(
				self.neg_half_power_of_diagonal_matrix(self.D_A), 
				self.A
				), 
			self.neg_half_power_of_diagonal_matrix(self.D_A)
			)
		with open("./matrix/S_A.mat", "wb") as S_A_file:
			pickle.dump(S_A, S_A_file)

		S_R = numpy.dot(
			numpy.dot(
				self.neg_half_power_of_diagonal_matrix(self.D_RM), 
				self.R
				), 
			self.neg_half_power_of_diagonal_matrix(self.D_RA)
			)
		with open("./matrix/S_R.mat", "wb") as S_R_file:
			pickle.dump(S_R, S_R_file)

		with open("./matrix/Y.mat", "wb") as Y_file:
			pickle.dump(self.Y, Y_file)

		with open("./matrix/Z.mat", "wb") as Z_file:
			pickle.dump(self.Z, Z_file)