import numpy
import pickle
import os

class TopN():
	def __init__(self):
		self.alpha = 0.2
		self.beta = 0.2
		self.gamma = 0.2
		self.mu = 0.2
		self.eta = 0.2

		self.N = 10

		with open("./matrix/S_M.mat", "rb") as S_M_file:
			self.S_M = numpy.mat(pickle.load(S_M_file))
		with open("./matrix/S_A.mat", "rb") as S_A_file:
			self.S_A = numpy.mat(pickle.load(S_A_file))
		with open("./matrix/S_R.mat", "rb") as S_R_file:
			self.S_R = numpy.mat(pickle.load(S_R_file))
		with open("./matrix/Y.mat", "rb") as Y_file:
			self.Y = numpy.mat(pickle.load(Y_file))
		with open("./matrix/Z.mat", "rb") as Z_file:
			self.Z = numpy.mat(pickle.load(Z_file))

		self.m = self.S_M.shape[0]
		self.n = self.S_A.shape[0]

		self.I_m = numpy.mat(numpy.eye(self.m))
		self.I_n = numpy.mat(numpy.eye(self.n))

		self.F = (1 - self.beta - self.eta) * self.I_m - self.alpha * self.S_M
		self.G = (1 - self.alpha - self.mu) * self.I_n - self.beta * self.S_A

		self.api_dict = dict()
		pos = 0
		for api_vec_filename in os.listdir("./vector/api"):
			self.api_dict[pos] = api_vec_filename[:-4]
			pos += 1

	def gen_topN(self):
		for i in range(self.m):
			y = self.Y[i].T
			z = self.Z[i].T

			f = (self.F - self.gamma * self.gamma * self.S_R * self.G.I * self.S_R.T).I * (self.gamma * self.eta * self.S_R * self.G.I * z + self.mu * y)
			g = self.G.I * (self.gamma * self.S_R.T * f - self.eta * z)

			topN_indices = numpy.argsort(numpy.array(g.T))[0][-10:]
			with open("./topN/%d.topN" % i, "a", encoding = "utf-8") as topN_file:
				for j in range(9, -1, -1):
					topN_file.write("%s\n" % self.api_dict[topN_indices[j]])