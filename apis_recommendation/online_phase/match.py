import os
import pickle
import numpy
from numpy import linalg

class Match(object):
	def __init__(self, TFIDF_vector):
		self.TFIDF_vector = TFIDF_vector

		self.cluster_vectors = []
		for vec_filename in os.listdir("../offline_phase/vector/cluster"):
			with open("../offline_phase/vector/cluster/%s" % vec_filename, "rb") as vec_file:
				self.cluster_vectors.append(pickle.load(vec_file))

	def cosine(self, vector1, vector2):
		numerator = float(numpy.dot(vector1, vector2.T))
		denominator = linalg.norm(vector1) * linalg.norm(vector2)
		return numerator / denominator

	def match(self):
		min_index = 0
		min_cos_dist = self.cosine(self.TFIDF_vector, self.cluster_vectors[min_index])

		for i in range(1, len(self.cluster_vectors)):
			cluster_vector = self.cluster_vectors[i]
			cos_dist = self.cosine(self.TFIDF_vector, cluster_vector)
			if cos_dist < min_cos_dist:
				min_cos_dist = cos_dist
				min_index = i

		return i