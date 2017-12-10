import pickle
import numpy
from sklearn.cluster import KMeans
import os

class Cluster():
	def __init__(self):
		self.samples = {"names": [], "names_dict": dict(), "X": []}
		self.clusters = dict()
		for i in range(100):
			self.clusters[i] = []

	def load_samples(self):	
		filenames = os.listdir("./vector/mashup")
		for i in range(len(filenames)):
			filename = filenames[i]
			self.samples["names"].append(filename[:-4])
			self.samples["names_dict"][filename[:-4]] = i
			with open("./vector/mashup/%s" % filename, "rb") as vec_file:
				self.samples["X"].append(pickle.load(vec_file))
		self.samples["X"] = numpy.array(self.samples["X"])
		self.vec_len = len(self.samples["X"][0])

	def cluster(self):
		clf = KMeans(n_clusters = 100)
		clf.fit(self.samples["X"])
		labels = clf.labels_
		for i in range(len(labels)):
			name = self.samples["names"][i]
			label = labels[i]
			self.clusters[label].append(name)

	def gen_vector(self):
		self.load_samples()
		self.cluster()
		
		for i in range(100):
			with open("./data/cluster/%d.info" % i, "a", encoding = "utf-8") as info_file:
				with open("./vector/cluster/%d.vec" % i, "wb") as vec_file:
					vector = []
					for name in self.clusters[i]:
						info_file.write("%s\n" % name)
						vector.append(self.samples["X"][self.samples["names_dict"][name]])
					vector = numpy.array(vector).sum(axis = 0)
					pickle.dump(vector, vec_file)