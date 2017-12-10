import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
import pickle

class TFIDF():
	def __init__(self):
		self.punctuations = set()
		self.stopwords = set()
		self.desc_dict = {"type": [], "name": [], "dirty_desc": [], "clean_desc": []}
		self.bag_of_words = []

		with open("./punctuations.list", "r", encoding = "utf-8") as punctuations_list:
			punctuation = punctuations_list.readline()
			while punctuation:
				self.punctuations.add(punctuation[:-1])
				punctuation = punctuations_list.readline()

		with open("./stopwords.list", "r", encoding = "utf-8") as stopwords_list:
			stopword = stopwords_list.readline()
			while stopword:
				self.stopwords.add(stopword[:-1])
				stopword = stopwords_list.readline()

	def del_punctuation(self, dirty_word):
		clean_word = dirty_word
		for _ in range(3):
			if clean_word[-1:] in self.punctuations:
				clean_word = clean_word[:-1]
		return clean_word

	def del_stopwords(self, dirty_wordlist):
		clean_wordlist = []
		for word in dirty_wordlist:
			if word not in self.stopwords:
				for i in range(2,4):
					if word[-i:] in self.stopwords:
						word = word[:-i]
				clean_wordlist.append(word)
		return clean_wordlist

	def load_desc_dict(self):
		for T in ["api", "mashup"]:
			for filename in os.listdir("./data/%s" % T):
				name = filename[:-5]
				with open("./data/%s/%s" % (T, filename), "r", encoding = "utf-8") as info_file:
					self.desc_dict["type"].append(T)
					self.desc_dict["name"].append(name)
					self.desc_dict["dirty_desc"].append(info_file.readline()[:-1])

	def clean_desc(self):
		for dirty_desc in self.desc_dict["dirty_desc"]:
			words_list = dirty_desc.split(" ")
			for i in range(len(words_list)):
				words_list[i] = self.del_punctuation(words_list[i])
			words_list = self.del_stopwords(words_list)
			clean_desc = " ".join(words_list)
			self.desc_dict["clean_desc"].append(clean_desc)

	def gen_TFIDF(self):
		vectorizer = CountVectorizer()
		transformer = TfidfTransformer()
		weight = transformer.fit_transform(vectorizer.fit_transform(self.desc_dict["clean_desc"])).toarray()
		bag_of_words = vectorizer.get_feature_names()
		
		self.bag_of_words = bag_of_words
		self.weight = weight

	def gen_vector(self):
		self.load_desc_dict()
		self.clean_desc()
		self.gen_TFIDF()

		for i in range(len(self.desc_dict["clean_desc"])):
			T = self.desc_dict["type"][i]
			name = self.desc_dict["name"][i]
			desc = self.desc_dict["clean_desc"][i]
			vector_list = []
			for j in range(len(self.bag_of_words)):
				vector_list.append(self.weight[i][j])
			with open("./vector/%s/%s.vec" % (T, name), "wb") as vector_file:
				vector = numpy.array(vector_list)
				pickle.dump(vector, vector_file)