import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy

class TFIDF(object):
	def __init__(self, input_desc):
		self.punctuations = set()
		self.stopwords = set()
		self.desc = []
		self.bag_of_words = []
		self.input_desc = input_desc

		with open("../offline_phase/punctuations.list", "r", encoding = "utf-8") as punctuations_list:
			punctuation = punctuations_list.readline()
			while punctuation:
				self.punctuations.add(punctuation[:-1])
				punctuation = punctuations_list.readline()

		with open("../offline_phase/stopwords.list", "r", encoding = "utf-8") as stopwords_list:
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

	def load_desc(self):
		for T in ["api", "mashup"]:
			for filename in os.listdir("../offline_phase/data/%s" % T):
				with open("../offline_phase/data/%s/%s" % (T, filename), "r", encoding = "utf-8") as info_file:
					self.desc.append(info_file.readline()[:-1])
		self.desc.append(self.input_desc)

	def clean_desc(self):
		clean_desc = []
		for dirty_desc in self.desc:
			words_list = dirty_desc.split(" ")
			for i in range(len(words_list)):
				words_list[i] = self.del_punctuation(words_list[i])
			words_list = self.del_stopwords(words_list)
			clean_desc.append(" ".join(words_list))
		self.desc = clean_desc

	def gen_TFIDF(self):
		vectorizer = CountVectorizer()
		transformer = TfidfTransformer()
		weight = transformer.fit_transform(vectorizer.fit_transform(self.desc)).toarray()
		bag_of_words = vectorizer.get_feature_names()
		
		self.bag_of_words = bag_of_words
		self.weight = weight

	def gen_vector(self):
		self.load_desc()
		self.clean_desc()
		self.gen_TFIDF()

		vector_list = []
		for i in range(len(self.bag_of_words)):
			vector_list.append(self.weight[len(self.desc) - 1][i])
		return numpy.array(vector_list)