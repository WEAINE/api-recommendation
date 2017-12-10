import os
from tfidf import TFIDF
from match import Match
from topN import TopN

def get_pred_api_set(desc):
	tfidf = TFIDF(desc).gen_vector()
	cluster = Match(tfidf).match()
	topN = TopN(cluster).get()
	return set(topN)

recalls = []
for filename in os.listdir("../offline_phase/data/mashup"):
	with open("../offline_phase/data/mashup/%s" % filename, "r", encoding = "utf-8") as mashup_info_file:
		info = mashup_info_file.readlines()
		desc = info[0][:-1]
		apis = set(info[2].split(","))
		pred_apis = get_pred_api_set(desc)

		numerator = len(apis & pred_apis)
		if numerator > 10:
			numerator = 10

		print(numerator)

		recalls.append(numerator / 10)
print(sum(recalls) / len(recalls))