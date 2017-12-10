class TopN(object):
	def __init__(self, match_result):
		self.topN_filename = "../offline_phase/topN/%d.topN" % match_result

	def get(self):
		with open(self.topN_filename, "r", encoding = "utf-8") as topN_file:
			line = topN_file.readline()
			while line:
				api = line[:-1]
				yield api
				line = topN_file.readline()