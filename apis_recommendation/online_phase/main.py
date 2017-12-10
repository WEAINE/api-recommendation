from tfidf import TFIDF
from match import Match
from topN import TopN
import sys

desc = sys.argv[1]

# online phase step 1
tfidf = TFIDF(desc).gen_vector()

# online phase step 2
cluster = Match(tfidf).match()

# online phase step 3
topN = TopN(cluster).get()
for i in topN:
	print(i)