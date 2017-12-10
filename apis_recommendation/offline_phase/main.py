from tfidf import TFIDF
from cluster import Cluster
from matrices import Matrices
from topN import TopN

# offline phase step 1
TFIDF.gen_vector()
Cluster.gen_vector()

# offline phase step 2
Matrices.gen_matrices()

#offline phase step 3 & step 4
TopN.gen_topN()