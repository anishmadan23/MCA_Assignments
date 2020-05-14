import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
	"""
		relevance feedback
		Parameters
		----------
		vec_docs: sparse array,
		    tfidf vectors for documents. Each row corresponds to a document.
		vec_queries: sparse array,
		    tfidf vectors for queries. Each row corresponds to a document.
		sim: numpy array,
		    matrix of similarities scores between documents (rows) and queries (columns)
		n: integer
		    number of documents to assume relevant/non relevant
		Returns
		-------
		rf_sim : numpy array
		matrix of similarities scores between documents (rows) and updated queries (columns)
		"""
	beta= 0.65
	alpha = 1.0

	new_vec_queries = np.zeros(vec_queries.shape)
	# print('NVQ',new_vec_queries.shape)
	# for epoch in range(3):
	for query_idx in range(vec_queries.shape[0]):
		orig_query = vec_queries[query_idx,:]
		sim_scores = sim[:,query_idx]
		# print('Sim scores',sim_scores[:10],sim_scores.shape)
		sim_scores_idx = np.argsort(-sim_scores)
		# print(sim_scores_idx.shape)    
		top_n_sim_idx = sim_scores_idx[:n]

		btm_n_sim_idx = sim_scores_idx[-n:]
		# print(top_n_sim_idx)
		rel_docs = vec_docs[top_n_sim_idx,:]
		nr_docs = vec_docs[btm_n_sim_idx,:]

		sum_rel_docs = np.sum(rel_docs,axis=0)
		sum_nr_docs = np.sum(nr_docs,axis=0)

		# sum_rel_docs = np.squeeze(sum_rel_docs)
		# print(sum_rel_docs.shape)
		rocch_query = orig_query+(alpha*sum_rel_docs)/n - (beta*sum_nr_docs)/n

		new_vec_queries[query_idx,:] = rocch_query
	vec_queries = new_vec_queries
	rf_sim = cosine_similarity(vec_docs,new_vec_queries)

	return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
	"""
	relevance feedback with expanded queries
	Parameters
	    ----------
	    vec_docs: sparse array,
	        tfidf vectors for documents. Each row corresponds to a document.
	    vec_queries: sparse array,
	        tfidf vectors for queries. Each row corresponds to a document.
	    sim: numpy array,
	        matrix of similarities scores between documents (rows) and queries (columns)
	    tfidf_model: TfidfVectorizer,
	        tf_idf pretrained model
	    n: integer
	        number of documents to assume relevant/non relevant

	Returns
	-------
	rf_sim : numpy array
	    matrix of similarities scores between documents (rows) and updated queries (columns)
	"""	
	# print(vec_docs.shape,vec_queries.shape)

	vec_docs = vec_docs.toarray()
	# print(vec_docs.shape)
	auto_thes = np.dot(vec_docs.T,vec_docs)
	# print(auto_thes.shape)
	# auto_thes = np.array(auto_thes)
	auto_thes_norm = np.linalg.norm(auto_thes,axis=1,ord=2)
	# print(auto_thes_norm)
	auto_thes = auto_thes/auto_thes_norm[:,None]

	beta= 0.4
	alpha = 1.0
	num_sim_words=4
	vec_queries = vec_queries.toarray()
	new_vec_queries = np.zeros(vec_queries.shape)

	for query_idx in range(vec_queries.shape[0]):

		orig_query = vec_queries[query_idx,:]
		# print('orig_query shape',orig_query.shape)
		sim_scores = sim[:,query_idx]
		sim_scores_idx = np.argsort(-sim_scores)
		top_n_sim_idx = sim_scores_idx[:n]
		btm_n_sim_idx = sim_scores_idx[-n:]

		rel_docs = vec_docs[top_n_sim_idx,:]
		nr_docs = vec_docs[btm_n_sim_idx,:]

		sum_rel_docs = np.sum(rel_docs,axis=0)
		sum_nr_docs = np.sum(nr_docs,axis=0)


		rocch_query = orig_query+(alpha*sum_rel_docs)/n - (beta*sum_nr_docs)/n
		max_tfidf_idx = np.argmax(rocch_query)	
		max_tfidf_val = rocch_query[max_tfidf_idx]

		sim_arr = auto_thes[max_tfidf_idx,:]
		
		sim_arr_idx = np.argsort(-sim_arr)
		# print(sim_arr_idx)
		top_sim_idxs = sim_arr_idx[1:num_sim_words+1]


		# for x in top_sim_idxs:

		rocch_query[top_sim_idxs] = max_tfidf_val

		new_vec_queries[query_idx] = rocch_query
	vec_queries = new_vec_queries

	# print(vec_docs.shape,vec_queries.shape)


	rf_sim = np.dot(vec_docs,vec_queries.T)
	# print(rf_sim.shape)
	# print(auto_thes.shape)
	# auto_thes = np.array(auto_thes)
	rf_norm = np.linalg.norm(rf_sim,axis=1,ord=2)
	# print(auto_thes_norm)
	rf_sim = rf_sim/rf_norm[:,None]
	# rf_sim = cosine_similarity(vec_docs,vec_queries)

	return rf_sim