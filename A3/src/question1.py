import os
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE as tsne
from tsnecuda import TSNE
from mpl_toolkits.mplot3d import Axes3D
# from save_embeddings import load_model, wordvec
import random
random.seed(100)

def make_sim_matrix(model_embedding,keys,vocab_data):
	word_to_pos_map = vocab_data['word_to_pos']

	sel_sim_mat = np.zeros((len(keys),model_embedding.shape[0]))


	dot_pdt = np.dot(model_embedding,model_embedding.T)
	# embed_norm = np.linalg.norm(dot_pdt,axis=0,ord=2)
	# print(embed_norm.shape)
	# sim_mat = dot_pdt/embed_norm[:,None]

	for i,word in enumerate(keys):
		idx = word_to_pos_map[word]
		sel_sim_mat[i,:] = dot_pdt[idx,:]

	return sel_sim_mat




def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# tsne_plot_similar_words('Similar words from Google News', keys, embeddings_en_2d, word_clusters, 0.7,
#                         'similar_words.png')

def tsne_plot_2d(label, embeddings,save_name, words=[],a=1):
	plt.figure(figsize=(16, 9))
	colors = cm.rainbow(np.linspace(0, 1, 1))
	x = embeddings[:,0]
	y = embeddings[:,1]
	plt.scatter(x, y, c=colors, alpha=a, label=label)
	for i, word in enumerate(words):
		plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2), 
			textcoords='offset points', ha='right', va='bottom', size=10)
	plt.legend(loc=4)
	plt.grid(True)
	plt.savefig(save_name, format='png', dpi=150, bbox_inches='tight')
	# plt.show()

with open('vocab_data.pkl','rb') as f:
	vocab_data = pickle.load(f)

pos_to_word_map = vocab_data['pos_to_word']


def tsne_plot_3d(title, label, embeddings, a=1):
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = cm.rainbow(np.linspace(0, 1, 1))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=colors, alpha=a, label=label)
    plt.legend(loc=4)
    plt.title(title)
    plt.show()

vocab_size=31024
# base_model_dir = '/media/tiwari/My Passport/lokender/anish/MCA_HW3/src/2020-04-27_20-18-30/'
base_model_dir = '/media/tiwari/My Passport/lokender/anish/MCA_HW3/src/2020-05-06_18-45-23/'


# sel_epoch=10
# keys = ['pool','Australian','years','Perth','climate','water','study']
# model_embed = np.load(base_model_dir+'epoch_'+str(sel_epoch)+'_embedding.npy')
# sim_mat = make_sim_matrix(model_embed,keys,vocab_data)

# embedding_clusters = []
# word_clusters = []
# topn=30
# for i,word in enumerate(keys):
#     embeddings = []
#     words = []
#     word_sim_idxs =  np.argsort(-sim_mat[i,:])[:topn]
#     for similar_word_idx in word_sim_idxs:
#     	sim_word = pos_to_word_map[similar_word_idx]
#     	words.append(sim_word)
#     	embeddings.append(model_embed[similar_word_idx,:])
#     	print(word,sim_word)
#     embedding_clusters.append(embeddings)
#     word_clusters.append(words)

# print(word_clusters)
# embedding_clusters = np.array(embedding_clusters)
# n, m, k = embedding_clusters.shape
# tsne_model_en_2d = TSNE(perplexity=30, n_components=2, init='random', n_iter=3500,verbose=False)
# embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

# tsne_plot_similar_words('Similar words from ABC Corpus', keys, embeddings_en_2d, word_clusters, 0.7,'similar_words.png')
#                         'similar_words.png')
# print(sim_mat.shape)



sel_words = random.sample(list(np.arange(vocab_size)),1500)
# for epoch in range(16):
# 	model_embeds = np.load(base_model_dir+'epoch_'+str(epoch)+'_embedding.npy')
# 	# print(model_embeds.shape)
# 	# model_embeds = model_embeds[sel_words,:]
# 	# print(model_embeds.shape,sel_words[:10])
# 	img_save_name2d = base_model_dir+'epoch_'+str(epoch)+'_tsne.png'
# 	img_save_name3d = base_model_dir+'epoch_'+str(epoch)+'_tsne3D.png'
# 	# model_embeds = model_embeds[:1000,:]

# 	tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='random', n_iter=3500,verbose=True)
# 	embeddings_ak_2d = tsne_ak_2d.fit_transform(model_embeds)
# 	tsne_plot_2d('ABC Corpus',embeddings_ak_2d,img_save_name2d,a=0.1)


for epoch in range(16):
	if epoch==5:
		model_embeds = np.load(base_model_dir+'epoch_'+str(epoch)+'_embedding.npy')
		print(model_embeds.shape)
		model_embeds = model_embeds[sel_words,:]
		img_save_name3d = base_model_dir+'epoch_'+str(epoch)+'_tsne3D.png'

		tsne_wp_3d = tsne(perplexity=30, n_components=3, init='pca', n_iter=3500, random_state=12, verbose=True,n_jobs=8)
		embeddings_wp_3d = tsne_wp_3d.fit_transform(model_embeds)
		tsne_plot_3d('Visualizing Embeddings using t-SNE ', 'ABC_Corpus', embeddings_wp_3d, a=0.5)
	else:
		continue
