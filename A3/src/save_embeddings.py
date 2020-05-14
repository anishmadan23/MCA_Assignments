import torch
import argparse
import os
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import argparse
import torch
import torch.nn as nn
# from train import Word2Vec
import random


parser = argparse.ArgumentParser()
parser.add_argument('--num_sim_words', default=10 , type=int, help='Number of similar words to visualize for given words')

parser.add_argument('--base_model_dir',default='/media/tiwari/My Passport/lokender/anish/MCA_HW3/src/2020-04-27_20-18-30/',type=str,help='Base_path for model to load from')
args = parser.parse_args()

keys = ['Australia','water','researchers','University','million','farmers','South','time','team','different']

with open('vocab_data.pkl','rb') as f:
	vocab_data = pickle.load(f)

class Word2Vec(nn.Module):
	def __init__(self,vocab_size,hidden_size):
		super(Word2Vec, self).__init__()
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size

		self.linear1 = nn.Linear(vocab_size,hidden_size,bias=False)
		self.linear2 = nn.Linear(hidden_size,vocab_size,bias=False)

	def forward(self,x):
		x = self.linear1(x)
		# print(x.shape)
		x = self.linear2(x)
		return x

class wordvec(torch.nn.Module):
	def __init__(self, eval_model, device='cuda'):
		super(wordvec, self).__init__()
		self.model = eval_model


	def hook_Layer(self,word):
		layer_outputs = {}
		model_layers = ['linear1','linear2']

		def get_layer_op(m,i,o):
			layer_outputs[m_layer] = o.data.detach().clone()

		for m_layer in model_layers:
			h = self.model._modules[m_layer].register_forward_hook(get_layer_op)
			with torch.no_grad():
				m_out = self.model(word)
			h.remove()

		return layer_outputs

def load_model(model,epoch_num):
	model_path = args.base_model_dir+'epoch_'+str(epoch_num)+'_model_checkpoint.pt'
	model_checkpoint = torch.load(model_path, map_location=device)
	model.load_state_dict(model_checkpoint['state_dict'])
	return model

device = torch.device('cuda')

def preprocess(word,word_to_pos_map,vocab_size):
	idx = word_to_pos_map[word]
	my_word = torch.zeros((1,vocab_size),dtype=torch.float)
	my_word[:,idx] = 1.0

	return my_word

def main():
	hidden_size = 128
	vocab_size = len(vocab_data['vocab'])
	base_model = Word2Vec(vocab_size,hidden_size)
	base_model = base_model.to(device)
	base_model.eval()

	word_to_pos_map = vocab_data['word_to_pos']
	# words = random.sample(vocab_data['vocab'],1000)
	words = vocab_data['vocab']
	my_words = []
	for word in words:
		my_words.append(preprocess(word,word_to_pos_map,vocab_size))
	for epoch in range(16): # 16
		model = load_model(base_model,epoch)
		w2v = wordvec(model,device)
		epoch_embeds = []
		for word in my_words:
			word = word.to(device)
			layer_ops = w2v.hook_Layer(word)
			embedding = layer_ops['linear1']
			# print(embedding.shape)
			epoch_embeds.append(embedding.cpu().numpy())
		epoch_embeds = np.concatenate(np.array(epoch_embeds),axis=0)
		print('Saving embedding representation for epoch:{} with shape:{}'.format(epoch,epoch_embeds.shape))
		save_name = 'epoch_'+str(epoch)+'_embedding.npy'
		np.save(args.base_model_dir+save_name,epoch_embeds)

if __name__=='__main__':
	main()