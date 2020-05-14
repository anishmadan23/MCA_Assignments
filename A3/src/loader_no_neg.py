import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import numpy as np 


# def get_negative_sample(sample_size,prob_dist,vocab_size):
# 	# print(sample_size,list(prob_dist.values())[:10],vocab_size)
# 	neg_sample = np.random.choice(vocab_size,sample_size,p=list(prob_dist.values()))
# 	return neg_sample

def one_hot(num,vocab_size):
	t = torch.zeros((1,vocab_size),dtype=torch.float)
	t[:,num] = 1
	return t

class LoadCCPairs(Dataset):

	def __init__(self,cc_pairs,vocab_size):
		self.cc_pairs = cc_pairs
		self.vocab_size = vocab_size
		# self.word_prob_dist = word_prob_dist
		# self.sample_size = sample_size

	def __len__(self):
		return len(self.cc_pairs)

	def __getitem__(self,idx):	
		cc_pair = self.cc_pairs[idx]
		input_token = one_hot(cc_pair[0],self.vocab_size)

		# my_neg_samples = get_negative_sample(self.sample_size,self.word_prob_dist,self.vocab_size)
		# neg_tsor = torch.zeros((len(my_neg_samples),self.vocab_size),dtype=torch.float)
		# for idx,x in enumerate(my_neg_samples):
		# 	neg_tsor[idx,:] = one_hot(x,self.vocab_size)

		target_token = torch.LongTensor([cc_pair[1]])
		# target_token = one_hot(cc_pair[1],self.vocab_size)

		return input_token,target_token#,neg_tsor