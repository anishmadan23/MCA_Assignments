import nltk.corpus as corp 
import re
import os
import numpy as np 
import argparse
import pickle
import random
random.seed(100)
import time 
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader_no_neg import LoadCCPairs
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from torch.utils.data import DataLoader
import copy
import logging



parser = argparse.ArgumentParser()
parser.add_argument('--window_size', default=3 , type=int, help='Window size for corpus')
parser.add_argument('--base_store_dir',default='/media/tiwari/My Passport/lokender/anish/MCA_HW3/src/')
parser.add_argument('--batch_size',default=16,type=int)
parser.add_argument('--hidden_size',default=128,type=int)
parser.add_argument('--num_epochs',default=20,type=int)
parser.add_argument('--resume',action='store_true',default=False)
parser.add_argument('--restore_model',default=None,help='Path to model checkpoint')
parser.add_argument('--sample_size',default=5,help='Num samples to consider for negative sampling')
parser.add_argument('--overwrite',action='store_true',default=False)
parser.add_argument('--lr',default=1e-3,type=float)
args = parser.parse_args()

store_dir = os.path.join(args.base_store_dir,time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
os.makedirs(store_dir,exist_ok=True)
corp_raw = corp.abc.raw()
vocab = corp.abc.words()

with open(os.path.join(store_dir, os.path.join(store_dir,'args.json')), 'w+') as args_file:
        json.dump(args.__dict__, args_file)

log_format = '%(levelname)-8s %(message)s'

log_file_name = 'train.log'

logfile = os.path.join(store_dir, log_file_name)
logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
logging.getLogger().addHandler(logging.StreamHandler())
logging.info(json.dumps(args.__dict__))

def preprocess_text(text):
	my_text = []
	text = text.split('\n')
	for idx,line_text in enumerate(text):
		line_text = line_text.strip()
		# line_text = re.sub('[^a-zA-Zа-яА-Я1-9<>]+', ' ', line_text)
		# line_text = re.sub("[^*$<,>?!']*$",'\'',line_text)
		line_text = re.sub("[*0-9*]+","<num> ",line_text)
		line_text = re.sub("[^a-zA-Z<> ]+"," ",line_text)
		my_tokenized_line = [x for x in line_text.split() if x!='']

		my_text.append(my_tokenized_line)

	return my_text

def compute_neg_sample_dist(vocab_data):
	pos_to_count_map = vocab_data['pos_to_count']
	size_of_vocab = len(vocab_data['vocab'])

	pos_count_vals = [pos_to_count_map[i] for i in range(size_of_vocab)]
	denom_norm = np.sum(np.array(pos_count_vals)**0.75)

	prob_dist = OrderedDict()
	for pos,count in pos_to_count_map.items():
		prob_dist[pos] = (count**0.75)/denom_norm

	print('Check if sum =1',np.sum(np.array(list(prob_dist.values()))))
	return prob_dist



def makeVocab(corp_raw):
	all_token_sentences = preprocess_text(corp_raw)

	vocabulary = []
	all_words = [x for sentence in all_token_sentences for x in sentence]

	vocabulary, word_freq = np.unique(all_words,return_counts=True)
	vocabulary = list(vocabulary)

	pos_to_word_map = OrderedDict()
	word_to_pos_map = OrderedDict()
	pos_to_count_map = OrderedDict()
	word_to_count_map = OrderedDict()
	for idx,val in enumerate(vocabulary):
		pos_to_word_map[idx] = val
		word_to_pos_map[val] = idx
		pos_to_count_map[idx] = word_freq[idx]
		word_to_count_map[val] = word_freq[idx]

	vocab_data = OrderedDict()
	vocab_data['vocab'] = vocabulary
	vocab_data['pos_to_word'] = pos_to_word_map
	vocab_data['word_to_pos'] = word_to_pos_map
	vocab_data['pos_to_count'] = pos_to_count_map
	vocab_data['word_to_count'] = word_to_count_map


	with open('vocab_data.pkl','wb') as f:
		pickle.dump(vocab_data,f)


	return all_token_sentences,vocab_data

def makeContextCentrePairs(tokenized_sentences,word_to_pos_map):
	cc_train_pairs = []
	cc_val_pairs = []
	for idx,sentence in enumerate(tokenized_sentences):
		print(idx)
		cc_pairs = []
		for i,centre_word in enumerate(sentence):
			
			centre_word_idx = word_to_pos_map[centre_word]
			start_idx = max(0,i-args.window_size)
			end_idx = min(len(sentence)-1,i+args.window_size)
			if i==0:
				context_words = sentence[i+1:end_idx+1]
			elif i==len(sentence)-1:
				context_words = sentence[start_idx:i]
			else:
				context_words = sentence[start_idx:i]
				context_words.extend(sentence[i+1:end_idx+1])

			context_words_pos = [word_to_pos_map[context_word] for context_word in context_words]
			# centre_words = [centre_word]*len(context_words)
			centre_words_pos = [centre_word_idx]*len(context_words)

			# pairs = list(zip(centre_words,context_words))
			pairs_idx = list(zip(centre_words_pos,context_words_pos))
			# print(pairs)
			# print(pairs_idx)
			cc_pairs.extend(pairs_idx)

		if random.random()<0.9:
			cc_train_pairs.extend(cc_pairs)
		else:
			cc_val_pairs.extend(cc_pairs)

	return cc_train_pairs,cc_val_pairs



def save_cc_pairs(corp_raw,overwrite=args.overwrite):
	if overwrite or (not os.path.exists(args.base_store_dir+'cc_pairs.pkl')): #or not os.path.exists(args.base_store_dir+'neg_samples_size_'+str(args.sample_size)+'.pkl')):
		all_token_sentences,vocab_data = makeVocab(corp_raw)
		cc_train_pairs,cc_val_pairs = makeContextCentrePairs(all_token_sentences,vocab_data['word_to_pos'])
		# neg_samples = get_negative_sample(args.sample_size,vocab_data)
		# with open('neg_samples_size_'+str(args.sample_size)+'.pkl','wb') as f:
		# 	pickle.dump(neg_samples,f)
		cc_pairs = OrderedDict()
		cc_pairs['train'] = cc_train_pairs
		cc_pairs['val'] = cc_val_pairs

		with open('cc_pairs.pkl','wb') as f:
			pickle.dump(cc_pairs,f)
	else:
		print('Already saved')

def load_cc_pairs(corp_raw,overwrite=args.overwrite):
	if os.path.exists(args.base_store_dir+'cc_pairs.pkl') and not overwrite: #and os.path.exists(args.base_store_dir+'neg_samples_size_'+str(args.sample_size)+'.pkl'):
		with open(args.base_store_dir+'cc_pairs.pkl','rb') as f:
			cc_pairs = pickle.load(f)

		# with open(args.base_store_dir+'neg_samples_size_'+str(args.sample_size)+'.pkl','rb') as ff:
		# 	neg_samples = pickle.load(ff)

		return cc_pairs#,neg_samples

	else:
		save_cc_pairs(corp_raw)
		args.overwrite=False
		# cc_pairs,neg_samples = load_cc_pairs(corp_raw)
		cc_pairs = load_cc_pairs(corp_raw,overwrite=args.overwrite)


	return cc_pairs#,neg_samples


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

def train():
	since = time.time()
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# device=torch.device('cpu')
	print(device)
	cc_pairs = load_cc_pairs(corp_raw)
	with open(args.base_store_dir+'vocab_data.pkl','rb') as f:
		vocab_data = pickle.load(f)
	vocab_size = len(vocab_data['vocab'])
	print('Vocab Size',vocab_size)
	cc_train = cc_pairs['train']
	cc_val = cc_pairs['val']
	
	# word_prob_dist = compute_neg_sample_dist(vocab_data)

	print('Loading Train-val Context-Centre Pairs ')
	train_set = LoadCCPairs(cc_train,vocab_size)#,word_prob_dist,args.sample_size)
	val_set = LoadCCPairs(cc_val,vocab_size)#,word_prob_dist,args.sample_size)
	print('Loaded')

	print('Loading DataLoader')
	train_loader = DataLoader(train_set, batch_size=args.batch_size,shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_set, batch_size=args.batch_size,shuffle=True, num_workers=4, pin_memory=True)

	print('Loaded')
	criterion = nn.CrossEntropyLoss()

	model = Word2Vec(vocab_size,args.hidden_size)
	model.to(device)
	if args.resume:
		model_checkpoint = torch.load(args.restore_model, map_location=device)
		model.load_state_dict(model_checkpoint['state_dict'])

	optimizer = optim.Adam(model.parameters(),args.lr,betas=(0.9, 0.999))
	scheduler = ReduceLROnPlateau(optimizer, 'min',patience=500,factor=0.8,min_lr=1e-7)
	dataset_sizes = {}
	dataset_sizes['train'] = len(train_set)
	dataset_sizes['val'] = len(val_set)

	best_loss = 99999
	best_model = copy.deepcopy(model.state_dict)

	for epoch in range(args.num_epochs):

		for phase in ['train','val']:
			if phase=='train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			if phase=='train':
				loader = train_loader
			else:
				loader = val_loader
			for step,data in enumerate(loader):
				inp_vec, target_vec = data[0].to(device), data[1].to(device)
				inp_vec.squeeze_(1)
				target_vec.squeeze_(1)
				target_vec = target_vec.long()
				# print(inp_vec.shape,target_vec.shape,neg_vec.shape)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase=='train'):
					output = model(inp_vec)
					# op_inp_vec = model(inp_vec)
					# op_neg_vec = model(neg_vec)

					# int_med1 = torch.bmm(op_inp_vec,target_vec.permute(0,2,1)).squeeze_(2)
					# # print('Int med',int_med1.shape)
					# loss1 = torch.sum(F.logsigmoid(int_med1))
					# # print(target_vec.shape)
					# loss2 = torch.sum(F.logsigmoid(-torch.bmm(op_neg_vec,target_vec.permute(0,2,1)).squeeze_(2)),(0,1))
					# # print('Loss1:{}, Loss2:{}'.format(loss1,loss2))
					# loss = -(loss1+loss2)/inp_vec.size(0)


					# print('Output shape',output.shape)
					# print(target_vec.type(),target_vec.shape,output.shape)
					_,pred = torch.max(output,1)
					loss = criterion(output,target_vec)
					if step%10==0:
						logging.info('Phase:{}, Epoch:{}/{}, LR:{}, Step:{}/{}, Loss:{:.4f}'.format(phase,epoch,args.num_epochs,optimizer.param_groups[0]['lr'],step,dataset_sizes[phase]//args.batch_size,loss))
					if phase == 'train':
						loss.backward()
						optimizer.step()
						scheduler.step(loss)

				running_loss += loss.item() * inp_vec.size(0)


			epoch_loss = running_loss / dataset_sizes[phase]
			logging.info('-'*80)
			logging.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

                        # deep copy the model
			if phase == 'val': #and epoch_loss<best_loss:
				best_loss = epoch_loss
				torch.save({'epoch': epoch,
					'loss':epoch_loss,
					'state_dict': model.state_dict()},
			        os.path.join(store_dir, 'epoch_'+str(epoch)+'_model_checkpoint.pt'))

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Loss: {:4f}'.format(best_loss))

if __name__ =='__main__':
	train()



	
















