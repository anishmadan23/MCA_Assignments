import os
import numpy as np
from PIL import Image
from PIL import ImageFilter 
import pickle
from sklearn.externals import joblib
from utils import *
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--save_idx',type=int,default=0)
args = parser.parse_args()

def get_neighborhood_score(mask,radius,x_src,y_src):
	# print(mask.shape[1]-1,radius,x_src,y_src)
	topl_x = max(0,x_src-radius)
	topl_y = max(0,y_src-radius)

	btmr_x = min(mask.shape[0]-1,x_src+radius)
	btmr_y = min(mask.shape[1]-1,y_src+radius)
	# print('POints',x_src,y_src,topl_x,topl_y,btmr_x,btmr_y)


	mask_sum = 0
	mask_sum += np.sum(mask[topl_x,topl_y:btmr_y+1])
	mask_sum += np.sum(mask[topl_x:btmr_x+1,topl_y])
	mask_sum += np.sum(mask[btmr_x,topl_y:btmr_y+1])
	mask_sum += np.sum(mask[topl_x:btmr_x+1,btmr_y])

	perimeter = 2*((btmr_y-topl_y)+1 + (btmr_x-topl_x)+1)

	return mask_sum,perimeter


def get_ac_img_feat(kmeans,img,dist,color,color_idx):
	all_cols = img.reshape(-1,img.shape[2])
	all_cols = [all_cols[i,:] for i in range(all_cols.shape[0])]

	kmeans_preds = kmeans.predict(all_cols)
	kmeans_color_pred_img = kmeans_preds.reshape(img.shape[0],img.shape[1])

	mask_img = np.zeros_like(kmeans_color_pred_img)

	mask_img[kmeans_color_pred_img==color_idx] = 1

	mask_indices = list(zip(np.where(mask_img==1)[0],np.where(mask_img==1)[1]))
	num_points = len(mask_indices)
	img_sum = 0
	img_perimeter = 0
	# print('Num Points',num_points)
	if num_points==0:
		return 0
	for idx,(x,y) in enumerate(mask_indices):
		# print('Mask Img',mask_img.shape)

		win_sum,win_pmter = get_neighborhood_score(mask_img,dist,x,y)
		img_sum+=win_sum
		img_perimeter+=win_pmter

	return img_sum/img_perimeter

def bin_img_four(img_np):
	img_np = img_np//4
	img_np = img_np*4+2
	assert(len(list(np.unique(img_np.flatten())))<=64)
	return img_np

def make_ac_feats(kmeans,img_paths,dists_arr,colors_arr,save_name,return_feat=False):
	my_data_dict = {}

	for iidx,img_path in enumerate(img_paths):

		img_name = img_path.split('/')[-1]
		img = Image.open(img_path).convert('RGB')
		img = img.resize((100,100))
		# img = img.filter(ImageFilter.GaussianBlur(radius=3))
		img_np = np.array(img)
		img_np = bin_img_four(img_np)


		my_feat_mat = np.zeros((len(colors_arr),len(dists_arr)))

		for cidx,color in enumerate(colors_arr):
			for didx,dist in enumerate(dists_arr):
				# print('Running Img No:{}, Color No:{}, Dist No:{}'.format(iidx+1,cidx+1,didx+1))
				my_feat_mat[cidx,didx] = get_ac_img_feat(kmeans,img_np,dist,color,cidx)

		my_data_dict[img_name] = my_feat_mat
	if return_feat:
		return my_data_dict

	with open(save_name,'wb') as f:
		pickle.dump(my_data_dict,f)

def parse_eval_dir(base_dir,query_file_name):
	gt_path = base_dir+'ground_truth/'
	query_path = base_dir+'query/'
	gt_files = {}
	gt_files['good'] = []
	gt_files['junk'] = []
	gt_files['ok'] = []

	gt_file_name_good = query_file_name.split('_query')[0]+'_good.txt'
	gt_file_name_ok = query_file_name.split('_query')[0]+'_ok.txt'
	gt_file_name_junk = query_file_name.split('_query')[0]+'_junk.txt'

	with open(query_path+query_file_name,'r') as f:
		line = f.readlines()[0].strip()
		tokens = line.split(' ')

		img_query_name = str(tokens[0])
		bbox_query = [float(tokens[1]),float(tokens[2]),float(tokens[3]),float(tokens[4])]

	with open(gt_path+gt_file_name_good,'r') as f:
		lines = f.readlines()
		for line in lines:
			gt_files['good'].append(line.strip()+'.jpg')

	with open(gt_path+gt_file_name_ok,'r') as f:
		lines = f.readlines()
		for line in lines:
			gt_files['ok'].append(line.strip()+'.jpg')

	with open(gt_path+gt_file_name_junk,'r') as f:
		lines = f.readlines()
		for line in lines:
			gt_files['junk'].append(line.strip()+'.jpg')

	return img_query_name, bbox_query, gt_files

def ac_feat_similarity(img1_feat,stacked_feat):
	img1_feat = np.expand_dims(img1_feat,axis=0)
	one_constt_mat = np.ones_like(stacked_feat)

	sim_score_num = np.abs(img1_feat - stacked_feat)
	sim_score_denom = one_constt_mat+img1_feat+stacked_feat

	m = stacked_feat.shape[1]
	sim_score = (1/m)* np.sum(np.divide(sim_score_num,sim_score_denom),axis=(1,2))           # sum after dividing elt wise division
	print('sim_score shape',sim_score.shape)
	return sim_score

def combine_feats(paths_feats):
	my_new_dict = {}
	my_new_dict['img_names'] = []
	my_new_dict['stacked_ac_feats'] = np.zeros((5063,64,4))
	for ix,x in enumerate(paths_feats):
		with open(x,'rb') as f:
			dd = pickle.load(f)

			for idx,(key,val) in enumerate(dd.items()):
				print(ix,idx)
				my_new_dict['img_names'].append(key)
				my_new_dict['stacked_ac_feats'][idx,:,:] = val

	with open('./binned_ac_dset_feats/ac_dset_feats_bin_combined.pkl','wb') as f:
		pickle.dump(my_new_dict,f)


def query_images(kmeans,colors_arr,dists_arr,stacked_feats_dict_path):


	with open(stacked_feats_dict_path,'rb') as f:
		stacked_dset_feats = pickle.load(f)

	img_names = stacked_dset_feats['img_names']
	stacked_feats = stacked_dset_feats['stacked_ac_feats']
	base_dir = './train/'

	query_files = os.listdir(base_dir+'query/')
	# query_files = query_files[7:10]
	num_files = len(query_files)
	final_scores = {}

	for idx,query_file in enumerate(query_files):
		print('\n')
		print('Running Image :',idx+1)
		l = []
		query_img_name,_, gt_dict = parse_eval_dir('./train/',query_file)
		query_img_name = query_img_name[query_img_name.find('_')+1:]+'.jpg'
		# print(query_img_name)
		# print(len(gt_dict['good']))
		# print(len(gt_dict['ok']))
		# print(len(gt_dict['junk']))

		all_gt_files = np.union1d(gt_dict['good'],gt_dict['ok'])
		all_gt_files = np.union1d(all_gt_files,gt_dict['junk'])
		num_top = len(all_gt_files)
		print('Num Top',num_top)

		img_feat_dict = make_ac_feats(kmeans,['./images/'+query_img_name],dists_arr,colors_arr,'junk_save_name',return_feat=True)
		img_feat = img_feat_dict[query_img_name]
		sim_score = ac_feat_similarity(img_feat,stacked_feats)
		print(np.max(sim_score),np.min(sim_score))
		zipped_arr = list(zip(img_names,sim_score))
		sorted_zipped_score = sorted(zipped_arr,key=lambda x:x[1])

		img_scores = compute_scores(sorted_zipped_score,gt_dict,stacked_feats_dict_path,topk =[200,300,400,500,1000])
		final_scores['query_img_name'] = img_scores

	my_scores_dict = {}
	my_scores_dict['all_min_precision'] = []
	my_scores_dict['all_max_precision'] = []
	my_scores_dict['all_avg_precision'] = []
	my_scores_dict['all_min_recall'] = []
	my_scores_dict['all_max_recall'] = []
	my_scores_dict['all_avg_recall'] = []
	my_scores_dict['all_min_f1'] = []
	my_scores_dict['all_max_f1'] = []
	my_scores_dict['all_avg_f1'] = []
	my_scores_dict['all_good_percentage'] = []
	my_scores_dict['all_ok_percentage'] = []
	my_scores_dict['all_junk_percentage'] = []

	for key,val in final_scores.items():
		my_scores_dict['all_min_precision'].append(val['min_precision'])
		my_scores_dict['all_max_precision'].append(val['max_precision'])
		my_scores_dict['all_avg_precision'].append(val['avg_precision'])
		my_scores_dict['all_min_recall'].append(val['min_recall'])
		my_scores_dict['all_max_recall'].append(val['max_recall'])
		my_scores_dict['all_avg_recall'].append(val['avg_recall'])
		my_scores_dict['all_min_f1'].append(val['min_f1'])
		my_scores_dict['all_max_f1'].append(val['max_f1'])
		my_scores_dict['all_avg_f1'].append(val['avg_f1'])
		my_scores_dict['all_good_percentage'].append(val['good_percentage'])
		my_scores_dict['all_ok_percentage'].append(val['ok_percentage'])
		my_scores_dict['all_junk_percentage'].append(val['junk_percentage'])

	print(my_scores_dict['all_min_precision'])
	print(my_scores_dict['all_max_precision'])
	print(my_scores_dict['all_avg_precision'])
	print('Mean Min Precision',np.mean(my_scores_dict['all_min_precision']))
	print('Mean Max Precision',np.mean(my_scores_dict['all_max_precision']))
	print('Mean Avg Precision',np.mean(my_scores_dict['all_avg_precision']))
	print('Mean Min Recall',np.mean(my_scores_dict['all_min_recall']))	
	print('Mean Max Recall',np.mean(my_scores_dict['all_max_recall']))
	print('Mean Avg Recall',np.mean(my_scores_dict['all_avg_recall']))
	print('Mean Min F1',np.mean(my_scores_dict['all_min_f1']))
	print('Mean Max F1',np.mean(my_scores_dict['all_min_f1']))
	print('Mean Avg F1',np.mean(my_scores_dict['all_min_f1']))
	print('Mean Good',np.mean(my_scores_dict['all_good_percentage']))
	print('Mean Ok',np.mean(my_scores_dict['all_ok_percentage']))		
	print('Mean Junk',np.mean(my_scores_dict['all_junk_percentage']))		

	# return sorted_zipped_score, gt_dict

def compute_scores(sorted_zipped_score,gt_dict,stacked_feats_dict_path,topk):
	precision_dict = {}
	recall_dict = {}
	f1_dict = {}

	for k in topk:
		precision_dict[k],_,_,_ = precision_recall(gt_dict,sorted_zipped_score,k=k)
		recall_dict[k],good_percentage,ok_percentage,junk_percentage = precision_recall(gt_dict,sorted_zipped_score,k=k,recall=True)

		if precision_dict[k]+recall_dict[k]==0:
			f1_dict[k] = 0
		else:
			f1_dict[k] = 2*((precision_dict[k]*recall_dict[k])/(precision_dict[k]+recall_dict[k]))

		print('Precision at k={} is {}'.format(k,precision_dict[k]))
		print('Recall at k={} is {}'.format(k,recall_dict[k]))
		print('F1 score at k={} is {}'.format(k,f1_dict[k]))

	scores = {}
	scores['min_precision'] = min(list(precision_dict.values()))
	scores['max_precision'] = max(list(precision_dict.values()))
	scores['avg_precision'] = np.mean(list(precision_dict.values()))

	scores['min_recall'] = min(list(recall_dict.values()))
	scores['max_recall'] = max(list(recall_dict.values()))
	scores['avg_recall'] = np.mean(list(recall_dict.values())) 

	scores['min_f1'] = min(list(f1_dict.values()))
	scores['max_f1'] = max(list(f1_dict.values()))
	scores['avg_f1'] = np.mean(list(f1_dict.values()))

	scores['good_percentage'] = good_percentage
	scores['ok_percentage'] = ok_percentage
	scores['junk_percentage'] = junk_percentage
	return scores

if __name__=='__main__':
	start=time.time()
	X = load_dset('./images/')	
	imgs_root = './images/.'

	# kmeans_file = './kmeans_quant_obj.pkl'
	kmeans_file = './kmeans_quant_obj_binned_four_64.pkl'
	kmeans = joblib.load(kmeans_file)

	colors_arr = kmeans.cluster_centers_
	dists_arr = [1,3,5,7]


	# sorted_zipped_score,all_gt_files = query_images(kmeans,colors_arr,dists_arr,stacked_feats_dict_path)
	# print(sorted_zipped_score[:20],'\n')
	# print(all_gt_files[:20])

	# save_idx_map = {0:[0,1000],1:[1000,2000],2:[2000,3000],3:[3000,4000],4:[4000,5063]}
	new_save_dir = './binned_ac_dset_feats/'
	# save_name = 'ac_dset_feats_'+str(args.save_idx)+'.pkl'

	# new_save_name = new_save_dir+save_name
	# os.makedirs(new_save_dir,exist_ok=True)

	# img_paths = glob.glob(imgs_root+'*jpg')[save_idx_map[args.save_idx][0]:save_idx_map[args.save_idx][1]]

	# make_ac_feats(kmeans,img_paths,dists_arr,colors_arr,new_save_name)

	# saved_feats= glob.glob(new_save_dir+'*.pkl')
	# combine_feats(saved_feats)

	stacked_feats_dict_path = './binned_ac_dset_feats/ac_dset_feats_bin_combined.pkl'
	query_images(kmeans,colors_arr,dists_arr,stacked_feats_dict_path)

	end = time.time()
	print('Time Taken ',end-start)
	# print(sorted_zipped_score[:20],'\n')
	# print(all_gt_files[:20])



