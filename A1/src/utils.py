import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
import glob
import os
import pickle
from PIL import ImageFilter
from sklearn.cluster import KMeans
from itertools import product


def visualize(img_np):
	img = Image.fromarray(np.uint8(img_np)).convert('RGB')
	plt.figure()
	plt.imshow(img)
	plt.show()


def save_dset(base_imgs_path,img_size=(100,100)):
	all_imgs = glob.glob(base_imgs_path+'*.jpg')
	num_imgs = len(all_imgs)
	print('Num Imgs',num_imgs)
	X = np.zeros((num_imgs,img_size[0],img_size[1],3))
	for idx,x in enumerate(all_imgs):
		print('Saving Dataset...  {}/{}'.format(idx+1,num_imgs))
		img = Image.open(x).convert("RGB")
		img = img.resize(img_size)
		X[idx,:,:] = np.array(img)

	with open('img_np_dset.pkl','wb') as f:
		pickle.dump(X,f)

def load_dset(base_imgs_path,img_size=(100,100)):
	if not os.path.exists('./img_np_dset.pkl'):

		save_dset(base_imgs_path,img_size)

	with open('img_np_dset.pkl','rb') as f:
		X = pickle.load(f)

	return X


def precision_recall(gt_dict,scores,k=50,recall=False):
	good_imgs = gt_dict['good']
	ok_imgs = gt_dict['ok']
	junk_imgs = gt_dict['junk']

	all_gt_imgs = np.union1d(good_imgs,ok_imgs)
	all_gt_imgs = np.union1d(all_gt_imgs,junk_imgs)
	print('Len all gt imgs',len(all_gt_imgs))

	all_gt_imgs = [x for x in all_gt_imgs]
	sorted_pred_names = [score[0] for score in scores]
	sorted_pred_scores = [score[1] for score in scores]
	print(len(sorted_pred_names))
	sorted_pred_names_k = sorted_pred_names[:k]
	# print('sorted_pred_names_k',sorted_pred_names_k)
	# print('all_gt_imgs',all_gt_imgs)
	if recall:
		score = len(np.intersect1d(sorted_pred_names_k,all_gt_imgs))/len(all_gt_imgs)
	else:
		score = len(np.intersect1d(sorted_pred_names_k,all_gt_imgs))/k

	if len(np.intersect1d(sorted_pred_names_k,all_gt_imgs)) == 0:
		good_percentage = 0 
		ok_percentage = 0
		junk_percentage = 0
	else:
		good_percentage = 100*(len(np.intersect1d(sorted_pred_names_k,good_imgs))/(len(np.intersect1d(sorted_pred_names_k,all_gt_imgs))))
		ok_percentage = 100*(len(np.intersect1d(sorted_pred_names_k,ok_imgs))/(len(np.intersect1d(sorted_pred_names_k,all_gt_imgs))))
		junk_percentage = 100*(len(np.intersect1d(sorted_pred_names_k,junk_imgs))/(len(np.intersect1d(sorted_pred_names_k,all_gt_imgs))))

	return score,good_percentage,ok_percentage,junk_percentage

def get_paths_for_queries(base_img_path,base_path):
	base_query_path = base_path+'query/'
	all_query_files = glob.glob(base_query_path+'*.txt')
	img_paths = []
	for query_file in all_query_files:
		with open(query_file,'r') as f:
			line = f.readlines()[0].strip()
			tokens = line.split(' ')

			img_query_name = tokens[0][tokens[0].find('_')+1:]+'.jpg'
			img_paths.append(base_img_path+img_query_name)

	return img_paths

	# base_path 


def quantize_colors_bin(dset_np):
	# if not os.path.exists('kmeans_quant_obj_binned_four_64.pkl'):
		
	all_img_vals = np.arange(256)
	binned_img_vals = [x for x in all_img_vals if x%4==2]
	possible_tuples = product(binned_img_vals,repeat=3)               # 3 times as rgb tuple

	kmeans = KMeans(n_clusters=64, random_state=0,n_jobs=-1).fit(possible_tuples)

	kmeans_cls_centres = kmeans.cluster_centers_

	with open('dset_color_cluster_centres_32.pkl','wb') as f:
		pickle.dump(kmeans_cls_centres,f)