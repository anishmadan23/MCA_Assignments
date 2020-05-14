import numpy as np
import os
import cv2
from itertools import product 
import glob 
import json 
from utils import *

k = 1.5
initial_sigma = 1

def log_filter(sigma):
	filter_size = 2*np.ceil(3*sigma)+1
	x = np.arange((-filter_size+1)/2,(filter_size+1)/2)
	y = np.arange((-filter_size+1)/2,(filter_size+1)/2)

	hg = np.exp(-(x*x)/(2*(sigma**2)))
	hg_out = np.outer(hg,hg.T)
	hg_out_sum = np.sum(hg_out)

	xy_tuples = np.array(list(product(x,repeat=2)))
	x_sq_y_sq = [x[0]**2+x[1]**2 - 2*(sigma**2) for x in xy_tuples]
	xy_resh = np.reshape(x_sq_y_sq,hg_out.shape)
	# print(xy_resh.shape)
	# print(hg_out.shape)

	scale_normalised_filter = (xy_resh*hg_out)/hg_out_sum

	return (sigma**2)*scale_normalised_filter

def sq_convolve(img,my_filter):
	dst = cv2.filter2D(img,-1,my_filter)
	dst = dst**2
	return dst

def get_response_scale_space(img,k=1.5,initial_sigma=1,num_scales=8):
	scale_space_responses = np.zeros((img.shape[0],img.shape[1],num_scales))
	sigma_scales = []
	for i in range(num_scales):
		sigma_scale = initial_sigma*pow(k,i)
		sigma_scales.append(sigma_scale)
		my_filter = log_filter(sigma_scale)
		# print(my_filter)
		scale_space_responses[:,:,i] = sq_convolve(img,my_filter)

	return scale_space_responses, sigma_scales

def findLocalMax(scale_space_responses,sigma_scales,thresh=215):
	local_max_points = []
	# local_max_values = []
	for idx_scale in range(1,len(sigma_scales)-1):
		cur_sc_resp = scale_space_responses[:,:,idx_scale]

		for x in range(1,cur_sc_resp.shape[0]-1):
			for y in range(1,cur_sc_resp.shape[1]-1):
				comp_cells = scale_space_responses[x-1:x+2,y-1:y+2,idx_scale-1:idx_scale+2].copy()

				max_cell_comp = np.zeros(comp_cells.shape)
				max_cell_comp[1,1,1] = cur_sc_resp[x,y]

				comp_cells_nbors_max = np.max((comp_cells-max_cell_comp))
				if cur_sc_resp[x,y]>comp_cells_nbors_max and cur_sc_resp[x,y]>thresh:          #get rid of low values of keypoints
					local_max_points.append((x,y,sigma_scales[idx_scale],idx_scale,scale_space_responses[x,y,idx_scale]))
					# local_max_values.append(cur_sc_resp[x,y])

	return local_max_points#,local_max_values

def compute_non_max_nbors(x_lmaxs,y_lmaxs,sigma_lmaxs,lval_lmaxs,pt):

	ind_nbors = np.where((x_lmaxs<=pt[0]+5) & (x_lmaxs>=pt[0]-5) & (y_lmaxs>=pt[1]-5) & (y_lmaxs<=pt[1]+5))

	if len(ind_nbors[0])>1:
		# print(ind_nbors[0])
		
		# print(len(ind_nbors[0]))
		local_lvals = [lval_lmaxs[k] for k in ind_nbors[0]]
		# print('local_vals',local_lvals)
		max_lval_idx = np.argmax(local_lvals)
		# print('max_lval_idx',max_lval_idx)
		max_ind_nbor = ind_nbors[0][max_lval_idx]
		# print('max_ind_nbor',np.array([max_ind_nbor,max_ind_nbor]),np.array(ind_nbors[0]))
		max_idx = int(np.where(ind_nbors[0]==max_ind_nbor)[0])
		# print('max idx',max_idx)
		to_remove_inds = []
		for i in range(len(ind_nbors[0])):
			if i!=max_idx:
				to_remove_inds.append(ind_nbors[0][i])
		# to_remove_inds = np.setdiff1d(np.array([max_ind_nbor,max_ind_nbor]),np.array(ind_nbors[0]))
		# print('to remove',to_remove_inds)
		return to_remove_inds
	return []

def removeOverlap(local_max_points,img):
	x_lmaxs = np.array([k[0] for k in local_max_points])
	y_lmaxs = np.array([k[1] for k in local_max_points])
	sigma_lmaxs = np.array([k[2] for k in local_max_points])
	lval_lmaxs = np.array([k[4] for k in local_max_points])

	to_remove_inds_all = []

	for x in range(1,img.shape[0]-1):
		for y in range(1,img.shape[1]-1):
			to_remove_inds = compute_non_max_nbors(x_lmaxs,y_lmaxs,sigma_lmaxs,lval_lmaxs,(x,y))
			if len(to_remove_inds)>0:
				to_remove_inds_all.extend(to_remove_inds)

	unique_removing_inds = np.unique(to_remove_inds_all)
	new_l_max_pts = []
	for z in range(len(local_max_points)):
		if z not in unique_removing_inds:
			new_l_max_pts.append(local_max_points[z])
	return new_l_max_pts

if __name__=='__main__':

	save_dir = 'blobs_query_res/'
	os.makedirs(save_dir,exist_ok=True)

	blob_detection_keypts_dict = {}

	base_path = './train/'
	base_img_path = './images/'
	img_paths = get_paths_for_queries(base_img_path,base_path)

	for idx,img_path in enumerate(img_paths):
		print('Running ',idx)
		# img_path = './images/all_souls_000026.jpg'
		img = cv2.imread(img_path)
		gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
		gray_img = cv2.resize(gray_img,(int(gray_img.shape[1]/4),int(gray_img.shape[0]/4)))

		scale_space_responses,sigma_scales = get_response_scale_space(gray_img)

		local_max_pts = findLocalMax(scale_space_responses,sigma_scales)
		print('Before removing overlap pts = ',len(local_max_pts))

		new_l_max_pts = removeOverlap(local_max_pts,gray_img)
		print('After removing overlap pts = ',len(new_l_max_pts))
		for (y,x,sigma,idx_sigma,lval) in new_l_max_pts:
			img = cv2.circle(img,(x,y), 4*idx_sigma,color=(0,0,255),thickness=1)

		save_blob_keypts = [[int(p[0]),int(p[1]),float(p[2])] for p in new_l_max_pts]

		blob_detection_keypts_dict[img_path.split('/')[-1]] = save_blob_keypts
		save_name = img_path.split('/')[-1].split('.')[0]+'_blobs.jpg'
		cv2.imwrite(save_dir+save_name,img)

		# print(np.min(local_max_values),np.max(local_max_values),np.mean(local_max_values))
	with open('blob_detection_keypts_query.json', 'w', encoding='utf-8') as f:
	    json.dump(blob_detection_keypts_dict, f, ensure_ascii=False, indent=4)