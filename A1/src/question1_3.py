import os
import numpy as np
import skimage
from skimage.transform import integral_image
from skimage import feature
import cv2
import glob
from utils import *
import json


def compute_non_max_nbors(x_lmaxs,y_lmaxs,sigma_lmaxs,pt):

	ind_nbors = np.where((x_lmaxs<=pt[0]+5) & (x_lmaxs>=pt[0]-5) & (y_lmaxs>=pt[1]-5) & (y_lmaxs<=pt[1]+5))

	if len(ind_nbors[0])>1:
		# print(ind_nbors[0])
		
		# print(len(ind_nbors[0]))
		local_lvals = [sigma_lmaxs[k] for k in ind_nbors[0]]
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

	to_remove_inds_all = []

	for x in range(1,img.shape[0]-1):
		for y in range(1,img.shape[1]-1):
			to_remove_inds = compute_non_max_nbors(x_lmaxs,y_lmaxs,sigma_lmaxs,(x,y))
			if len(to_remove_inds)>0:
				to_remove_inds_all.extend(to_remove_inds)

	unique_removing_inds = np.unique(to_remove_inds_all)
	new_l_max_pts = []
	for z in range(len(local_max_points)):
		if z not in unique_removing_inds:
			new_l_max_pts.append(local_max_points[z])
	return new_l_max_pts


def get_surf_feat(sigmas,image):
	integral_img = integral_image(image)
	# integral_image = np.float(integral_img)
	surf_keypts = []

	for idx,sigma in enumerate(sigmas):
		det_o_hess = feature.hessian_matrix_det(image, sigma=sigma)        
		lmax_pts = feature.peak_local_max(det_o_hess, num_peaks=50)

		for iidx in range(lmax_pts.shape[0]):
			surf_keypts.append((lmax_pts[iidx,0]  , lmax_pts[iidx,1], sigma))

	return surf_keypts


if __name__=='__main__':

	base_path = './train/'
	base_img_path = './images/'
	img_paths = get_paths_for_queries(base_img_path,base_path)
	print(img_paths)
	diff_sigma = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6]

	surf_keypts_dict = {}
	save_dir = 'surf_res2/'
	os.makedirs(save_dir,exist_ok=True)
	# img_paths = img_paths[args.start_idx,args.end_idx]

	for idx,img_path in enumerate(img_paths):
		print(idx)
		# print(img_path)
		img = cv2.imread(img_path)
		# print(img.shape)
		gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
		gray_img = cv2.resize(img,(int(gray_img.shape[1]/4),int(gray_img.shape[0]/4)))

		surf_keypts = get_surf_feat(diff_sigma,gray_img)
		print('Before Overlap',len(surf_keypts))
		new_surf_keypts = removeOverlap(surf_keypts,gray_img)
		print('After Overlap',len(new_surf_keypts))
		for (y,x,sigma) in new_surf_keypts:
			img = cv2.circle(img,(x,y), int(2*sigma),color=(0,0,255),thickness=1)

		save_surf_keypts = [[int(x[0]),int(x[1]),float(x[2])] for x in surf_keypts]
		surf_keypts_dict[img_path.split('/')[-1]] = save_surf_keypts
		
		save_name = img_path.split('/')[-1].split('.')[0]+'_surf_keypts.jpg'
		cv2.imwrite(save_dir+save_name,img)

	with open('surf_keypts_query.json', 'w', encoding='utf-8') as f:
	    json.dump(surf_keypts_dict, f, ensure_ascii=False, indent=4)




