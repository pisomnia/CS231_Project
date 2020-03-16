# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 22:18:16 2018

@author: shadi
"""

import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *
import matplotlib.pyplot as plt
from pylab import*
import gc

gc.collect()
flag_subset = False
boosting_type = 'Ada' #'Real' or 'Ada'
training_epochs = 100 if not flag_subset else 20
act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

# data configurations
pos_data_dir = 'newface16'
neg_data_dir = 'nonface16'
image_w = 16
image_h = 16
data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
data = integrate_images(normalize(data))

# number of bins for boosting
num_bins = 25

# number of cpus for parallel computing
#num_cores = 8 if not flag_subset else 1 #always use 1 when debugging
num_cores = 4
# create Haar filters
filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

# create visualizer to draw histograms, roc curves and best weak classifier accuracies
drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])

# create boost classifier with a pool of weak classifier
boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

#########   
start = time.clock()
boost.calculate_training_activations(act_cache_dir, act_cache_dir)
end = time.clock()
print('%f seconds for activation calculation' % (end - start))
#########

training_error=boost.train_Ada(chosen_wc_cache_dir)
boost.visualize()

'''boost.train_Real(chosen_wc_cache_dir)
boost.visualize()'''

#Plot the top 20 filters
'''plt.figure(figsize=(20,20))
for i in range(20):
	subplot(4,5,i+1)
	alpha, wc = boost.chosen_wcs[i]
	filter_plus=wc.plus_rects
	filter_minus=wc.minus_rects
	if wc.polarity==1:
		img=np.zeros((16,16,3))+0.5
		for j in range(len(filter_plus)):
			x1,y1,x2,y2=filter_plus[j]
			x1=int(x1)
			y1=int(y1)
			x2=int(x2)
			y2=int(y2)
			img[x1:x2+1,y1:y2+1]=1.0
		for k in range(len(filter_minus)):
			x1,y1,x2,y2=filter_minus[k]
			x1=int(x1)
			y1=int(y1)
			x2=int(x2)
			y2=int(y2)
			img[x1:x2+1,y1:y2+1]=0.0
	if wc.polarity==-1:
		img=np.zeros((16,16,3))+0.5
		for j in range(len(filter_plus)):
			x1,y1,x2,y2=filter_plus[j]
			x1=int(x1)
			y1=int(y1)
			x2=int(x2)
			y2=int(y2)
			img[x1:x2+1,y1:y2+1]=0.0
		for k in range(len(filter_minus)):
			x1,y1,x2,y2=filter_minus[k]
			x1=int(x1)
			y1=int(y1)
			x2=int(x2)
			y2=int(y2)
			img[x1:x2+1,y1:y2+1]=1.0
			
	plt.imshow(img)
plt.show()
plt.savefig('Top20_filters.png')'''



'''plt.figure()
step=np.linspace(1,100,100)
plt.plot(step, training_error)
plt.title('Training Error over steps')
plt.ylabel('Training Error of the Strong Classifier')
plt.xlabel('Over the number of steps')
plt.savefig('Training_Error')'''

'''original_img1 = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
result_img1 = boost.face_detection(original_img1)
cv2.imwrite('Result_img1_%s.png' % boosting_type, result_img1)'''

#get_hard_negative_patches
non_face1=cv2.imread('./Testing_Images/Non_Face_1.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch1 = boost.get_hard_negative_patches(non_face1)
for i in range(wrong_patch1.shape[0]):
     cv2.imwrite('./nonface16/nonface16_new1_%d.bmp' % (i+1), wrong_patch1[i])   
    
    
non_face2=cv2.imread('./Testing_Images/Non_Face_2.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch2=boost.get_hard_negative_patches(non_face2)
for i in range(wrong_patch2.shape[0]):
     cv2.imwrite('./nonface16/nonface16_new2_%d.bmp' % (i+1), wrong_patch2[i]) 

non_face3=cv2.imread('./Testing_Images/Non_Face_3.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch3=boost.get_hard_negative_patches(non_face3)
for i in range(wrong_patch3.shape[0]):
     cv2.imwrite('./nonface16/nonface16_new3_%d.bmp' % (i+1), wrong_patch3[i]) 
     
'''original_img1 = cv2.imread('./Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
result_img1 = boost.face_detection(original_img1)
cv2.imwrite('Result_img1_%s.png' % boosting_type, result_img1)

original_img2 = cv2.imread('./Testing_Images/Face_2.jpg', cv2.IMREAD_GRAYSCALE)
result_img2 = boost.face_detection(original_img2)
cv2.imwrite('Result_img2_%s.png' % boosting_type, result_img2)

original_img3 = cv2.imread('./Testing_Images/Face_3.jpg', cv2.IMREAD_GRAYSCALE)
result_img3 = boost.face_detection(original_img3)
cv2.imwrite('Result_img3_%s.png' % boosting_type, result_img3)'''