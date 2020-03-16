# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 07:33:35 2018

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
import sys

non_face1=cv2.imread('./Testing_Images/Non_Face_1.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch1 = boost.get_hard_negative_patches(non_face1)
for i in range(wrong_patch1.shape[1]):
     cv2.imwrite('./nonface16/nonface16_new1_%d.bmp' % (i+1), wrong_patch1[0,i,:,:])   
    
    
non_face2=cv2.imread('./Testing_Images/Non_Face_2.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch2=boost.get_hard_negative_patches(non_face2)
for i in range(wrong_patch2.shape[1]):
     cv2.imwrite('./nonface16/nonface16_new2_%d.bmp' % (i+1), wrong_patch2[0,i,:,:]) 

non_face3=cv2.imread('./Testing_Images/Non_Face_3.jpg', cv2.IMREAD_GRAYSCALE)
wrong_patch3=boost.get_hard_negative_patches(non_face3)
for i in range(wrong_patch3.shape[1]):
     cv2.imwrite('./nonface16/nonface16_new3_%d.bmp' % (i+1), wrong_patch3[0,i,:,:]) 
     


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
