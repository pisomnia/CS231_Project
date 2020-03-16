import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle

import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		#self.chosen_wcs = None
		self.chosen_wcs = []
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	# select weak classifiers to form a strong classifier
	# after training, by calling self.sc_function(), a prediction can be made
	# self.chosen_wcs should be assigned a value after self.train() finishes
	# call Weak_Classifier.calc_error() in this function
	# cache training results to self.visualizer for visualization
	#
	#
	# detailed implementation is up to you
	# consider caching partial results and using parallel computing
	def train_Ada(self, save_dir = None):
		######################
		weights=1/(self.data.shape[0])*np.ones(self.data.shape[0])
		num_iter=100
		error=np.zeros(len(self.weak_classifiers))
		F=np.zeros((num_iter,self.data.shape[0]))
		f=np.zeros(self.data.shape[0])
		H=np.zeros((num_iter,self.data.shape[0]))
		training_error=np.zeros(num_iter)
		#dict=[]
		self.visualizer.labels=self.labels
		if save_dir is not None and os.path.exists(save_dir):
			self.load_trained_wcs(save_dir)
		for t in range(num_iter):
			if save_dir is not None and os.path.exists(save_dir):
				_,weak_clf=self.chosen_wcs[t]
				
			else:
				for i, wc in enumerate(self.weak_classifiers):
					error[i],h= wc.calc_error( weights, self.labels)
					print('Number of the training weak classifier:%d----%d/10032' %(t,i))
				index=np.argmin(error)
				weak_clf=self.weak_classifiers[index]
			Error,ht=weak_clf.calc_error( weights, self.labels)
			alpha =1/2*np.log((1-Error)/Error)
			Z=(1-Error)*np.exp(-1*alpha)+Error*np.exp(alpha)
			#for j in range (self.data.shape[0]):
				#weights[j]=weights[j]*(1/Z)*np.exp(-1*alpha*ht[j]*self.labels[j])
			weights=weights*(1/Z)*np.exp(-1*alpha*ht*self.labels)
			self.chosen_wcs.append([alpha,weak_clf])
			f+=alpha*ht
			F[t]=f
			H[t]=np.sign(F[t])
			training_error[t]=1/(self.data.shape[0])*np.sum(H[t]!=self.labels)
			#print(self.visualizer.histogram_intervals)
			if (t+1) in self.visualizer.histogram_intervals:
				self.visualizer.strong_classifier_scores = {**self.visualizer.strong_classifier_scores,**{(t+1):F[t]}}
				print(self.visualizer.strong_classifier_scores)
			if (t+1) in self.visualizer.top_wc_intervals:
					Er=sorted(error)[:1000]
					self.visualizer.weak_classifier_accuracies ={**self.visualizer.weak_classifier_accuracies,**{(t+1):Er}}
		######################
		if save_dir is not None:
			if os.path.exists(save_dir):
				print("Model has been trained")
			else:
				pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))
		return training_error
		
	def train_Real(self, save_dir = None):
		######################
		weights=1/(self.data.shape[0])*np.ones(self.data.shape[0])
		f=np.zeros(self.data.shape[0])
		F=np.zeros((100,self.data.shape[0]))
		h=np.zeros(self.data.shape[0])
		self.visualizer.weak_classifier_accuracies = {}
		self.visualizer.strong_classifier_scores = {}
		if save_dir is not None and os.path.exists(save_dir):
			self.load_trained_wcs(save_dir)
		for t in range(100):
			_,weak_clf=self.chosen_wcs[t]
			thresholds=np.linspace(min(weak_clf.activations),max(weak_clf.activations),self.num_bins+1)
			thresholds=thresholds[1:self.num_bins]
			#bin_pqs=np.zeros((2,self.num_bins))
			p=np.zeros(self.num_bins)+0.0000001
			q=np.zeros(self.num_bins)+0.0000001
			print('Number of the real boosting training :%d' %t)
			for i in range(self.data.shape[0]):
				bin_idx = np.sum(thresholds < weak_clf.activations[i])
				if self.labels[i]==1:
					p[bin_idx]=p[bin_idx]+weights[i]
				else:
					q[bin_idx]=q[bin_idx]+weights[i]
			htb=0.5 * np.log(p / q)
			Z=2*np.sum(np.sqrt(p*q))
			for j in range (self.data.shape[0]):
				bin_idx = np.sum(thresholds < weak_clf.activations[j])
				h[j]=htb[bin_idx]
				weights[j]=weights[j]*(1/Z)*np.exp(-1*h[j]*self.labels[j])
			f+=h
			F[t]=f
			if (t+1) in self.visualizer.histogram_intervals:
				self.visualizer.strong_classifier_scores = {**self.visualizer.strong_classifier_scores,**{(t+1):F[t]}}

	def sc_function(self, image):
		return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])			

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, scale_step = 10):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0.0), np.sum(np.array(predicts) > 0.0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0.0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms = nms(pos_predicts_xyxy, 0.01)
		#xyxy_after_nms=pos_predicts_xyxy
		print('after nms:', xyxy_after_nms.shape[0])
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3

		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		predicts=np.array(predicts)
		wrong_patches = patches[np.where(predicts > 0), ...]
		return wrong_patches

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()
