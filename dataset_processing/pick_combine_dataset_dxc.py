
"""
03/20/2018 XD
Give list1 and list2,  first train list1, then give percentage 1 and percentage2, it will generate [list1*percentage1 + list2*percentage2] together, with the last FC layer replaced.
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import scipy.io as scio


def pick_data(dataset1, label1, percentage1,   dataset2,label2,percentage2): # return a mixed dataset including percentage1% of dataset1 and percentage2 of dataset1, then take random oder
	num_take1 = int(math.floor(percentage1 * dataset1.shape[0]))
	num_take2 = int(math.floor(percentage2 * dataset2.shape[0]))	
	# print(num_take1,num_take2)
	dataset1_picked = np.zeros([num_take1, dataset1.shape[1]])
	dataset2_picked = np.zeros([num_take2, dataset2.shape[1]])
	label1_picked = np.zeros([num_take1, label1.shape[1]])
	label2_picked = np.zeros([num_take2, label2.shape[1]])
	# print(dataset1.shape[0], dataset2.shape[0])
	index1 = random.sample(range(dataset1.shape[0]), num_take1)
	# print(len(index1))
	# index1 = sorted(index1)		
	dataset1_picked = dataset1[index1, :]
	label1_picked = label1[index1, :]

	index2 = random.sample(range(dataset2.shape[0]), num_take2)
	# index2 = sorted(index2)	
	dataset2_picked = dataset2[index2, :]
	label2_picked = label2[index2, :]

	# print(dataset2_picked[33,:] == dataset2[index2[33],:])	
	# print(label2_picked[33, :] == label2[index2[33],:])
	print('----------pick data----------')
	print('data taken from first and second list:', dataset1_picked.shape,dataset2_picked.shape)
	# print('label length:', label1_picked.shape[1])
	print('percentage:', percentage1, percentage2)
	return dataset1_picked,label1_picked,   dataset2_picked,label2_picked		


def combine_data(dataset1_picked,label1_picked,   dataset2_picked,label2_picked):
	dataset_stack = np.vstack((dataset1_picked, dataset2_picked))
	label_stack = np.vstack((label1_picked, label2_picked))

	indices = np.random.permutation(dataset_stack.shape[0]) 
	combined_data = dataset_stack[indices, :]
	combined_label = label_stack[indices, :]
	print('----------combine data----------')
	print('combined dataset and label shape:', combined_data.shape, combined_label.shape)

	return combined_data, combined_label


def data_shuffle(train_data, train_label): 
	# when data is in shape [XX, 28, 28, 1]
	# when label is in shape[XX, 10]
	permutation = np.random.permutation(train_label.shape[0])
	shuffled_dataset = train_data[permutation, :, :, :]
	shuffled_labels = train_label[permutation, :]
	return shuffled_dataset, shuffled_labels
