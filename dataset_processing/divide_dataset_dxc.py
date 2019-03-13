'''
# Xiaocong Du 03/05/2018

# Instructions: Code is used to divide dataset dataset into label_list1 and non-selected label, dataset_image should be the whole dataset image, dataset_label should be the one-hot label. 

# label_list1 is the array that contains the classes you want to train first.

# mode 0 : if label_list1 and label_list2 are in order and covers all 10 digits
#          Eg.: [0 1 2] & [3 4 5 6 7 8 9]
#          Eg.: [0 1 2 3 4] & [ 5 6 7 8 9]
#               In this case, 
# 	          	  short label of 3 will be [0 0 0 1 0 ]
# 	              long label of 3 will be  [0 0 0 1 0. 0 0 0 0 0], same with one-hot label
# 	              short label of 8 will be [0 0 0 1 0]
# 	              long label of 8 will be  [0 0 0 0 0. 0 0 0 1 0], same with one-hot label


# mode 1: if label_list1 and label_list2 are not in order and/or cannot cover 10 digits
#         Eg.: [0 2 3 5 6 8 9] & [1 4 7] are not in order but covers 10 digits
#             In this case,
#             	  short label of 3 will be [0 0 1 0 0 0 0 ]
# 	              long label of 3 will be  [0 0 1 0 0 0 0. 0 0 0], re-assign labels
# 	              short label of 4 will be [0 1 0]
# 	              long label of 4 will be  [0 0 0 0 0 0 0. 0 1 0], re-assign labels

#          Eg.: [0 1 4 7] & [8 3] cannot cover 10 digits
#              In this case,
#             	  short label of 4 will be [0 0 1 0]
# 	              long label of 4 will be  [0 0 1 0. 0 0], re-assign labels
# 	              short label of 3 will be [0 1]
# 	              long label of 3 will be  [0 0 0 0. 0 1], re-assign labels

# Actually, when given list is in order and cover 10 digits, mode 1 is same with mode 0;
# When mode=1, label_list1 = [0, ...,n] label_list2 = [n+1, ...,9], 0<n<9, mode 1 assigns same label with mnist.train.label

'''


# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import scipy.io as scio
import numpy as np

def divide_dataset(dataset_image, dataset_label, label_list1, label_list2, mode):	
	
	print('\nDividing dataset.....')
	m = dataset_label.shape[0]  #55000
	n = dataset_image.shape[1]

	num_selected = len(label_list1)
	num_second_selected = len(label_list2)
	num_total = num_selected + num_second_selected

	#initial
	first_image = np.zeros([m, n]) # [55000, 784]
	first_label_long = np.zeros([m, num_total])  # [55000, 10]
	first_label_short = np.zeros([m, num_selected])	
	first_label_global = np.zeros([m, 1])

	second_image = np.zeros([m, n])
	second_label_long = np.zeros([m, num_total])
	second_label_short = np.zeros([m, num_second_selected])
	second_global_label = np.zeros([m, 1])

	global_label = np.argmax(dataset_label, 1)  # it will return [7, 3, 4....] shape:(55000,)

	
	if mode == 0: 
		j = 0
		jj = 0
		for i in range(m):
			if global_label[i] in label_list1: 
				first_image[j, :] = dataset_image[i, :]
				first_label_long[j, :] = dataset_label[i, :]
				first_label_global[j, :] = global_label[i]
				j += 1
			elif global_label[i] in label_list2:
				second_image[jj, :] = dataset_image[i, :]
				second_label_long[jj, :] = dataset_label[i, :]
				second_global_label[jj, :] = global_label[i]
				jj += 1

		# delete extra-zeros using array slicing
		first_image = first_image[0:j, :]
		first_label_long = first_label_long[0:j, :]	
		first_label_global = first_label_global[0:j, :]
		first_label_short = first_label_long[:, 0:num_selected]

		second_image = second_image[0:jj, :]
		second_label_long = second_label_long[0:jj, :]	
		second_global_label = second_global_label[0:jj, :]
		second_label_short = second_label_long[:, num_selected:]

		# print
		print('Shape of list1 and list2 images: ',first_image.shape, second_image.shape)
		# print(first_label_long.shape, second_label_long.shape)
		print('Label groups are:',np.unique(np.argmax(first_label_long,1)),
			  np.unique(np.argmax(second_label_long,1)))  #return unique elements of global_label
		
		scio.savemat('divide_dataset_mode0.mat', {'first_image':first_image, 'first_label_long':first_label_long, 'first_label_short':first_label_short,
			'second_image':second_image, 'first_label_global':first_label_global, 'second_label_long':second_label_long, 'second_global_label':second_global_label, 'second_label_short':second_label_short})
		print('Dividing dataset finished and saved in divide_dataset_mode0.mat.....')
		print('mode 0\n')

		return first_image, first_label_long, first_label_short,first_label_global,  second_image, second_label_long, second_global_label, second_label_short



	if mode == 1: 	   

        # select image and slicing
		j = 0
		jj = 0
		for i in range(m):
			if global_label[i] in label_list1: 
				first_image[j, :] = dataset_image[i, :]
				first_label_global[j, :] = global_label[i]	
				j += 1
			elif global_label[i] in label_list2:
				second_image[jj, :] = dataset_image[i, :]
				second_global_label[jj, :] = global_label[i]
				jj += 1

		first_image = first_image[0:j, :]		
		second_image = second_image[0:jj, :]
		first_label_global = first_label_global[0:j, :]
		second_global_label = second_global_label[0:jj, :]
		
		# re-assign labels
		first_label_short = np.zeros([j, num_selected])	
		second_label_short = np.zeros([jj, num_second_selected])

		first_short_label_matrix = np.eye(num_selected, dtype=float)  
		second_short_label_matrix = np.eye(num_second_selected, dtype=float)  
		
		for i in range(j):
			for index in range(num_selected): 
				if first_label_global[i,:] == label_list1[index]:
					first_label_short[i, :] = first_short_label_matrix[index,:]
			
		for i in range(jj):
			for index in range(num_second_selected): 
				if second_global_label[i,:] == label_list2[index]:
					second_label_short[i, :] = second_short_label_matrix[index,:]

		first_label_long = np.c_[first_label_short, np.zeros([j, num_second_selected])]
		second_label_long = np.c_[np.zeros([jj, num_selected]), second_label_short]


		print('Shape of list1 and list2 images: ',first_image.shape, second_image.shape)
		# print(first_label_long.shape, second_label_long.shape)
		print('Label groups are:',np.unique(first_label_global),
			  np.unique(second_global_label))  #return unique elements of global_label
		# scio.savemat('divide_dataset_mode1.mat', {'first_image':first_image, 'first_label_long':first_label_long, 'first_label_short':first_label_short,'second_image':second_image, 'first_label_global':first_label_global, 'second_label_long':second_label_long, 'second_global_label':second_global_label, 'second_label_short':second_label_short})
		print('Dividing dataset finished and saved in divide_dataset_mode1.mat')
		print('mode 1\n')

		return first_image, first_label_long, first_label_short,first_label_global,  second_image, second_label_long, second_global_label, second_label_short



# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import scipy.io as scio


# '''mode 0 test'''

# # download dataset
# mnist = input_data.read_data_sets("dataset_data/", one_hot=True)
# # Divide dataset 
# label_list1 = [0,1,2,3,4,5,6]
# label_list2 = [7,8,9]

# first_image, first_label_long, first_label_short,first_label_global, second_image, second_label_long, second_global_label, second_label_short = divide_dataset(mnist.train.images, mnist.train.labels, label_list1, label_list2, 0)



'''mode 1 test'''
# ## download dataset
# mnist = input_data.read_data_sets("dataset_data/", one_hot=True)
# # Divide dataset 
# label_list1 = [0,1,4,7]
# label_list2 = [3,8]
# first_image, first_label_long, first_label_short,first_label_global, second_image, second_label_long, second_global_label, second_label_short = divide_dataset(mnist.train.images, mnist.train.labels, label_list1, label_list2, 1)



# label_list1 = [0,1,2,3,4,5,6,7,8,9]
# label_list2 = []
# print(np.r_[first_label_long, second_label_long] == mnist.train.labels)