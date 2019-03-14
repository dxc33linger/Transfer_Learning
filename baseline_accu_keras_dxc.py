from __future__ import print_function

"""03/22/2018"""
"""XD"""
"""lenet + mnist + keras, function model"""

""" one list training to get baseline accuracy
"""

import keras
import tensorflow as tf
import datetime
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
import numpy as np
import math
from keras import initializers
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt
from divide_dataset_dxc import divide_dataset
import random
import scipy.io as scio
from keras.models import load_model


# from keras.models import Sequential
now = datetime.datetime.now  # timing counting

def add_new_last_layer(base_model, nb_classes):
	x = base_model.output
	predictions = Dense(nb_classes, activation='softmax',kernel_initializer='random_normal')(x) #new softmax layer
	model = Model(inputs=base_model.input, outputs=predictions)
	return model


def setup_to_transfer_learn(model, base_model):
	"""Freeze all layers and compile the model"""
	for layer in base_model.layers:
		layer.trainable = False
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])


def setup_to_finetune(model, layer_name):    
	for layer in model.layers[:layer_name]:
		layer.trainable = False
	for layer in model.layers[layer_name:]:
		layer.trainable = True	
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])


def plot_training1(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))

	scio.savemat('history_this_baseline.mat', {'acc':acc, 'epochs': epochs, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss})
	print('acc saved in history_this_baseline.mat')

	plt.plot(epochs, acc, 'b-')
	plt.plot(epochs, val_acc, 'b-*')
	# plt.legend(['first train_acc', 'first test_acc'], loc='best')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	# plt.xlim(0, epochs)
	plt.title('Training and testing accuracy')

	# plt.figure()
	# plt.plot(epochs, loss, 'r-*')
	# plt.plot(epochs, val_loss, 'r*')
	# plt.legend(['train_loss', 'val_loss'], loc='best')
	# plt.xlabel('epoch')
	# plt.title('Training and testing loss')
	# plt.show()



def train_model(model, X_train, y_train, X_test, y_test, epoch, batch_size):
	# Another way to define your optimizer
	
	# We add metrics to get more results you want to see
	# model.compile(optimizer=adam,
	#               loss='categorical_crossentropy',
	#               metrics=['accuracy'])
	print('training data shape:', X_train.shape)
	print('training label shape:', y_train.shape)
	X_train = X_train.reshape(-1, 1, 28, 28)
	X_test = X_test.reshape(-1, 1, 28, 28)
	t = now()
	# Another way to train the model
	history = model.fit(X_train, y_train, 
		epochs= epoch, 
		batch_size=batch_size, 
		validation_data=(X_test, y_test),
		)
	print('\nTraining time: %s' % (now() - t))

	# Evaluate the model with the metrics we defined earlier
	print('-------------------------testing----------------------------')
	loss, accuracy = model.evaluate(X_test, y_test)
	print('test loss:', loss)
	print('test accuracy:', accuracy)
	return history


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
print('\n-------------------------define function finished------------------------------')



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)/255.
X_test = X_test.reshape(-1, 784)/255.
# print(np.unique(y_train))
# turn label into one-hot label
y_train = np_utils.to_categorical(y_train, num_classes=10) # (60000, 10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(np.unique(np.argmax(y_train, 1)))

# divide dataset [0,1,2,3,4,5,6,7,8,9]
label_list1 = [5,6,7,8,9]
label_list2 = label_list1

percentage1 = 1.
percentage2 = 0.

print('\n---------------------------prepare training data--------------------------------')
# divide dataset according to 2 lists
first_train_image, first_train_label_long, first_train_label_short,first_train_label_global, second_train_image, second_train_label_long, second_global_label, second_train_label_short = divide_dataset(X_train, y_train, label_list1, label_list2, 1)

# replace "second_train_label_long" into 'second_train_label_short'
dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_train_image, first_train_label_short, percentage1, second_train_image, second_train_label_short, percentage2)	
# combine XX% list1 and XX% list2 data
combined_train_data, combined_train_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)


print('\n------------------------prepare testing data------------------------------------')
first_test_image, first_test_label_long, first_test_label_short,first_test_label_global, second_test_image, second_test_label_long, second_global_label, second_test_label_short = divide_dataset(X_test, y_test, label_list1, label_list2, 1)


dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_test_image, first_test_label_short, percentage1, second_test_image, second_test_label_short, percentage2)	

combined_test_data, combined_test_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)






print('\n---------------------------baseline training--------------------------------------')
# Define feature extraction model=conv+pool+conv+pool+fc+fc
digit_input = Input(shape = (1, 28, 28))
x = Convolution2D(batch_input_shape=(None, 1, 28, 28), filters=32, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_first')(digit_input)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first',)(x)
x = Convolution2D(batch_input_shape=(None, 1, 28, 28), filters=64, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_first')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first',)(x)
x = Flatten(name = 'flatten')(x)
feature = Dense(1024, activation='relu', name ='FC1')(x)

base_model_FE = Model(digit_input, feature)

model1 = add_new_last_layer(base_model_FE, len(combined_test_label[1])) # list1 classes


epoch = 15
batch_size = 64
lr1 = 5e-4
# lr2 = 5e-4

adam = Adam(lr=lr1)
model1.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])

history1 = train_model(model1, combined_train_data, combined_train_label, combined_test_data, combined_test_label, epoch, batch_size)
# plot_model(model1, to_file='baseline_keras_function.png')



# print('\n---------------------------transfer training------------------------------------')
# model2 = add_new_last_layer(base_model_FE, combined_test_label.shape[1]) #10 classes
# adam = Adam(lr=lr2)
# setup_to_transfer_learn(model2, base_model_FE)	 #include compile

# history2 = train_model(model2, combined_train_data, combined_train_label, combined_test_data, combined_test_label, epoch, batch_size)
# plot_model(model2, to_file='model2_keras_function.png')



# print('\n--------------------------------plotting----------------------------------------\n')
# plot_training1(history1)
# plt.legend(['baseline train acc', 'baseline test acc'], loc='best')
# locs, labels = plt.xticks()          # Get locations and labels
# plt.xticks(np.arange(1, epoch+1, step=1)) # Set locations and labels
 
# dataFile = './history_this_baseline.mat'  
# data = scio.loadmat(dataFile) 

# print('\nacc=', data['acc'])
# print('val_acc=',data['val_acc'])
# # model2.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'
# # del model2  # deletes the existing model
# # print('my_model2.h5 saved')
# # returns a compiled model
# # # identical to the previous one
# # model = load_model('my_model.h5')

# plt.savefig('./train_curve_baseline_accu_keras_dxc.png')
# plt.show()
