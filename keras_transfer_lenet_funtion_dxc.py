from __future__ import print_function

"""
Date: 03/14/2018
Author: XD
LeNet + MNIST + keras, function model

 Give list1 and list2,  first train list1, then give percentage 1 and percentage2,
It will train [list1*percentage1 + list2*percentage2] together, with the last FC layer replaced.
e.g.: list1 = [0,1,2,3,4], percentage1 = 0.1,
		 list2 = [5,6,7,8,9], percentage2 = 0.9,

		 first training: train ~30K 0 to 4 digits, FC5
		 then train ~3k 0 to 4 digits, ~27k 5 to 9 digits, with FC10, but bottom layers frozen

		 modify the second percentage1 to change testing distribution form list1.
"""

import keras
import tensorflow as tf
import datetime
from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model, np_utils
from keras.optimizers import Adam, SGD
import numpy as np
import math
from keras import initializers
from keras.models import Model, load_model
import matplotlib.pyplot as plt
from divide_dataset_dxc import divide_dataset
import random
import scipy.io as scio



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


def setup_to_finetune(model, base_model):  
	for layer in base_model.layers:
			layer.trainable = True
	model.compile(loss='categorical_crossentropy',
					  optimizer=adam,
					  metrics=['accuracy'])  
	# for layer in model.layers[:layer_name]:
	# 	layer.trainable = False
	# for layer in model.layers[layer_name:]:
	# 	layer.trainable = True	
	# model.compile(loss='categorical_crossentropy',
	# 			  optimizer=adam,
	# 			  metrics=['accuracy'])


def plot_training1(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))

	scio.savemat('history1.mat', {'acc':acc, 'epochs': epochs, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss})
	print('acc saved in history1.mat')

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


def plot_training2(history):
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(acc))

	scio.savemat('history2.mat', {'acc':acc, 'epochs': epochs, 'val_acc':val_acc, 'loss':loss, 'val_loss':val_loss})
	print('acc saved in history2.mat')

	plt.plot(epochs, acc, 'r-')
	plt.plot(epochs, val_acc, 'r-*')
	# plt.legend(['second train_acc', 'second test_acc'], loc='best')
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	# plt.xlim(0, epochs)

	# plt.title('Training and testing accuracy')

	# plt.figure()
	# plt.plot(epochs, loss, 'r-*')
	# plt.plot(epochs, val_loss, 'r*')
	# plt.legend(['train_loss', 'val_loss'], loc='best')
	# plt.xlabel('epoch')
	# plt.title('Training and testing loss')
	# plt.show()


def train_model(model, X_train, y_train, X_test, y_test, epoch, batch_size):

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
label_list1 = [0,1,2,3,4]

label_list2 = [5,6,7,8,9]


print('\n---------------------------prepare training data--------------------------------')

percentage1 = 0.1  # pick XX% from list1 data, XX% from list2 data

percentage2 = 0.1  #percentage 2 should not be set to 0

# divide dataset according to 2 lists
first_train_image, first_train_label_long, first_train_label_short,first_train_label_global, second_train_image, second_train_label_long, second_global_label, second_train_label_short = divide_dataset(X_train, y_train, label_list1, label_list2, 1)

if percentage2 == 0.: #
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_train_image, first_train_label_short, percentage1, second_train_image, second_train_label_short, percentage2)	
else:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_train_image, first_train_label_long, percentage1, second_train_image, second_train_label_long, percentage2)	

# combine XX% list1 and XX% list2 data
combined_train_data, combined_train_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)


print('\n------------------------prepare testing data------------------------------------')

percentage1 = 0.

percentage2 = 0.1

first_test_image, first_test_label_long, first_test_label_short,first_test_label_global, second_test_image, second_test_label_long, second_global_label, second_test_label_short = divide_dataset(X_test, y_test, label_list1, label_list2, 1)

if percentage2 == 0.:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_test_image, first_test_label_short, percentage1, second_test_image, second_test_label_short, percentage2)	
else:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_test_image, first_test_label_long, percentage1, second_test_image, second_test_label_long, percentage2)

combined_test_data, combined_test_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)



print('\n-----------------------------Base model---------------------------------------')
# Define feature extraction model=conv+pool+conv+pool+fc+fc
digit_input = Input(shape = (1, 28, 28))
x = Convolution2D(batch_input_shape=(None, 1, 28, 28), filters=32, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_first', kernel_initializer='random_normal')(digit_input)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)
x = Convolution2D(batch_input_shape=(None, 1, 28, 28), filters=64, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_first', kernel_initializer='random_normal')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_first')(x)
x = Flatten(name = 'flatten')(x)
feature = Dense(1024, activation='relu', name ='FC1', kernel_initializer='random_normal')(x)

base_model_FE = Model(digit_input, feature)



print('\n---------------------------first training--------------------------------------')
epoch = 15
batch_size = 64
lr1 = 5e-4
lr2 = 5e-4


model1 = add_new_last_layer(base_model_FE, len(label_list1)) # list1 classes
adam = Adam(lr=lr1)
model1.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])

history1 = train_model(model1, first_train_image, first_train_label_short, first_test_image, first_test_label_short, epoch, batch_size)
plot_model(model1, to_file='model1_keras_function.png')



print('\n---------------------------transfer training------------------------------------')
model2 = add_new_last_layer(base_model_FE, combined_test_label.shape[1]) #10 classes
adam = Adam(lr=lr2)

setup_to_transfer_learn(model2, base_model_FE)	 #include compile

# setup_to_finetune(model2, base_model_FE)

history2 = train_model(model2, combined_train_data, combined_train_label, combined_test_data, combined_test_label, epoch, batch_size)
plot_model(model2, to_file='model2_keras_function.png')


print('\n--------------------------------plotting----------------------------------------\n')
plot_training1(history1)
plot_training2(history2)
#baseline plot
# base_train_acc = np.array([0.948083333396912,	0.984983333396912,	0.989950000000000,	0.992216666793823,	0.994716666730245,	0.995933333333333,	0.99670,	0.997050000000000,	0.99770,	0.998016666730245,	0.998733333333333,	0.998150000063578,	0.99850,	0.998283333333333,	0.998966666666667,	0.998650000000000,	0.999350000000000,	0.998783333333333,	0.999833333333333,	0.99950]) 
# base_test_acc = np.array([0.98230,	0.98820,	0.98990,	0.99060,	0.98980,	0.99140,	0.99180,	0.99150,	0.99200,	0.99160,	0.99310,	0.98960,	0.99200,	0.99350,	0.99220,	0.99210,	0.99090,	0.99130,	0.99340,	0.99310])
# dataFile = './history_0to9_baseline.mat'  
# data = scio.loadmat(dataFile)
# base_09train_acc = data['acc'][0]
# base_09test_acc = data['val_acc'][0]
# plt.plot(np.linspace(1,epoch, epoch), base_09train_acc[:epoch], 'g-')
# plt.plot(np.linspace(1,epoch, epoch), base_09test_acc[:epoch], 'g-*')


# baseline_5to9_train_acc = np.array([0.952217385397851,	0.988300911440620,	0.992109917026201,	0.994728608352605,	0.996531084206231,	0.996973200925044,	0.998333560059856,	0.997959461297783,	0.998367569038226,	0.999285811454224,	0.999387838389335,	0.998537613930078,	0.999115766562372,	0.998401578016596,	0.999455856346075])

# baseline_5to9_test_acc = np.array([0.989714051305794,	0.990536929206872,	0.991565522196557,	0.994239869038771,	0.994445589559359,	0.993622714024810,	0.990331210635911,	0.994857026002359,	0.994857028368887,	0.994445588026633,	0.988274020757285,	0.993828432595772,	0.991976959473359,	0.993622712075183,	0.994034152417437])
# plt.plot(np.linspace(1,epoch, epoch), baseline_5to9_train_acc[:epoch], 'k-')
# plt.plot(np.linspace(1,epoch, epoch), baseline_5to9_test_acc[:epoch], 'k-*')


# plt.legend(['source data train acc', 'source data test acc','target data train acc', 'target data test acc', '[0-9] baseline train acc', '[0-9] baseline test acc', '[5-9] baseline train acc', '[5-9] baseline test acc'], loc='best')
# locs, labels = plt.xticks()          # Get locations and labels
# plt.xticks(np.arange(1, epoch+1, step=1)) # Set locations and labels


model2.save('my_model2.h5')  # creates a HDF5 file 'my_model.h5'
del model2  # deletes the existing model
print('my_model2.h5 saved')

print(' epoch, batch, lr1, lr2: ', epoch, batch_size, lr1, lr2, '           ')
print('---------------------------------------------------------------------------------')

# returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')

# plt.savefig('./train_curve_keras_function_lenet_mnist.png')
# plt.show()
