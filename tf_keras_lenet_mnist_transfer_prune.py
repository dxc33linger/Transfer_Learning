from __future__ import print_function

"""03/14/2018"""
"""xiaocong du"""
"""lenet + mnist + keras, function model"""

""" Give list1 and list2,  first train list1, then give percentage 1 and percentage2,
It will train [list1*percentage1 + list2*percentage2] together, with the last FC layer replaced."""
"""e.g.: list1 = [0,1,2,3,4], percentage1 = 0.1,
		 list2 = [5,6,7,8,9], percentage2 = 0.9,

		 first training: train ~30K 0 to 4 digits, FC5
		 then train ~3k 0 to 4 digits, ~27k 5 to 9 digits, with FC10, but bottom layers frozen
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
from keras.layers import Lambda




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
	#   layer.trainable = False
	# for layer in model.layers[layer_name:]:
	#   layer.trainable = True  
	# model.compile(loss='categorical_crossentropy',
	#             optimizer=adam,
	#             metrics=['accuracy'])


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
	# Another way to define your optimizer
	
	# We add metrics to get more results you want to see
	# model.compile(optimizer=adam,
	#               loss='categorical_crossentropy',
	#               metrics=['accuracy'])
	print('training data shape:', X_train.shape)
	print('training label shape:', y_train.shape)
	X_train = X_train.reshape(-1, 28, 28, 1)
	X_test = X_test.reshape(-1, 28, 28, 1)
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


img = tf.placeholder(tf.float32, shape = [None, 784])
label_ = tf.placeholder(tf.float32, shape = [None, None])
keep_prob = tf.placeholder(tf.float32)



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)/255.
X_test = X_test.reshape(-1, 784)/255.
# print(np.unique(y_train))
# turn label into one-hot label
y_train = np_utils.to_categorical(y_train, num_classes=10) # (60000, 10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
# print(np.unique(np.argmax(y_train, 1)))

# divide dataset [0,1,2,3,4,5,6,7,8,9]
label_list1 = [0,1]

label_list2 = [5,6]

percentage1 = 0.1  # pick XX% from list1 data, XX% from list2 data

percentage2 = 0.1  #percentage 2 should not be set to 0


print('\n---------------------------prepare training data--------------------------------')
# divide dataset according to 2 lists
first_train_image, first_train_label_long, first_train_label_short,first_train_label_global, second_train_image, second_train_label_long, second_global_label, second_train_label_short = divide_dataset(X_train, y_train, label_list1, label_list2, 1)

if percentage2 == 0.: #
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_train_image, first_train_label_short, percentage1, second_train_image, second_train_label_short, percentage2)    
else:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_train_image, first_train_label_long, percentage1, second_train_image, second_train_label_long, percentage2)  

# combine XX% list1 and XX% list2 data
combined_train_data, combined_train_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)


percentage1 = 0.

# percentage2 = 0.1

print('\n------------------------prepare testing data------------------------------------')
first_test_image, first_test_label_long, first_test_label_short,first_test_label_global, second_test_image, second_test_label_long, second_global_label, second_test_label_short = divide_dataset(X_test, y_test, label_list1, label_list2, 1)

if percentage2 == 0.:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_test_image, first_test_label_short, percentage1, second_test_image, second_test_label_short, percentage2)    
else:
	dataset1_picked,label1_picked, dataset2_picked,label2_picked = pick_data(first_test_image, first_test_label_long, percentage1, second_test_image, second_test_label_long, percentage2)

combined_test_data, combined_test_label = combine_data(dataset1_picked,label1_picked, dataset2_picked,label2_picked)


keep_prob_val = 1.0

print('\n-----------------------------Base model---------------------------------------')
# Define feature extraction model=conv+pool+conv+pool+fc+fc
digit_input = Input(shape = (None, 784))
x = tf.reshape(digit_input, [-1,28,28,1])
# x = digit_input.reshape(-1, 28, 28, 1)

x = Convolution2D(batch_input_shape=(None, 28, 28, 1), filters=32, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_last', kernel_initializer='random_normal')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)
x = Convolution2D(batch_input_shape=(None, 28, 28, 1), filters=64, kernel_size=5, strides=1, activation='relu', padding='same', data_format='channels_last', kernel_initializer='random_normal')(x)
x = MaxPooling2D(pool_size=2, strides=2, padding='same', data_format='channels_last')(x)
x = Flatten(name = 'flatten')(x)
feature = Dense(1024, activation='relu', name ='FC1', kernel_initializer='random_normal')(x)
# feature = Dropout(keep_prob_val)(x)


base_model_FE = Model(digit_input, feature)

model1 = add_new_last_layer(base_model_FE, len(label_list1)) # list1 classes
model2 = add_new_last_layer(base_model_FE, combined_test_label.shape[1]) #10 classes


print('\n---------------------------first training--------------------------------------')
epoch = 15
batch_size = 64
lr1 = 5e-4
lr2 = 5e-4


def tf_compile(train_model, lr, train_img, train_label):
	train_img = Input(shape=(None,784))
	model_output = train_model(train_img)
	cross_entropy = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=model_output))
	train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(train_label,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return cross_entropy, train_step, accuracy
	

def tf_train_sess(epoch, batch_size, train_img, train_label, test_img, test_label, keep_prob_val):
	iteration = int(100*((epoch * 60000 / batch_size)//100+1))
	print('iteration:', iteration)
	for i in range(iteration):
		# batch_img = train_img.next_batch(batch_size)
		# batch_label = train_label.next_batch(batch_size)

		if i%1000 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				img: train_img[i*batch_size:(i+1)*batch_size, :], 
				label_: train_label[i*batch_size:(i+1)*batch_size, :], 
				keep_prob: keep_prob_val})

			print("iteration %d, training accuracy %g"%(i,train_accuracy)) 

		train_step.run(feed_dict={
			img:test_img, 
			label_:test_label, 
			keep_prob:keep_prob_val})

	print("test accuracy %g"%accuracy.eval(feed_dict={
		img: test_img, label_: test_label, keep_prob:keep_prob_val}))



sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

cross_entropy, train_step, accuracy = tf_compile(model1, lr1, first_train_image, first_train_label_short)
tf_train_sess(epoch, batch_size, first_train_image, first_train_label_short, first_test_image, first_test_label_short, keep_prob_val)

