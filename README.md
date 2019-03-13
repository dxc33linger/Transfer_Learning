# Implement Transfer learning with Keras on MNIST.



## Required environment

1. Python 3.6/2.7
2. TensorFlow
3. Keras
4. NumPy, SciPy, Matplotlib, etc.


##  Code
[Baseline training](https://github.com/dxc33linger/transfer_learning/blob/master/baseline_accu_keras_dxc.py): baseline_accu_keras_dxc.py

[With Sequential method](https://github.com/dxc33linger/transfer_learning/blob/master/keras_transfer_lenet_sequential_dxc.py): keras_transfer_lenet_sequential_dxc.py

[With Function method](https://github.com/dxc33linger/transfer_learning/blob/master/keras_transfer_lenet_funtion_dxc.py): keras_transfer_lenet_funtion_dxc.py

<img src="https://github.com/dxc33linger/transfer_learning/blob/master/dataset_processing/train_curve_keras_function_lenet_mnist.png" width="500" alt="learning curve"/>

## Dataset_processing folder

Includes files/code/output related to dataset pre-prosessing.

1. [divide_dataset_dxc.py](https://github.com/dxc33linger/transfer_learning/blob/master/dataset_processing/divide_dataset_dxc.py): Divide datasets into groups
2. [pick_combine_dataset_dxc.py](https://github.com/dxc33linger/transfer_learning/blob/master/dataset_processing/pick_combine_dataset_dxc.py): Pick weighted combination of data and combine into new dataset.



Reference:
1. LeCun, Yann. "LeNet-5, convolutional neural networks." URL: http://yann.lecun.com/exdb/lenet20(2015).
2. https://keras.io/


