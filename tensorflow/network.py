from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import cifar10
import prettytensor as pt

cifar10.data_path = "data/CIFAR-10/"
cifar10.maybe_download_and_extract() # pre fetch data
class_names = cifar10.load_class_names()
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()
# print(cifar10.load_training_data()[0])
# print(cls_train)

IMG_SIZE_CROPPED = 24 #24*24 pixel
SAVE_DIR = 'save/'

print(class_names)

x = tf.placeholder(tf.float32, shape=[None, cifar10.img_size, cifar10.img_size, cifar10.num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, cifar10.num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[IMG_SIZE_CROPPED, IMG_SIZE_CROPPED, cifar10.num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=IMG_SIZE_CROPPED,
                                                       target_width=IMG_SIZE_CROPPED)

    return image

def pre_process(images, training):
	# Use TensorFlow to loop over all the input images and call
	# the function above which takes a single image as input.
	images = tf.map_fn(lambda image: pre_process_image(image, training), images)

	return images

def main_network(images, training):
    # Wrap the input images as a Pretty Tensor object.
    x_pretty = pt.wrap(images)

    # Pretty Tensor uses special numbers to distinguish between
    # the training and testing phases.
    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    # Create the convolutional neural network using Pretty Tensor.
    # It is very similar to the previous tutorials, except
    # the use of so-called batch-normalization in the first layer.
    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=cifar10.num_classes, labels=y_true)

    return y_pred, loss


def create_network(training):
    # Wrap the neural network in the scope named 'network'.
    # Create new variables during training, and re-use during testing.
    with tf.variable_scope('network', reuse=not training):
        # Just rename the input placeholder variable for convenience.
        images = x

        # Create TensorFlow graph for pre-processing.
        images = pre_process(images=images, training=training)

        # Create TensorFlow graph for the main processing.
        y_pred, loss = main_network(images=images, training=training)

    return y_pred, loss


if __name__ == '__main__':
	if not os.path.exists(SAVE_DIR):
		os.makedirs(SAVE_DIR)

	global_step = tf.Variable(initial_value=0,
							name='global_step', trainable=False)
	if sys.argv[1] == 'train':
		_, loss = create_network(training=True)
		optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
		y_pred, _ = create_network(training=False)
		y_pred_cls = tf.argmax(y_pred, dimension=1)
		correct_prediction = tf.equal(y_pred_cls, y_true_cls)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		saver = tf.train.Saver()
	# tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)