#%% Direct data import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import keras.backend as k
import os
import tqdm


#####################################################################################%%


# Main Network

class ConvolutionalNeuralNetwork:
    def __init__(self, fc_input, fc_hidden, fc_output, filter_1, filter_2): # batchsize, learningrate

        self.filter1_w = filter_1[1]
        self.filter1_h = filter_1[0]

        self.filter2_w = filter_2[1]
        self.filter2_h = filter_2[0]

        self.fc_inputnodes = fc_input
        self.fc_hiddennodes = fc_hidden
        self.fc_outputnodes = fc_output

        self.flatten_nodes = 147

        # Initializing weights
        # conv_1 # filter size = (3, 5, 5)
        self.conv1_filter1 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))
        self.conv1_filter2 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))
        self.conv1_filter3 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))
        self.conv1_filter4 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))
        self.conv1_filter5 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))
        self.conv1_filter6 = np.random.normal(0.0, pow(self.filter1_w, -0.5), (3, self.filter1_h, self.filter1_w))

        # conv_2 # filter size = (3, 5, 5)
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))
        self.conv2_filter1 = np.random.normal(0.0, pow(self.filter2_w, -0.5), (3, self.filter2_h, self.filter2_w))

        # FC-Layer weights Initialization
        self.wih = np.random.normal(0.0, pow(self.flatten_nodes, -0.5), (self.fc_hiddennodes, self.flatten_nodes)) # ih = input to hidden weights
        self.who = np.random.normal(0.0, pow(self.fc_hiddennodes, -0.5), (self.fc_outputnodes, self.fc_hiddennodes)) # ho = hidden to output weights # (10, 200)

        self.bias_hidden1 = np.random.normal(0.0, pow(self.flatten_nodes, -0.5), (self.fc_hiddennodes, 1))
        self.bias_hidden2 = np.random.normal(0.0, pow(self.fc_hiddennodes, -0.5), (self.fc_outputnodes, 1))
        # Learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def Conv2D_forward(self, image, flter):

        image_c, image_h, image_w = image.shape # channels, rows (height), columns (width)
        filter_c, filter_h, filter_w = flter.shape

        # final image size = (e.g.) (input height - kernel height)/stride + 1

        final_featuremap = np.zeros((image_h - filter_h + 1, image_w - filter_w +1))

        for row in range(image_h - filter_h + 1):
            for column in range(image_w - filter_w + 1):
                final_featuremap[row, column] = np.sum(np.multiply(flter[:, :, :], image[:, row:row+filter_w, column:column+filter_h])) # row:row+filter_width, row]) # image[all channels: selected height: selected width]

        return final_featuremap

    def Max_Pooling2D(self, input_image, filter_size, stride=2):
        image_c, image_h, image_w = input_image.shape
        filter_height, filter_width = filter_size  # height = rows, width = columns
        maxpool_c, maxpool_h, maxpool_w = image_c, int((image_h - filter_height) / 2 + 1), int(
            (image_w - filter_width) / 2 + 1)

        final_pooled_feature = np.zeros((maxpool_c, maxpool_h, maxpool_w))
        locations = np.zeros((maxpool_c, maxpool_h, maxpool_w))

        for row in range(maxpool_h):
            for column in range(maxpool_w):
                h_start = row * stride
                h_end = h_start + filter_height
                w_start = column * stride
                w_end = w_start + filter_width

                slice_to_max_extract = input_image[:, h_start:h_end, w_start:w_end]  # Shape is (3, 2, 2)
                final_pooled_feature[:, row, column] = np.amax(slice_to_max_extract, axis=(
                1, 2))  # axis = (1, 2) means, to find the max along a 2D feature map

                for channel in range(locations.shape[0]):
                    locations[channel, row, column] = np.argmax(slice_to_max_extract[channel, :, :].flatten())

        return final_pooled_feature, locations


    def RELU_Activation(self, x): # RELU
        activated_x = np.maximum(0, x)
        return activated_x

    def deri_activation(self, x):
        x = Activation(x)(1 - Activation(x))
        return x


    def train(self, inputs_list, targets_list):

        inputs = inputs_list
        targets = targets_list

        conv1op_1, conv1op_2, conv1op_3, conv1op_4, conv1op_5, conv1op_6 = Conv2D_forward(self.input_image, self.conv1_filter1), Conv2D_forward(self.input_image, self.conv1_filter2), Conv2D_forward(self.input_image, self.conv1_filter3), Conv2D_forward(self.input_image, self.conv1_filter4), Conv2D_forward(self.input_image, self.conv1_filter5), Conv2D_forward(self.input_image, self.conv1_filter6)
        conv1_final_feature_maps = np.stack((conv1op_1, conv1op_2, conv1op_3, conv1op_4, conv1op_5, conv1op_6), axis = 0)
        # ReLU
        conv1_output = Activation(conv1_final_feature_maps) # size = (3, 28, 28)

        conv2op_1, conv2op_2, conv2op_3, conv2op_4, conv2op_5, conv2op_6 = Conv2D_forward(conv1_final_feature_maps, self.conv2_filter1), Conv2D_forward(conv1_final_feature_maps, self.conv2_filter2), Conv2D_forward(conv1_final_feature_maps, self.conv2_filter3), Conv2D_forward(conv1_final_feature_maps, self.conv2_filter4), Conv2D_forward(conv1_final_feature_maps, self.conv2_filter5), Conv2D_forward(conv1_final_feature_maps, self.conv2_filter6)
        conv2_final_feature_maps = np.stack((conv2op_1, conv2op_2, conv2op_3, conv2op_4, conv2op_5, conv2op_6), axis = 0)
        # ReLU
        conv2_output = Activation(conv2_final_feature_maps) # size = (3, 24, 24)

        # Max Pooling
        maxpool_output = Max_Pooling2D(conv2_output, (2, 2)) # size = (3, 7, 7)

        # Flattened max-pool output
        flattened_array = maxpool_output.flatten() # shape = (147,)
        input_to_fc = np.reshape(flattened_array, (147, 1)) # This is final size we get after (3, 32, 32) input

        # FC Layers with biases
        hidden_inputs = np.add(np.dot(self.wih, input_to_fc), self.bias_hidden1) # (200, 147) x (147, 1)
        hidden_outputs = Activation(hidden_inputs) # (200, 1)

        final_inputs = np.add(np.dot(self.who, hidden_outputs), self.bias_hidden2) # (10, 200) x (200, 1)
        final_outputs = Activation(final_inputs) # Final Logits # (10, 1)

        # Output layer
        error_outputs = targets - final_outputs (10, 1)

        errors_hidden = np.dot(self.who, error_outputs) # (200, 10) x (10, 1) [this 10-vector is the error vector] = (200, 1)
.

        errors_fc_input = np.dot(self.wih, errors_hidden)


    # # train the neural network
    #
    # def train(self, inputs_list, targets_list): # TODO
    #     # convert inputs list to 2d array
    #     inputs = numpy.array(inputs_list, ndmin=2).T
    #     targets = numpy.array(targets_list, ndmin=2).T
    #
    # # Defining relevant functions




#####################################################################################%%
# Training the Network

# Hyperparameters

# input_data = # something filters/images
fc_hidden = 200
fc_output = 10
filter_1 = (5, 5)
filter_2 = (5, 5)

learning_rate = 0.1
BatchSize = 16

# create instance of the CNN
network = convolutionalNN(input_data, fc_hidden, fc_output, filter_1, filter_2)

# Loading Training Data

from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.transpose(0, 3, 1, 2) # (50000, 3, 32, 32)
x_test = x_test.transpose(0, 3, 1, 2) # (10000, 3, 32, 32)
# y_train = (50000, 1)
# y_test = (10000, 1)
y_train =  np.reshape(y_train,(50000)) # (50000,)
targets_train = np.zeros([10,50000]) + 0.01 # (10, 50000)

for column in range(y_train):
    class_index = y_train[i]
    targets_train[class_index][i] = 0.99


for e in range(epochs):
    # go through all records in the training data set
    for input in x_train:
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        #targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        #targets[int(all_values[0])] = 0.99
        #n.train(inputs, targets)
        network = ConvolutionalNeuralNetwork(input_data = inputs, fc_hidden = 200, fc_output = 10, filter_1 = (5, 5), filter_2= (5, 5))



