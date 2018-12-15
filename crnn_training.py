import numpy 
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import aifc
import os
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import image
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, normalize
from sklearn.metrics.pairwise import pairwise_distances
from matplotlib import mlab
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import layers
from keras.layers import Input, Embedding, Lambda, LSTM, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, Conv3D, MaxPooling2D, Concatenate, MaxPooling3D
from keras.models import load_model, Sequential, Model
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
import tensorflow as tf
from matplotlib.pyplot import imshow
import keras
import keras.backend as K
K.set_image_data_format('channels_first')

def data():
    ''' data Method
            Load in pre-defined datasets (training, validation, test. etc.)
            
            Returns: 
                X_train_n: 4-D matrix of the (normalized) training set
                X_val_n: 4-D matrix of the (normalized) validation set
                X_testV_n: 4-D matrix of the (normalized) vertical-contrast-enhanced test set
                X_testH_n: 4-D matrix of the (normalized) horizontal-contrast-enhanced test set
                Y_train: 2-D matrix of the labels for the training set
                Y_val: 2-D matrix of the labels for the validation set
                Y_test: 2-D matrix of the labels for the test set
                autoencoder_train_n: 4-D matrix of the samples used to train the autoencoder
                X_cv_train_n: 4-D matrix of the dataset used for K-Fold Cross Validation
                Y_cv_train: 2-D matrix of the labels for the K-Fold Cross Validation set
                
    '''
    # All pre-defined datasets have already been normalized between 0 and 1
    # Since numpy.loadtxt() loads in txt files as 2D matrices, they must be reshaped into their original shape
    # Training set
    X_train_n = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/X_train_n.txt',delimiter=',')
    X_train_n = numpy.reshape(X_train_n,(76546,1,50,59))
    # Training set labels
    Y_train = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/Y_train.txt',delimiter=',')
    # Validation set
    X_val_n = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/X_val_n.txt',delimiter=',')
    X_val_n = numpy.reshape(X_val_n,(9568,1,50,59))
    # Validation set labels
    Y_val = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/Y_val.txt',delimiter=',')
    # Vertical-contrast-enhanced test set
    X_testV_n = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/X_testV_n.txt',delimiter=',')
    X_testV_n = numpy.reshape(X_testV_n,(4784,1,50,59))
    # Horizontal-contrast-enhanced test set
    X_testH_n = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/X_testH_n.txt',delimiter=',')
    X_testH_n = numpy.reshape(X_testH_n,(4784,1,50,59))
    # Test set labels
    Y_test = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/Y_test.txt',delimiter=',')
    # Samples from the original dataset used to train the autoencoder
    autoencoder_train_n = numpy.loadtxt('Documents/Bioacoustics_MachineLearning/train1_final/autoencoder_train_10000.txt',delimiter=',')
    autoencoder_train_n = numpy.reshape(autoencoder_train_n,(10000,1,50,59))
    # K-Fold Cross Validation will be used to obtain performance statistics for the model. The pre-defined training 
    # and validation sets will be combined into one larger dataset used in the cross validation procedure, since
    # the partitioning process creates a "training set" (all folds used to train the model for the particular
    # iteration) and a "validation set" (the one fold not used for training). 
    X_cv_train_n = numpy.concatenate((X_train_n,X_val_n),axis=0)
    # Cross Validation dataset labels
    Y_cv_train = numpy.append(Y_train,Y_val)
    return X_train_n,X_val_n,X_testV_n,X_testH_n,Y_train,Y_val,Y_test,autoencoder_train_n,X_cv_train_n,Y_cv_train

def ExtractPatches_KMeans(X, patch_size, max_patches, n_clusters):
    ''' MiniBatchKMeansAutoConv Method 
            Extract patches from the input X and pass the complete set of patches to MiniBatchKMeans to 
            learn a dictionary of filters 
            
            Args:
                X: (Number of samples, Height, Width, Number of Filters)
                patch_size: int size of patches to extract from X 
                max_patches: float decimal percentage of maximum number of patches to 
                             extract from X, else 'None' to indicate no maximum
                n_clusters: int number of centroids (filters) to learn via MiniBatchKMeans
            Returns: 
                learned centroids of shape: (Number of samples, Number of filters, Height, Width)
                
    '''
    # Batch size for MiniBatchKMeans
    batch_size=50
    # Input Shape: (Number of samples, Number of Filters, Height, Width)
    # Reshape into: (Number of samples, Height, Width, Number of Filters)
    X = np.transpose(X,(0,2,3,1))
    # Dimensions of X
    sz = X.shape
    # Extract patches from each sample up to the maximum number of patches using sklearn's
    # PatchExtractor
    X = image.PatchExtractor(patch_size=patch_size,max_patches=max_patches).transform(X)
    # For later processing, ensure that X has 4 dimensions (add an additional last axis of
    # size 1 if there are fewer dimensions)
    if(len(X.shape)<=3):
        X = X[...,numpy.newaxis]
    # Local centering by subtracting the mean
    X = X-numpy.reshape(numpy.mean(X, axis=(1,2)),(-1,1,1,X.shape[-1])) 
    # Local scaling by dividing by the standard deviation 
    X = X/(numpy.reshape(numpy.std(X, axis=(1,2)),(-1,1,1,X.shape[-1])) + 1e-10) 
    X = X.transpose((0,3,1,2)) 
    # Reshape X into a 2-D array for input into MiniBatchKMeans
    X = numpy.asarray(X.reshape(X.shape[0],-1),dtype=numpy.float32)
    # Convert X into an intensity matrix with values ranging from 0 to 1
    X = mat2gray(X)
    # Perform PCA whitening
    pca = PCA(whiten=True)
    X = pca.fit_transform(X)
    # Scale input samples individually to unit norm (using sklearn's "normalize")
    X = normalize(X)
    # Use "MiniBatchKMeans" on the extracted patches to find a dictionary of n_clusters 
    # filters (centroids)
    km = MiniBatchKMeans(n_clusters = n_clusters,batch_size=batch_size,init_size=3*n_clusters).fit(X).cluster_centers_
    # Reshape centroids into shape: (Number of samples, Number of filters, Height, Width)
    return km.reshape(-1,sz[3],patch_size[0],patch_size[1])

def mat2gray(X):
    ''' mat2gray Method 
            Convert the input X into an intensity image with values ranging from 0 to 1
            
            Args:
                X: (Number of samples, Number of filters, Height, Width)         
            Returns: 
                X: 2-D intensity image matrix
                
    '''
    # Input Shape: (Number of samples, Number of features)
    # Find the minima of X along axis=1, i.e. the minimum value feature for each sample
    m = numpy.min(X,axis=1,keepdims=True)
    # Find the maxima of X along axis=1 and subtract the minima of X from it to obtain the
    # range of values of X for each sample
    X_range = numpy.max(X,axis=1,keepdims=True)-m
    # Set the values of X so that the maximum corresponds to 1, the minimum corresponds to 0,
    # and values larger than the maximum and smaller than the minimum are set to 1 and 0,
    # respectively
    # For samples where the maximum of the features is equal to the minimum of the features
    # set the feature values to 0
    idx = numpy.squeeze(X_range==0)
    X[idx,:] = 0
    # For samples other than those indexed by idx, subtract the minima of the feature values 
    # from the feature values and divide by X_range
    X[numpy.logical_not(idx),:] = (X[numpy.logical_not(idx),:]-m[numpy.logical_not(idx)])/X_range[numpy.logical_not(idx)]
    return X

def model_fp(model,model_train,layer_name):
    ''' model_fp Method 
            Create an intermediate version of the input Keras model, where the output is 
            derived from an earlier layer specified by "layer_name"
            Args:
                model: Compiled Keras model 
                model_train: 4-D array of inputs to pass through the new intermediate model 
                             and generate the desired output from an earlier layer.
                layer_name: string name of the layer to produce the ouput from in the Keras
                            model
            Returns: 
                output: 4-D array of outputs from the layer specified by "layer_name"
                
    '''
    # Create an intermediate version of the input model, where the intput is the same, but 
    # the output is derived from a layer preceding the input model's output. ("layer_name"
    # specifies which layer should be the intermediate model's output)
    intermediate_layer_model = Model(inputs=model.input,outputs=model.get_layer(layer_name).output)  
    # Determine the output of the intermediate model when given an input of model_train
    output = intermediate_layer_model.predict(model_train)
    return output 

def separate_trainlabels(Y_train):
    ''' separate_trainlabels Method 
            Randomly shuffle the indices of the training set and extract arrays "Y_pos"
            and "Y_neg" holding indices of solely positive and solely negative labels, 
            respectively 
            Args:
                Y_train: Labels for the training set X_train of shape: (Number of samples),
                         with labels being either 0 or 1
            Returns: 
                Y_pos: 1-D numpy array of indices to positive labels 
                Y_neg: 1-D numpy array of indices to negative labels
                indices: 1-D array of shuffled indices into the original array Y_train
                
    '''
    # Randomly shuffle the indices of Y_train
    indices = numpy.arange(len(Y_train))
    numpy.random.shuffle(indices)
    # Reorder Y_train according to the shuffled indices 
    labels = Y_train[indices]
    # Initialize arrays "Y_pos" and "Y_neg" to hold the indices of the positive examples 
    # and negative examples, respectively. 
    Y_pos = []
    Y_neg = []
    for ii in range(len(labels)):
        # Add the index of a positive label to Y_pos if the length of Y_pos is less than 2500.
        # This ensures Y_pos eventually has a total of 2500 positive labels. 
        if labels[ii] == 1:
            if len(Y_pos) < 2500:
                Y_pos.append(ii)
        # Add the index of a negative label to Y_neg if the length of Y_neg is less than 2500.
        # This ensures Y_neg eventually has a total of 2500 negative labels.
        else:
            if len(Y_neg) < 2500:
                Y_neg.append(ii)
        # Break the for loop as soon as Y_pos and Y_neg both have 2500 elements 
        if len(Y_pos) == 2500 and len(Y_neg) == 2500:
            break
    return Y_pos,Y_neg,indices

# Load in pre-defined datasets
X_train_n,X_val_n,X_testV_n,X_testH_n,Y_train,Y_val,Y_test,autoencoder_train_n,X_cv_train_n,Y_cv_train = data()

# Input shape of samples from the training set (excluding batch axis)
input_shape0 = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
# Input layer
X_input0 = Input(shape=input_shape0,name='input0')
# Dropout on the visible layer (1 in 10 probability of dropout) 
X_drop0 = Dropout(0.1,name='drop0')(X_input0)
# BatchNorm on axis=1 (on the axis corresponding to the number of filters)
X_bn0 = BatchNormalization(axis=1,name='bn0')(X_drop0)
# First convolutional layer with filters of size 7x7 and stride of 2 (to achieve downsampling without max pooling)
# Since the filters are determined through unsupervised learning, they are not updated through backpropagation 
# (i.e. trainable=False)
X_conv0_2d = Conv2D(filters=256,strides=2,kernel_size=(7,7),padding='valid',data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv0_2d')(X_bn0)
# Batch norm
X_bn1 = BatchNormalization(axis=1,name='bn1')(X_conv0_2d)
# Convolutional Autoencoder: Instead of directly passing the output of the first convolutional layer to the second
# convolutional layer, the convolutional autoencoder performs dimensionality reduction to collapse the feature map
# into one "summary" channel, thereby improving the performance of K-Means Clustering (less chance of empty clusters
# and poor performance with lower dimensionality inputs)
# The convolutional autoencder consists of six 1x1 convolutional layers which each halve the number of channels. As
# opposed to conventional max pooling, the 1x1 convolutional channel pooling also allows for the learning of useful
# nonlinear relationships, since a 1x1 convolution can be viewed as a multilayer perceptron slid across the image
X_conv0_1x1 = Conv2D(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0')(X_bn1)
X_conv1_1x1 = Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1')(X_conv0_1x1)
X_conv2_1x1 = Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2')(X_conv1_1x1)
X_conv3_1x1 = Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3')(X_conv2_1x1)
X_conv4_1x1 = Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce4')(X_conv3_1x1)
# Encoding learned by the convolutional autoencoder is a feature map of the same height and width dimensions but
# only one "summary" channel
encoded1 = Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1')(X_conv4_1x1)
# Recall that the autoencoder learns a mapping from the input space, to a compressed representation of the input space,
# back to the input space. The first half of the architecture from the input space to the compressed representation
# is called the "encoder," while the second half of the architecture from the compressed representation back to the
# input space is called the "decoder." The weights of the autoencoder are learned unsupervised (since the input and 
# output are identical). Once the weights are learned, the decoder can also be discarded, since only the encoder
# is needed in the final model to perform dimensionality reduction. (The decoder is only needed during training to
# learn the desired mapping).
# Decoder uses transposed convolutional layers to upsample the "summary" feature map back to the original 
# dimensionality of the input space. The order of the layers is the exact reverse of that in the encoder, doubling
# the number of channels with each layer (instead of halving the number of channels with each layer in the encoder).
X_deconv0 = Conv2DTranspose(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='deconv0')(encoded1)
X_deconv1 = Conv2DTranspose(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='deconv1')(X_deconv0)
X_deconv2 = Conv2DTranspose(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='deconv2')(X_deconv1)
X_deconv3 = Conv2DTranspose(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='deconv3')(X_deconv2)
X_deconv4 = Conv2DTranspose(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='deconv4')(X_deconv3)
decoded1 = Conv2DTranspose(filters=256,kernel_size=1,activation='relu',data_format='channels_first',name='decoded1')(X_deconv4)
# For the autoencoder, in order to learn the desired mapping to the compressed representation that preserves enough
# information to reconstruct the original inputs, a mean squared error loss function is used to compare the
# reconstructed representation with the original input. This process is facilitated by flattening the reconstructed
# representation and comparing it elementwise with a flattened version of the original input.
X_flatten_autoencoder1 = Flatten(name='flatten_autodencoder1')(decoded1)
X_output_autoencoder1 = Dense(1,activation='sigmoid',name='output_autoencoder1')(X_flatten_autoencoder1)
# It is important to note in the design of this architecture that the autoencoder contains all layers from the input
# layer through the decoder. (The autoencoder is not connected to the rest of the final model). The following set of 
# layers, however, which are identical to the layers comprising the encoder, are connected to the rest of the model.
# Therefore, during training of the final model, the learned weights of the encoder are loaded into these layers.
# (This, in effect, retains the learned encoder and discards the unneeded decoder). 
X_conv0_1x1t = Conv2D(filters=128,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='reduce0')(X_bn1)
X_conv1_1x1t = Conv2D(filters=64,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='reduce1')(X_conv0_1x1t)
X_conv2_1x1t = Conv2D(filters=32,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='reduce2')(X_conv1_1x1t)
X_conv3_1x1t = Conv2D(filters=16,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='reduce3')(X_conv2_1x1t)
X_conv4_1x1t = Conv2D(filters=8,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='reduce4')(X_conv3_1x1t)
encoded1t = Conv2D(filters=1,kernel_size=1,trainable=False,activation='relu',data_format='channels_first',name='encoded1')(X_conv4_1x1t)
# Pass the compresed representation of the output of the first convolutional layer to the second convolutional layer
X_conv1_2d = Conv2D(filters=256,kernel_size=(7,7),padding='valid',dilation_rate=2,data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv1_2d')(encoded1t)
# Perform max pooling for translation invariance
X_pool1_2d = MaxPooling2D((2,2),data_format='channels_first',name='pool1_2d')(X_conv1_2d)
# Batch Norm
X_bn2 = BatchNormalization(axis=1,name='bn2')(X_pool1_2d)
# Reshape the output to the shape expected by an LSTM, i.e. (batch_size, time_steps, num_features)
# In this case, the "time_steps" are interpreted as the columns of the images (corresponding to the time axis of the
# original spectrograms) and the features are interpreted as the rows of the images for each channel (corresponding
# to the frequency axis of the original spectrograms)
X_reshape_freq = Reshape((K.int_shape(X_bn2)[2],K.int_shape(X_bn2)[3]*K.int_shape(X_bn2)[1]),name='reshape_freq')(X_bn2)
# Bidirectional LSTM
a_freq = Bidirectional(LSTM(45,return_sequences=False,name='rnn_freq'))(X_reshape_freq)
# Fully-connected layer outputs class prediction using sigmoid activation function
X_output_lstm = Dense(1,activation='sigmoid',name='output')(a_freq)
# Adam optimizer
opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.01)
# Create model
model_lstm = Model(X_input0,X_output_lstm)
# Compile model
model_lstm.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])

# Input shape of samples from the training set (excluding batch axis)
input_shape = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
# The variable k_train0_t will store the samples from the original dataset used to train the autoencoder
k_train0_t = autoencoder_train_n
# Use ExtractPatches_KMeans() to calculate centroids for patches extracted from the spectrogram images of the
# training set
centroids0_2d = ExtractPatches_KMeans(k_train0_t,(7,7),None,256)
# The calculated centroids will be set as the filters of the first convolutional layer
layer_toset = model_lstm.get_layer('conv0_2d')
# Reshape the learned filters so that they are the same shape as expected by the Keras layer
filters = centroids0_2d.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset.set_weights(filters)
# Adam Optimizer for the pre_autoencoder and autoencoder model
opt1 = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01)
# The "pre_autoencoder" will be used to create the training set for the autoencoder. Recall that k_train0_t
# contains the samples from the original dataset used to train the autoencoder. However, these samples are
# not yet in the form required by the autoencoder model. (The autoencoder expects input with the same 
# dimensions as the output of the first convolutional layer. In addition, since the input is equal to the 
# output for an autoencoder, the output must have the same dimensions as the output of the first convolutional
# layer). Therefore, the pre_autoencoder is necessary to pass the samples from the original dataset (specified
# by k_train0_t) through the model from the input layer to the first convolutional layer. The samples from 
# the original dataset "transformed" by this procedure will be of the appropriate dimensions for training the
# autoencoder.
# Create the pre_autoencoder, comprising layers from the input layer through the batch normalization layer 
# following the first convolutional layer
pre_autoencoder1 = Model(inputs=X_input0,outputs=X_bn1)
# Compile the model
pre_autoencoder1.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
# Pass the samples from the original dataset specified by k_train_0t through the pre_autoencoder to obtain
# the "transformed" samples of the appropriate dimensions. 
pre_autoencoder_train = pre_autoencoder1.predict(k_train0_t)
# The autoencoder will now take the "transformed" samples (the output of the first convolutional layer) and
# find an optimal compressed representation.
# Create the autoencoder, comprising layers from the input layer through the decoder
autoencoder1 = Model(inputs=X_input0,outputs=decoded1)
# Compile the model
autoencoder1.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
# Note that the input and output to the autoencoder do not appear to be identical. This is due to "autoencoder1"
# comprising layers in the final model from the input layer to the decoder (not just from the encoder to the
# decoder). Therefore, while the input to the model is k_train0_t, by passing k_train0_t through the model 
# up to the encoder, the input given to the autoencoder itself will be the "transformed" samples as desired 
# (corresponding to the output "pre_autoencoder_train").
autoencoder1.fit(k_train0_t,pre_autoencoder_train,epochs=5,batch_size=100)
# Save the weights of the autoencoder
autoencoder1.save('autoencoder1.hdf5')
# Recall that the input to the second convolutional layer is the output of the first convolutional layer (after
# being compresed by the encoder to achieve dimensionality reduction). Therefore, in order to learn filters for the 
# second convolutional layer, patches must be extracted from the compressed representations of the samples learned 
# by the encoder. "model1" is created below comprising layers from the input layer to the encoder, so that the
# output of the model are the expected inputs to the second convolutional layer.
# In order to learn the filters of the second convolutional layer unsupervised, patches are once again extracted
# from these expected inputs and the learned centroids (via MiniBatchKMeans) are set as the second convolutional
# layer's filters
model1 = Model(inputs=X_input0,outputs=encoded1)
model1.compile(optimizer=opt1,loss='binary_crossentropy',metrics=['accuracy'])
# k_train1 are the expected inputs to the second convolutional layer
k_train1 = model1.predict(k_train0_t)
# "model_lstm" is the full, final model. Load the learned weights of the encoder into the corresponding layers
# of "model_lstm." 
model_lstm.load_weights('autoencoder1.hdf5',by_name=True)
# Learn filters for the second convolutional layer
centroids1_2d = ExtractPatches_KMeans(k_train1,(7,7),None,256)
# Reshape the centroids and set them as the filters of the second convolutional layer
filters = centroids1_2d.transpose((2,3,1,0))
filters = filters[numpy.newaxis,...]
layer_toset = model_lstm.get_layer('conv1_2d')
layer_toset.set_weights(filters)
# Save the final model
model_lstm.save('model_0.h5')