import numpy
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.transform import resize
import aifc
import os
from skimage.transform import resize
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
from keras.layers import TimeDistributed, Permute, Conv2DTranspose, AveragePooling2D, MaxoutDense, Bidirectional, Input, RepeatVector, Dot, Embedding, Lambda, LSTM, GRU, Dense, Reshape, Dropout, Activation, BatchNormalization, Flatten, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Concatenate, MaxPooling3D
from keras.models import load_model, Sequential, Model
from keras.optimizers import SGD, Adam
from keras.engine.topology import Layer, InputSpec
from keras.utils import plot_model
import tensorflow as tf
from matplotlib.pyplot import imshow
import keras
import keras.backend as K
from keras.utils import np_utils
K.set_image_data_format('channels_last') 

def create_model(patch_size=7,num_filters1=128,num_filters2=256):
    ''' create_model Method 
            Create model for sklearn's GridSearchCV

            Args:
                patch_size: Int tuple of height and width dimensions for desired patches
                num_filters1: Number of filters for first convolutional layer
                num_filters2: Number of filters for second convolutional layer
            Returns: 
                model: Compiled model

    '''
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


    def create_autoencoder(patch_size,num_filters1,num_filters2):
        ''' create_autoencoder Method 
                Create autoencoder model
                Args:
                    patch_size: Int tuple of height and width dimensions for desired patches
                    num_filters1: Number of filters for first convolutional layer
                    num_filters2: Number of filters for second convolutional layer
                Returns: 
                    model: Compiled model

        '''
        # Input shape for samples of the dataset
        input_shape0 = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
        # Create a Sequential Model that replicates the architecture of the original model 
        # (Sequential models are necessary for use with sklearn's GridSearchCV)
        model = Sequential()
        # Dropout
        model.add(Dropout(0.1,input_shape=input_shape0,name='drop0'))
        # Batch Norm on channels axis
        model.add(BatchNormalization(axis=1,name='bn0'))
        # First convolutional layer
        model.add(Conv2D(filters=num_filters1,strides=2,kernel_size=(patch_size,patch_size),padding='valid',data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv0_2d'))
        # Batch Norm
        model.add(BatchNormalization(axis=1,name='bn1'))
        # Create a different autoencoder architecture depending on the number of desired filters for the first 
        # convolutional layer. Recall that each layer of the encoder halves the number of channels of the preceding
        # layer and each layer of the decoder doubles the number of channels of the preceding layer. The number
        # of layers in. the encoder and decoder and the number of filters for the relevant 1x1 convolutional layers
        # will therefore differ depending on how many filters are used for the first convolutional layer
        if num_filters1==64:
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
            model.add(Conv2DTranspose(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='deconv0'))
            model.add(Conv2DTranspose(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='deconv1'))
            model.add(Conv2DTranspose(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='deconv2'))
            model.add(Conv2DTranspose(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='decoded1'))
        elif num_filters1==128:
            model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
            model.add(Conv2DTranspose(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='deconv0'))
            model.add(Conv2DTranspose(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='deconv1'))
            model.add(Conv2DTranspose(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='deconv2'))
            model.add(Conv2DTranspose(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='deconv3'))
            model.add(Conv2DTranspose(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='decoded1'))
        elif num_filters1==256:
            model.add(Conv2D(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce4'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
            model.add(Conv2DTranspose(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='deconv0'))
            model.add(Conv2DTranspose(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='deconv1'))
            model.add(Conv2DTranspose(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='deconv2'))
            model.add(Conv2DTranspose(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='deconv3'))
            model.add(Conv2DTranspose(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='deconv4'))
            model.add(Conv2DTranspose(filters=256,kernel_size=1,activation='relu',data_format='channels_first',name='decoded1'))
        # Compile the model
        opt1 = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01)
        model.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
        return model 
    
    def create_preautoencoder(patch_size,num_filters1,num_filters2):
        ''' create_preautoencoder Method 
                Create pre_autoencoder model
                Args:
                    patch_size: Int tuple of height and width dimensions for desired patches
                    num_filters1: Number of filters for first convolutional layer
                    num_filters2: Number of filters for second convolutional layer
                Returns: 
                    model: Compiled model

        '''
        # Recall that the pre_autoencoder is used to create the dataset needed to train
        # the autoencoder. (The input and output of the autoencoder are the output of the
        # first convolutional layer). The pre_autoencoder is a subset of the final model, 
        # with output of the desired dimensions for training the autoencoder
        # Input shape for samples of the training set
        input_shape0 = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
        # Sequential Model
        model = Sequential()
        # Dropout
        model.add(Dropout(0.1,input_shape=input_shape0,name='drop0'))
        # Batch Norm
        model.add(BatchNormalization(axis=1,name='bn0'))
        # First convolutional layer
        model.add(Conv2D(filters=num_filters1,strides=2,kernel_size=(patch_size,patch_size),padding='valid',data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv0_2d'))
        # Batch Norm
        model.add(BatchNormalization(axis=1,name='bn1'))
        # Compile the model
        opt1 = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01)
        model.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
        return model
    
    def create_model1(patch_size,num_filters1,num_filters2):
        ''' create_model1 Method 
                Create model1 model
                Args:
                    patch_size: Int tuple of height and width dimensions for desired patches
                    num_filters1: Number of filters for first convolutional layer
                    num_filters2: Number of filters for second convolutional layer
                Returns: 
                    model: Compiled model

        '''
        # Model1 is a subset of the final model that comprises layers from the input layer
        # to the encoder. The output of model1 is therefore used to learn filters for the
        # second convolutional layer
        # Input shape for samples of the training set
        input_shape0 = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
        # Sequential Model
        model = Sequential()
        # Dropout
        model.add(Dropout(0.1,input_shape=input_shape0,name='drop0'))
        # Batch Norm
        model.add(BatchNormalization(axis=1,name='bn0'))
        # First Convolutional layer
        model.add(Conv2D(filters=num_filters1,strides=2,kernel_size=(patch_size,patch_size),padding='valid',data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv0_2d'))
        # Batch Norm
        model.add(BatchNormalization(axis=1,name='bn1'))
        # Create a different architecture for the encoder depending on the number of filters in the first 
        # convolutional layer
        if num_filters1==64:
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
        elif num_filters1==128:
            model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
        elif num_filters1==256:
            model.add(Conv2D(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
            model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
            model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
            model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
            model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce4'))
            model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
        # Compile the model
        opt1 = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01)
        model.compile(optimizer=opt1,loss='binary_crossentropy',metrics=['accuracy'])
        return model
    
    class Reshape_Seq(Layer):
        ''' Reshape_Seq Class
        Uses Layer abstract base class to build a new layer 
        '''
        def __init__(self, **kwargs):
            super(Reshape_Seq, self).__init__(**kwargs)

        def build(self, input_shape):
            self.shape = input_shape
            super(Reshape_Seq, self).build(input_shape)

        def call(self, x):
            # Reshape the input x into the shape expected by an RNN i.e. (batch_size, time_steps, num_features)
            # Intepret the columsn as the time steps (corresponding to the time axis of the original 
            # spectrogram images) and the rows for all channels as the features (corresponding to the frequency
            # axis of the original spectrogram images)
            x_reshaped = K.reshape(x,(K.shape(x)[0],K.int_shape(x)[2],K.int_shape(x)[3]*K.int_shape(x)[1]))
            self.shape = K.int_shape(x_reshaped)
            return x_reshaped

        def compute_output_shape(self, input_shape):
            return self.shape
    
    # Input shape for samples of the training set
    input_shape0 = (X_train_n.shape[1],X_train_n.shape[2],X_train_n.shape[3])
    # Sequential Model
    model = Sequential()
    # Dropout
    model.add(Dropout(0.1,input_shape=input_shape0,name='drop0'))
    # Batch Norm
    model.add(BatchNormalization(axis=1,name='bn0'))
    # First Convolutional layer
    model.add(Conv2D(filters=num_filters1,strides=2,kernel_size=(patch_size,patch_size),padding='valid',data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv0_2d'))
    # Batch Norm
    model.add(BatchNormalization(axis=1,name='bn1'))
    # Create a different encoder architecture depending on the number of filters in the first convolutional
    # layer
    if num_filters1==64:
        model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
        model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
        model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
        model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
    elif num_filters1==128:
        model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
        model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
        model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
        model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
        model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1')) 
    elif num_filters1==256:
        model.add(Conv2D(filters=128,kernel_size=1,activation='relu',data_format='channels_first',name='reduce0'))
        model.add(Conv2D(filters=64,kernel_size=1,activation='relu',data_format='channels_first',name='reduce1'))
        model.add(Conv2D(filters=32,kernel_size=1,activation='relu',data_format='channels_first',name='reduce2'))
        model.add(Conv2D(filters=16,kernel_size=1,activation='relu',data_format='channels_first',name='reduce3'))
        model.add(Conv2D(filters=8,kernel_size=1,activation='relu',data_format='channels_first',name='reduce4'))
        model.add(Conv2D(filters=1,kernel_size=1,activation='relu',data_format='channels_first',name='encoded1'))
    # Second Convolutional layer
    model.add(Conv2D(filters=num_filters2,kernel_size=(patch_size,patch_size),padding='valid',dilation_rate=2,data_format='channels_first',use_bias=False,trainable=False,activation='relu',name='conv1_2d'))
    # Max pooling
    model.add(MaxPooling2D((2,2),data_format='channels_first',name='pool1_2d'))
    # Batch Norm
    model.add(BatchNormalization(axis=1,name='bn2'))
    # Reshape_Seq Layer to reshape output of the Batch Normalization layer to the shape expected by an RNN
    model.add(Reshape_Seq(name='reshape_seq'))
    # Bidirectional LSTM
    model.add(Bidirectional(LSTM(45,return_sequences=False,name='rnn_freq')))
    # Output lyaer
    model.add(Dense(1,activation='sigmoid',name='output'))
    # Compile the model
    opt = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.01)
    model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
    opt1 = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, decay = 0.01)
    # k_train0_t are the samples of the original dataset used to learn the weights of the autoencoder
    k_train0_t = autoencoder_train_n[0:500]
    # Learn filters for the first convolutional layer
    centroids0_2d = ExtractPatches_KMeans(k_train0_t,(patch_size,patch_size),None,num_filters1)
    # Set filters for the first convolutional layer
    layer_toset = model.get_layer('conv0_2d')
    filters = centroids0_2d.transpose((2,3,1,0))
    filters = filters[numpy.newaxis,...]
    layer_toset.set_weights(filters)
    # Save the model
    model.save('/output/model.hdf5')
    # Create the pre_autoencoder using patch_size, num_filters1, and num_filters2 as specified by
    # the arguments to the create_model() method
    pre_autoencoder = create_preautoencoder(patch_size,num_filters1,num_filters2)
    pre_autoencoder.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
    pre_autoencoder.load_weights('/output/model.hdf5',by_name=True)
    pre_autoencoder_train = pre_autoencoder.predict(autoencoder_train_n)
    # Create the autoencoder using patch_size, num_filters1, and num_filters2 as specified by
    # the arguments to the create_model() method
    autoencoder = create_autoencoder(patch_size,num_filters1,num_filters2)
    autoencoder.compile(optimizer=opt1,loss='mean_squared_error',metrics=['accuracy'])
    autoencoder.load_weights('/output/model.hdf5',by_name=True)
    # Train the autoencoder
    autoencoder.fit(autoencoder_train_n,pre_autoencoder_train,epochs=5,batch_size=100)
    autoencoder.save('/output/autoencoder.hdf5')
    # Create model1 using patch_size, num_filters1, and num_filters2 as specified by the arguments 
    # to the create_model() method
    model1 = create_model1(patch_size,num_filters1,num_filters2)
    model1.compile(optimizer=opt1,loss='binary_crossentropy',metrics=['accuracy'])
    model1.load_weights('/output/autoencoder.hdf5',by_name=True)
    # k_train1 is the output of the encoder used to learn filters for the second convolutional layer
    k_train1 = model1.predict(k_train0_t)
    model.load_weights('/output/autoencoder.hdf5',by_name=True)
    # Learn and set weights for the second convolutional layer
    centroids1_2d = ExtractPatches_KMeans(k_train1,(patch_size,patch_size),None,num_filters2)
    filters = centroids1_2d.transpose((2,3,1,0))
    filters = filters[numpy.newaxis,...]
    layer_toset = model.get_layer('conv1_2d')
    layer_toset.set_weights(filters)
    return model
    
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
    X_train_n = numpy.loadtxt('/my_data/X_train_n.txt',delimiter=',')
    X_train_n = numpy.reshape(X_train_n,(76546,1,50,59))
    # Training set labels
    Y_train = numpy.loadtxt('/my_data/Y_train.txt',delimiter=',')
    # Validation set
    X_val_n = numpy.loadtxt('/my_data/X_val_n.txt',delimiter=',')
    X_val_n = numpy.reshape(X_val_n,(9568,1,50,59))
    # Validation set labels
    Y_val = numpy.loadtxt('/my_data/Y_val.txt',delimiter=',')
    # Vertical-contrast-enhanced test set
    X_testV_n = numpy.loadtxt('/my_data/X_testV_n.txt',delimiter=',')
    X_testV_n = numpy.reshape(X_testV_n,(4784,1,50,59))
    # Horizontal-contrast-enhanced test set
    X_testH_n = numpy.loadtxt('/my_data/X_testH_n.txt',delimiter=',')
    X_testH_n = numpy.reshape(X_testH_n,(4784,1,50,59))
    # Test set labels
    Y_test = numpy.loadtxt('/my_data/Y_test.txt',delimiter=',')
    # Samples from the original dataset used to train the autoencoder
    autoencoder_train_n = numpy.loadtxt('/my_data/autoencoder_train_10000.txt',delimiter=',')
    autoencoder_train_n = numpy.reshape(autoencoder_train_n,(10000,1,50,59))
    # K-Fold Cross Validation will be used to obtain performance statistics for the model. The pre-defined training 
    # and validation sets will be combined into one larger dataset used in the cross validation procedure, since
    # the partitioning process creates a "training set" (all folds used to train the model for the particular
    # iteration) and a "validation set" (the one fold not used for training). 
    X_cv_train_n = numpy.concatenate((X_train_n,X_val_n),axis=0)
    # Cross Validation dataset labels
    Y_cv_train = numpy.append(Y_train,Y_val)
    return X_train_n,X_val_n,X_testV_n,X_testH_n,Y_train,Y_val,Y_test,autoencoder_train_n,X_cv_train_n,Y_cv_train

# Load in the pre-defined datasets
X_train_n,X_val_n,X_testV_n,X_testH_n,Y_train,Y_val,Y_test,autoencoder_train_n,X_cv_train_n,Y_cv_train = data()
# Wrap the model using KerasClassifier for use with sklearn's GridSearchCV
model = KerasClassifier(build_fn=create_model,epochs=2,batch_size=100)
# Desired hyperparameters to tune with values to test
patch_size=[5,7,9]
num_filters1 = [64,128,256]
num_filters2 = [128,256,512]
# Use AUC score to determine the most optimal model
scoring = 'roc_auc'
# Create grid for hyperparameter tuning via randomized grid search
param_grid = dict(patch_size=patch_size,num_filters1=num_filters1,num_filters2=num_filters2)
grid = RandomizedSearchCV(estimator=model,param_distributions=param_grid,scoring=scoring,n_iter=20)
# For each different combination of hyperparameters to be tested, sklearn will pass the current
# settings of the hyperparameters to the create_model() method as specified above, which
# will create a different architecture depending on the hyperparameter values
grid_result = grid.fit(X_train_n[0:10000],Y_train[0:10000])
print('Best: %f using %s' % (grid_result.best_score_,grid_result.best_params_))
# Print the mean and standard deviation of AUC scores (obtained via cross-validation) for each
# combination of hyperparameters tested
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
with open('/output/grid.txt', 'a') as f:
    for mean,std,param in zip(means,stds,params):
        f.write('%f (%f) with: %r\n' % (mean,std,param))