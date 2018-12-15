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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
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

# roc_callback class from: https://github.com/keras-team/keras/issues/3230#issuecomment-319208366
class roc_callback(Callback):
    ''' roc_callback Class
        Uses keras.callbacks.Callback abstract base class to build new callbacks 
        for visualization of internal states/statistics of the CNN model 
        during training. In this case, print the roc_auc score (from sklearn)
        for every epoch during training 
    '''
    def __init__(self,training_data):
         ''' __init__ Method 
            
            Args:
                validation_data: 2 element tuple, the first element of which is the 
                test dataset and the second element of which is the test
                labels, to use for calculating the roc_auc score
        '''  
        super(roc_callback,self).__init__()
        self.x = training_data[0]
        self.y = training_data[1]
    def on_train_begin(self, logs={}):
        ''' on_train_begin Method 
            
            Args:
                logs: Dictionary containing keys for quantities relevant 
                to current epoch 
        '''  
        # Add the metric "roc_auc_val" if it does not already exist 
        if not('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
    # The following two methods are not necessary for calculating the roc_auc 
    # score. Threfore, simply return 
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        ''' on_epoch_end Method 
            
            Args:
                epoch: Current epoch number  
                logs: Dictionary containing keys for quantities relevant 
                to current epoch 
        '''  
        # Initialize the value of "roc_auc_val" to -inf so that the first calculated value 
        # registers as an improvement in the value of "roc_auc_val" for the EarlyStopping
        # Callback
        logs['roc_auc_val'] = float('-inf')
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc_val'] = roc
        print('\rroc-auc: %s' % (str(round(roc,4))),end=100*' '+'\n')
        return
    # The following two methods are not necessary for calculating the roc_auc 
    # score. Threfore, simply return 
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
    
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

def cv_trainingset(model_name,num_random_seeds,X_train,Y_train,X_testV,X_testH,Y_test,numsamples,cv_dict,class_weight_dict=dict()): 
    ''' cv_trainingset Method 
            Obtain performance statistics for the model using Repeated K-Fold Cross Validation
            
            Args:
                model_name: String name of the model to load from file
                num_random_seeds: Number of different iterations of the model to create and train
                                  with different random seeds
                X_train: 4-D matrix of the training set
                Y_train: 2-D matrix of labels for the training set
                X_testV: 4-D matrix of the vertical-contrast-enhanced test set
                X_testH: 4-D matrix of the horizontal-contrast-enhanced test set
                Y_test: 2-D matrix of labels for the test set
                numsamples: Number of times to repeat K-Fold Cross-Validation
                cv_dict: Dictionary where keys are ints corresponding to each of the desired (repeated)
                        iterations of K-Fold Cross Validation. Values are dictionaries where keys are 
                        Strings indicating the training set or validation set. Values are arrays of indices 
                        indicating which samples to use for training or validation for the current iteration 
                        of K-Fold Cross Validation.
                class_weight_dict: Dictionary indicating what weights to apply to classes during calculation
                                   of the loss function
            Returns: 
                all_dict: Dictionary where keys are Strings indicating accuracy or AUC score for the training,
                          validaiton, or test sets. Values are the actual accuracy or AUC scores corresponding
                          to the keys
                
    '''
    # Instantiate empty arrays to hold the accuracy and AUC scores for the training, validation, and test sets
    # for each of the repeated iterations of K-Fold Cross Validation.
    all_acc_train = []
    all_auc_train = []
    all_acc_val = []
    all_auc_val = []
    all_acc_test = []
    all_auc_test = []
    # Instantiate the all_dict dictionary to be output
    all_dict = dict()
    # Filepath to save the weights of the best model identified during cross-validation
    model37_filepath = '/output/weights37.best.hdf5'
    # Number of epochs to train
    num_epochs = 100
    # Outer for loop will run "numsamples" times (equal to the number of desired repeated
    # iterations of K-Fold Cross Validation)
    for ii in range(numsamples):
        # Set random seeds for this iteration
        os.environ['PYTHONHASHSEED'] = '0'
        np_random_seed = 42
        tf_random_seed = 1234
        # Inner for loop will run "num_random_seeds" times (equal to the number of desired
        # different random seeds to use when training models)
        for jj in range(num_random_seeds):
            os.environ['PYTHONHASHSEED'] = '0'
            np.random.seed(np_random_seed)
            tf.set_random_seed(tf_random_seed)
            # Note that num_random_seeds different versions of the model were created and
            # saved, each of which had its weights initialized with a different random seed. 
            # An integer (here denoted by index "jj") is used in the String name of each
            # model to indicate which random seed is used
            # Load model with desired random seed
            model_name_temp = model_name+'_'+str(jj)+'.h5'
            model = load_model(model_name_temp)
            # Use cv_dict to identify which indices of samples to use for training and
            # validation for this round of K-Fold Cross Validation
            sample_indices_train = cv_dict[ii]['train']
            sample_indices_val = cv_dict[ii]['val']
            # Use the indices to identify the corresponding samples in the training set and
            # validation set to use 
            X_train_sample = X_train[sample_indices_train]
            Y_train_sample = Y_train[sample_indices_train]
            X_val_sample = X_train[sample_indices_val]
            Y_val_sample = Y_train[sample_indices_val]
            # "validation_train" will be identical to the validation set used for this round
            # of K-Fold Cross Validation and will be used for the EarlyStopping callback. In
            # other words, when the performance of the model no longer improves for a certain
            # (specified) number of epochs, training will be stopped early to avoid overfitting.
            validation_train = X_val_sample
            validation_labels = Y_val_sample
            # Use the ModelCheckpoint callback to save the model with the best performance
            checkpoint = ModelCheckpoint(model37_filepath,monitor='roc_auc_val',verbose=1,save_best_only=True,mode='max')
            early_stopping = EarlyStopping(monitor='roc_auc_val',patience=1,mode='max')
            callbacks_list = [roc_callback(training_data=(validation_train,validation_labels)),checkpoint,early_stopping]
            # Train the model with class weights if specified
            if bool(class_weight_dict):
                model.fit(X_train_sample,Y_train_sample,callbacks=callbacks_list,epochs=num_epochs,batch_size=256,class_weight=class_weight_dict)
            else:
                model.fit(X_train_sample,Y_train_sample,callbacks=callbacks_list,epochs=num_epochs,batch_size=256)
            # After training, obtain predictions for the model on the training set, validation set,
            # vertical-contrast-enhanced test set, and horizontal-contrast-enhanced test set
            Y_pred_train = np.squeeze(model.predict(X_train_sample))
            Y_pred_val = np.squeeze(model.predict(X_val_sample))
            Y_pred_testV = np.squeeze(model.predict(X_testV))
            Y_pred_testH = np.squeeze(model.predict(X_testH))
            # Clear session of all models to save memory
            K.clear_session()
            # Calculate accuracy and AUC score for the training set and append the values to the
            # relevant arrays
            numerrors = (Y_train_sample != Y_pred_train).sum()
            acc = (len(Y_pred_train)-numerrors)/len(Y_pred_train)
            auc = roc_auc_score(Y_train_sample,Y_pred_train)
            all_acc_train.append(acc)
            all_auc_train.append(auc)
            # Calculate accuracy and AUC score for the validation set and append the values to the
            # relevant arrays
            numerrors = (Y_val_sample != Y_pred_val).sum()
            acc = (len(Y_pred_val)-numerrors)/len(Y_pred_val)
            all_acc_val.append(acc)
            all_auc_val.append(auc)
            # Recall that final predictions for the test set are the union of predictions on the
            # vertical-contrast-enhanced set and horizontal-contrast-enhanced set
            Y_pred_test = Y_pred_testV + Y_pred_testH 
            Y_pred_test[Y_pred_test>1] = 1
            # Calculate accuracy and AUC score for the test set and append the values to the
            # relevant arrays
            numerrors = (Y_test != Y_pred_test).sum()
            acc = (len(Y_pred_test)-numerrors)/len(Y_pred_test)
            auc = roc_auc_score(Y_test,Y_pred_test)
            all_acc_test.append(acc)
            all_auc_test.append(auc)
            # Decrement the random seeds so that a different random seed is used for the next round) 
            np_random_seed -= 1
            tf_random_seed -= 1
    # Calculate the mean and standard deviation of accuracy and AUC score across all rounds of 
    # Repeated K-Fold Cross Validation and store them in the relevant keys of all_dict
    all_dict['acc_train'] = [np.mean(all_acc_train),np.std(all_acc_train)]
    all_dict['auc_train'] = [np.mean(all_auc_train),np.std(all_auc_train)]
    all_dict['acc_val'] = [np.mean(all_acc_val),np.std(all_acc_val)]
    all_dict['auc_val'] = [np.mean(all_auc_val),np.std(all_auc_val)]
    all_dict['acc_test'] = [np.mean(all_acc_test),np.std(all_acc_test)]
    all_dict['auc_test'] = [np.mean(all_auc_test),np.std(all_auc_test)]
    return all_dict

# Load in the pre-defined datasets
X_train_n,X_val_n,X_testV_n,X_testH_n,Y_train,Y_val,Y_test,autoencoder_train_n,X_cv_train_n,Y_cv_train = data()
# Number of random seeds
num_random_seeds = 1
start_index = 0
num_models = 1

# Number of rounds of Repeated (Stratified) K-Fold Cross Validation, "Stratified"
# to preserve the class distribution in each fold
numsamples = 10
# Use sklearn's StratifiedKFold() to create numsamples different partitions of the 
# dataset into training and validation sets that preserve the class distribution 
skf = StratifiedKFold(n_splits=numsamples)
counter = 0
cv_dict = dict()
# Save the indices corresponding to each of the different partitions in the cv_dict
# dictionary
for train_index, test_index in skf.split(X_cv_train_n, Y_cv_train):
    cv_dict[counter] = dict()
    cv_dict[counter]['train'] = train_index
    cv_dict[counter]['val'] = test_index
    counter += 1
# Optional class weight dictionary
class_weight_dict = dict()
for ii in range(start_index,start_index+num_models):
    class_weight_dict[ii] = dict()
# Instantiate empty arrays to hold the final performance statistics for accuracy and 
# AUC score
acc_array = np.zeros((num_models,3))
auc_array = np.zeros((num_models,3))

# Use cv_trainingset() to obtain the performance statistics
for ii in range(start_index,start_index+num_models):
    model_name = '/my_model/model'
    all_dict = cv_trainingset(model_name,num_random_seeds,X_cv_train_n,Y_cv_train,X_testV_n,X_testH_n,Y_test,numsamples,cv_dict,class_weight_dict[ii])
    acc_array[ii-start_index,0] = all_dict['acc_train'][0]
    auc_array[ii-start_index,0] = all_dict['auc_train'][0] 
    acc_array[ii-start_index,1] = all_dict['acc_val'][0]
    auc_array[ii-start_index,1] = all_dict['auc_val'][0]
    acc_array[ii-start_index,2] = all_dict['acc_test'][0]
    auc_array[ii-start_index,2] = all_dict['auc_test'][0]

# Save and print the performance statistics
np.savetxt('/output/acc_array.txt',acc_array,delimiter=',')
np.savetxt('/output/auc_array.txt',auc_array,delimiter=',')
print('acc_train: '+str(all_dict['acc_train'][0]))
print('auc_train: '+str(all_dict['auc_train'][0]))
print('acc_val: '+str(all_dict['acc_val'][0]))
print('auc_val: '+str(all_dict['auc_val'][0]))
print('acc_test: '+str(all_dict['acc_test'][0]))
print('auc_test: '+str(all_dict['auc_test'][0]))