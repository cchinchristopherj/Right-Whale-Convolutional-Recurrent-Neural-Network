Whale Convolutional Recurrent Neural Network
=========================

Convolutional Recurrent Neural Network to Recognize Right Whale Upcalls 

Tackles the same [challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) and uses the same [dataset](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) as the supervised CNN model. 

Experimentation with unsupervised learning of convolutional filters (via energy-correlated receptive field grouping and 1x1 convolutions) failed to improve upon the performance of the baseline supervised CNN. In this application, two improvements will be made that will allow the model to surpass baseline performance: the use of a convolutional autoencoder for dimensionality reduction and the addition of recurrent layers for capturing informative, long-range dependencies.

Convolutional Autoencoder
=========================

According to the manifold hypothesis in machine learning, the informative variations in R^n, where n is the dimensionality of the input vectors, are concentrated on a lower-dimensional manifold, or embedding in high-dimensional space, where each dimension of the manifold corresponds to a local direction of variation that one can move along to remain on the manifold. To see this, imagine a (√n  x √n) image where the values of each of the n pixels (n dimensions) is sampled from a uniform distribution. (This represents a scenario where informative variations occur across all R^n). Note that shapes, faces, etc. will never appear in images derived in this manner because the actual probability distributions used to generate these structured arrangements are much more concentrated, i.e. their informative variations are located on a much lower-dimensional manifold. The goal of (undercomplete) autoencoders is to learn the coordinates on and structure of these lower-dimensional manifolds: in their design, autoencoders are tasked with learning a mapping from X -> X, in the process identifying a compressed, latent representation containing enough information to accurately reconstruct the original inputs. These compressed representations can provide more informative features for neural network models to make their classification predictions. 

Note that this projection of data into a lower-dimensional space is very reminiscent of PCA - in fact, if the decoder activation functions are the identity and the loss function is the least squares criterion, the subspace learned by the undercomplete autoencoder is exactly the same as that learned by PCA. The advantage of autoencoders becomes more clear when nonlinear activation functions are used in the encoder and decoder functions. In this case, the autoencoders can discover lower-dimensional nonlinear manifolds and nonlinear variations in the vectors that may improve the informational content of the extracted features. 

One can see how this concept of autoencoders can easily be extended to CNNs, allowing for the learning of more informative, compressed representations of images. The first half of the autoencoder (which maps from the input space to the latent representation) is known as the "encoder" and typically consists of several convolutional layers that perform downsampling to reduce dimensionality, while the second half of the autoencoder (which maps from the latent representation back to the input space) is known as the "decoder" and typically consists of several deconvolutional layers that perform upsampling back to the original dimensionality. (It can be shown that deconvolution, or the transposed convolution, functions as a "reverse" convolution by interchanging the forward and backward propagation steps of the convolution operation. In other words, the backpropagation step of convolution, used to compute the weight updates during gradient descent, is the forward propagation step of deconvolution and vice-versa).  

In this application, a convolutional autoencoder is tasked with reducing the dimensionality of the output of the model's first convolutional layer, in order to improve the efficacy of filter learning via K-Means for the second convolutional layer. The convolutional autoencoder's encoder is comprised of several 1x1 convolutional layers, which sequentially halve the number of channels until the number of channels is reduced to 1 (all while preserving the height and width dimensions of the original inputs). In this way, the encoder collapses the original input feature map volume into one "summary" channel that contains enough information for reconstruction. 

Once training of the autoencoder is complete, the decoder is discarded and the learned weights of the encoder are used to perform the desired dimensionality reduction of the output of the first convolutional layer. 

Recurrent Layers
=========================

A crucial missing component in all the models studied thus far is sequence modeling: none of the models have made use of the fact that the inputs (spectrogram images) are time series of frequency vectors, the long-range dependencies of which can be used to improve classification decisions. In particular, the characteristic specto-temporal shape of a right whale upcall may be better identified through sequence modeling rather than cross-correlation of small patches with convolutional filters. Recurrent layers, which are specifically designed for time series inputs (notably through use of modifiable "memory" cell states), can therefore be added to the model to improve performance. 

Many authors have combined convolutional neural networks (CNNs) and recurrent neural networks (RNNs) into what are known as CRNNs (Convolutional Recurrent Neural Networks): these models use CNNs as local feature extractors and RNNs as temporal summarizers to identify global temporal patterns, thereby benefiting from the unique advantages of both neural network architectures. Fully-connected layers are then typically placed after the RNN (with sigmoid or softmax activation function) to act as the classifier. 

In this application, a bidirectional LSTM (capable of using information from not only past time steps, but also future time steps, to make predictions) is placed after the second convolutional layer. A single, fully-connected layer that follows serves as the final classifier. 


The Kaggle challenge was to differentiate between short audio recordings of ambient noise vs right whale upcalls. A Convolutional Neural Network approach was chosen:
- A spectrogram was first taken of every audio file in the training set 
- The spectrogram's contrast was enhanced both horizontally (temporal dimension) and vertically (frequency dimension) by removing extreme values and implementing a sliding mean. Performing these two pre-processing steps separately on the original training set resulted in a new dataset twice as long.
- All spectrograms were resized to (64x64x1) to ensure uniformity in input shape and facilitate input to the CNN 
- The CNN model's hyperparameters were optimized through 3-Fold Cross Validation via GridSearchCV
- The CNN was fit using the new, double-length, contrast-enhanced training set (84000 audio files). 
- The test set consisted of 10000 audio files (~10% of the training set size) unseen by the CNN model. For prediction, a vertically-enhanced and horizontally-enhanced spectrogram was produced for each audio file and fed into the CNN. A "union" approach was taken, i.e. if the vertically-enhanced input yielded 1 and the horizontally-enhanced input yielded 0, the final predicted label was 1. 
- Due to the unbalance between the sizes of the positive class (right whale upcall) and negative class (ambient noise), with the negative class being significantly larger, accuracy is not as useful a metric for model evaluation. (For example, a model that consistently predicted the negative class would yield a high accuracy, but fail to ever predict the positive class). The Receiver Operating Characteristic (ROC) curve was instead chosen for evaluation, being a measure of the true positive rate vs false positive rate as the discrimination threshold of the binary classifier is varied. The area under the curve (AUC) is a single number metric of a binary classifier's ROC curve and it is this ROC-AUC score that is used for evaluation of the CNN model. 
- The final ROC-AUC score for the test set after 45 epochs and a batch size of 100 was: **98.25%**

Modules and Installation Instructions
=========================

**"Standard" Modules Used (Included in Anaconda Distribution):** numpy, matplotlib, pylab, glob, aifc, os

If necessary, these modules can also be installed via PyPI. For example, for the "numpy" module: 

        pip install numpy

**Additional Modules used:** skimage, sklearn, keras

skimage and sklearn can be installed via PyPI. For example, for the "scikit-image" module:

        pip install scikit-image

For Keras, follow the instructions given in the [documentation](https://keras.io/#installation). Specifically, install the TensorFlow backend and, of the optional dependencies, install HDF5 and h5py if you would like to load and save your Keras models. 

Correct Usage
=========================

My trained CNN model architecture and weights are saved in the "whale_cnn.h5" file. This file can be loaded using the command:

    loaded_model = load_model('whale_cnn.h5')  
    
Note: "load_model" is a function from keras.models. 
With this model loaded, you can follow the procedure as described in training.py to predict the label of a new audio file that may or may not contain a right whale upcall. 
Note: Currently, code is not streamlined to predict the label of a new audio file not originating from train_2.zip (i.e. a new audio file from the user). A future implementation will most likely use sklearn's Pipeline to streamline this prediction process by automatically taking an input audio file, producing the vertically-enhanced and horizontally-enhanced spectrograms, feeding them into the CNN, and unioning the predicted labels to produce the final predicted label. 

If you would like to replicate the process I used to train the CNN model, perform the following:
First, download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) to the desired directory on your computer.
Then run:

    python training.py 
    
This trains the weights of the CNN model on the dataset. With the "model" variable referring to the sklearn-wrapped Keras model, this trained model can be saved to your computer using the command:

    model.model.save('whale_cnn.h5')  
    
Note: Since "model" has an sklearn wrapper, you must use model.model.save instead of model.save (as you would for a normal Keras model) to save it to your computer. 

Code is also provided demonstrating my hyperparameter optimization process via GridSearchCV. If you would like to replicate this procedure to tune the number of neurons in the CNN model's fully-connected layer, first download the training set "train_2.zip" from [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) if not previously downloaded. 
Then run:

    python tuning.py 
    
This prints the best score and corresponding parameter (best # of neurons) found by GridSearchCV, along with the mean scores and standard deviation of the scores found for all of the other parameters. 
Note: For simplicity, these "scores" produced by GridSearchCV were accuracy-based, which is not ideal metric due to the nature of this particular dataset, as described previously. A future implementation will use the ROC-AUC scores for hyperparameter tuning. 

Results of Pre-Processing
=========================

Pre-Processing involved generation of spectrograms with vertically-enhanced contrast and horizontally-enhanced contrast for input to the CNN. Below is an example of an audio file with right whale upcall present before and after pre-processing.

- Spectrogram of an audio file with right whale upcall (original before pre-processing):
![upcall_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_original%5B93%5D.png)

*Note: You can see a faint diagonal upward trajectory in the spectrogram from about 0.50 to 1.25 seconds that indicates the 
presence of the right whale upcall. This feature was enhanced in the following two images.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![upcall_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_v_enhanced%5B93%5D.png)

*Note: The vertically-enhanced contrast emphasizes the upcall feature in the frequency domain, removing extreme values and 
local means to facilitate classification.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![upcall_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Upcall/upcall_h_enhanced%5B93%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the upcall feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For comparison, below is an example of an audio file with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient1_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_original%5B0%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient1_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_v_enhanced%5B0%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.8 to 1.0 seconds is emphasized. By comparison with the spectrograms of ambient noise (2 of 2), which empahsize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient1_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient1/ambient1_h_enhanced%5B0%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

For further comparison, below is another example of an audio with right whale upcall absent (solely ambient noise) before and after pre-processing.

- Spectrogram of an audio file with right whale upcall absent (original before pre-processing):
![ambient2_original](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_original%5B1%5D.png)

*Note: At first glance with no pre-processing, there are no distinguishable features in the image, only a background
of low frequency ambient ocean noise.*

- Spectrogram of the original audio file with vertically-enhanced contrast via the "SlidingWindowV" function in "whale_cnn.py":
![ambient2_v_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_v_enhanced%5B1%5D.png)

*Note: The vertically-enhanced contrast removes extreme values and local means to facilitate classification. A small noise
feature from 0.50 to 1.50 seconds is emphasized. By comparison with the spectrograms of ambient noise (1 of 2), which emphasize a different shape of noise feature, there are a large variety of noise feature shapes that the classifier must learn to distinguish from features characteristic of right whale upcalls.*

- Spectrogram of the original audio file with horizontally-enhanced contrast via the "SlidingWindowH" function in "whale_cnn.py":
![ambient2_h_enhanced](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Ambient2/ambient2_h_enhanced%5B1%5D.png)

*Note: The horizontally-enhanced contrast emphasizes the noise feature in the temporal domain and offers a different 
perspective of the same spectrogram for input to the classifier.*

Results of Tuning
=========================

Hyperparameter Optimization was conducted using GridSearchCV's default 3-Fold Cross Validation 
As an example, the number of neurons in the fully-connected layer ("fc1" in the diagram below) was tuned using this method in tuning.py. The printed results were as follows, with the best accuracy associated with 200 neurons. This is the number of neurons used in the final model architecture. 

| Hyperparameters          | Average AUC Score  | 
|--------------------------|--------------------|
| f=9; K[1]=64; K[2]=256   | 0.9460             | 
| f=5; K[1]=64; K[2]=128   | 0.9389             | 
| f=9; K[1]=128; K[2]=256  | 0.9460             | 
| f=7; K[1]=64; K[2]=128   | 0.9437             | 
| f=5; K[1]=256; K[2]=512  | 0.9611             | 
| f=9; K[1]=256; K[2]=128  | 0.9697             | 
| f=7; K[1]=256; K[2]=128  | 0.9308             | 
| f=7; K[1]=256; K[2]=256  | 0.9461             | 
| f=5; K[1]=256; K[2]=128  | 0.9607             | 
| f=9; K[1]=256; K[2]=512  | 0.9451             | 
| f=7; K[1]=128; K[2]=256  | 0.9632             | 
| f=7; K[1]=256; K[2]=512  | 0.9583             | 
| f=9; K[1]=128; K[2]=512  | 0.9695             | 
| f=5; K[1]=64; K[2]=256   | 0.9587             | 
| f=5; K[1]=128; K[2]=128  | 0.9567             | 
| f=5; K[1]=256; K[2]=256  | 0.9451             | 
| f=7; K[1]=128; K[2]=128  | 0.9611             | 
| f=7; K[1]=64; K[2]=512   | 0.9650             | 
| f=9; K[1]=128; K[2]=128  | 0.9174             | 
| f=5; K[1]=128; K[2]=512  | 0.9691             | 

**Best: Neurons = 200, Mean = 0.9633, Std = 0.003427**


The final model architecture was based off of the results of hyperparameter optimization, as well as the approach taken by June Zbontar (5th in the Competition) in [The Marinexplore and Cornell University Whale Detection Challenge.](https://www.kaggle.com/c/whale-detection-challenge) The final tuned architecture is depicted below: 

![cnn_architecture](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/cnn_architecture.png)

Filter Visualization (0th Layer)
=========================

The filters of the 0th convolutional layer in CNNs (applied to the raw input images) are often "human-interpretable" and have patterns that are easy to correlate with patterns of the input images.

![filters_sup](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/filters_sup.png)

*Note: Many patches appear to have patterns from the higher-intensity, brightly yellow-colored areas of the spectrogram containing a right whale upcall.*

Results of Training
=========================

The CNN model was trained for 14 epochs and a batch size of 100 on a training set of 84000 audio files (42000 vertically-enhanced spectrograms and 42000 horizontally-enhanced spectrograms). Training took approximately 7 minutes on a Tesla K80 GPU (via FloydHub Cloud Service). The test set consisted of 10000 audio files (5000 vertically-enhanced spectrograms and 5000 horizontally-enhanced spectrograms). The loss and accuracy of the training set, and ROC-AUC score of the test set were evaluated by Keras for every epoch during training and depicted below. The final ROC-AUC score for the training set after 14 epochs was found to be 98.50%, while the ROC-AUC score for the test set was found to be 98.29%.

- ROC-AUC Score vs Epoch (Graph)

![AUC-Epoch_ModelSup](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/AUC-Epoch_ModelSup.png)

- ROC-AUC Score vs Epoch (Table)

| Epoch                 | Loss        | Accuracy    | ROC-AUC     | 
|-----------------------|-------------|-------------|-------------|
| 1/14                  | 0.1606      | 0.9462      | 0.9548      | 
| 2/14                  | 0.1092      | 0.9613      | 0.9534      | 
| 3/14                  | 0.0992      | 0.9646      | 0.9712      | 
| 4/14                  | 0.0953      | 0.9660      | 0.9782      | 
| 5/14                  | 0.0922      | 0.9665      | 0.9749      | 
| 6/14                  | 0.0899      | 0.9673      | 0.9571      | 
| 7/14                  | 0.0875      | 0.9678      | 0.9751      | 
| 8/14                  | 0.0855      | 0.9684      | 0.9825      | 
| 9/14                  | 0.0852      | 0.9687      | 0.9798      | 
| 10/14                 | 0.0847      | 0.9690      | 0.9662      | 
| 11/14                 | 0.0832      | 0.9698      | 0.9775      | 
| 12/14                 | 0.0836      | 0.9689      | 0.9818      | 
| 13/14                 | 0.0810      | 0.9700      | 0.9796      | 
| 14/14                 | 0.0812      | 0.9697      | 0.9709      | 

**Test ROC-AUC Score = 0.9829**

ROC Curves
------

- Training Set ROC Curve vs Test Set ROC Curve
![ROC_ModelSup_BP](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_BP.png)

*Note: Predictions on the test set are made using the union of the predictions on the vertically-enhanced spectrograms and horizontally-enhanced spectrograms (BP=Both Predictions).*

- Test Set ROC Curves

![ROC_ModelSup_TestOnly](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_TestOnly.png)

*Note: The three curves represent predictions only on the vertically-enhanced spectrograms in the test set (VP=Vertically-Enhanced Predictions, predictions only on the horizontally-enhanced spectrograms in the test set (HP=Horizontally-Enhanced Predictions), and the union of the predictions on both types of images (BP=Both Predictions).*

- Training Set ROC Curve vs Test Set ROC Curves

![ROC_ModelSup_All](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Neural-Network/blob/master/Images/Graphs/ROC_ModelSup_All.png)
