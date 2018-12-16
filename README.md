Whale Convolutional Recurrent Neural Network
=========================

Convolutional Recurrent Neural Network to Recognize Right Whale Upcalls 

Tackles the same [challenge](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux) and uses the same [dataset](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) as the supervised CNN model. 

Experimentation with unsupervised learning of convolutional filters (via energy-correlated receptive field grouping and 1x1 convolutions) failed to improve upon the performance of the baseline supervised CNN. In this application, two improvements will be made that will allow the model to surpass baseline performance: the use of a convolutional autoencoder for dimensionality reduction and the addition of recurrent layers for capturing informative, long-range dependencies.

Convolutional Autoencoder
=========================

According to the manifold hypothesis in machine learning, the informative variations in R^n, where n is the dimensionality of the input vectors, are concentrated on a lower-dimensional manifold, or embedding in high-dimensional space, where each dimension of the manifold corresponds to a local direction of variation that one can move along to remain on the manifold. 

![manifold](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/manifold.png)

To see this, imagine a (√n  x √n) image where the values of each of the n pixels (n dimensions) is sampled from a uniform distribution. (This represents a scenario where informative variations occur across all R^n). Note that shapes, faces, etc. will never appear in images derived in this manner because the actual probability distributions used to generate these structured arrangements are much more concentrated, i.e. their informative variations are located on a much lower-dimensional manifold. The goal of (undercomplete) autoencoders is to learn the coordinates on and structure of these lower-dimensional manifolds: in their design, autoencoders are tasked with learning a mapping from X -> X, in the process identifying a compressed, latent representation containing enough information to accurately reconstruct the original inputs. These compressed representations can provide more informative features for neural network models to make their classification predictions. 

Note that this projection of data into a lower-dimensional space is very reminiscent of PCA - in fact, if the decoder activation functions are the identity and the loss function is the least squares criterion, the subspace learned by the undercomplete autoencoder is exactly the same as that learned by PCA. 

![pca](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/pca.png)

The advantage of autoencoders becomes more clear when nonlinear activation functions are used in the encoder and decoder functions. In this case, the autoencoders can discover lower-dimensional nonlinear manifolds and nonlinear variations in the vectors that may improve the informational content of the extracted features. 

One can see how this concept of autoencoders can easily be extended to CNNs, allowing for the learning of more informative, compressed representations of images. 

![conv_autoencoder](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/conv_autoencoder.png)

The first half of the autoencoder (which maps from the input space to the latent representation) is known as the "encoder" and typically consists of several convolutional layers that perform downsampling to reduce dimensionality, while the second half of the autoencoder (which maps from the latent representation back to the input space) is known as the "decoder" and typically consists of several deconvolutional layers that perform upsampling back to the original dimensionality. (It can be shown that deconvolution, or the transposed convolution, functions as a "reverse" convolution by interchanging the forward and backward propagation steps of the convolution operation. In other words, the backpropagation step of convolution, used to compute the weight updates during gradient descent, is the forward propagation step of deconvolution and vice-versa).  

![deconv](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/deconv.jpeg)

In this application, a convolutional autoencoder is tasked with reducing the dimensionality of the output of the model's first convolutional layer, in order to improve the efficacy of filter learning via K-Means for the second convolutional layer. The convolutional autoencoder's encoder is comprised of several 1x1 convolutional layers, which sequentially halve the number of channels until the number of channels is reduced to 1 (all while preserving the height and width dimensions of the original inputs). In this way, the encoder collapses the original input feature map volume into one "summary" channel that contains enough information for reconstruction. 

Once training of the autoencoder is complete, the decoder is discarded and the learned weights of the encoder are used to perform the desired dimensionality reduction of the output of the first convolutional layer. 

Recurrent Layers
=========================

A crucial missing component in all the models studied thus far is sequence modeling: none of the models have made use of the fact that the inputs (spectrogram images) are time series of frequency vectors, the long-range dependencies of which can be used to improve classification decisions. In particular, the characteristic specto-temporal shape of a right whale upcall may be better identified through sequence modeling rather than cross-correlation of small patches with convolutional filters. Recurrent layers, which are specifically designed for time series inputs (notably through use of modifiable "memory" cell states), can therefore be added to the model to improve performance. 

![lstm](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/lstm.png)

Many authors have combined convolutional neural networks (CNNs) and recurrent neural networks (RNNs) into what are known as CRNNs (Convolutional Recurrent Neural Networks): these models use CNNs as local feature extractors and RNNs as temporal summarizers to identify global temporal patterns, thereby benefiting from the unique advantages of both neural network architectures. 

![crnn](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/crnn.png)

Fully-connected layers are then typically placed after the RNN (with sigmoid or softmax activation function) to act as the classifier. 

In this application, a bidirectional LSTM (capable of using information from not only past time steps, but also future time steps, to make predictions) is placed after the second convolutional layer. 

![bidirectional_lstm](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/bidirectional_lstm.jpg)

A single, fully-connected layer that follows serves as the final classifier. 

Results of Tuning
=========================

Hyperparameter optimization was performed via stratified k-fold cross validation where k=3, i.e. the dataset was partitioned 3 different times and performance statistics obtained over all the folds. A grid was created of the potential values for each hyperparameter of interest:
- Patch Size (f): {5,7,9}
- Number of Filters in First Convolutional Layer (K[1]): {128, 256}
- Number of Filters in Second Convolutional Layer (K[2]): {128, 256}

Twenty random combinations were created of the selection of hyperparameters above. Each of the 20 models was trained on 10000 samples from the dataset for 2 epochs with a batch size of 100 and the Adam optimizer with α=0.0001, β_1=0.9, β_2=0.999, and an additional learning rate decay such that the learning rate for each subsequent epoch was 0.99 times the learning rate of the previous epoch. Computations were conducted on a GPU via the FloydHub Cloud service - the average AUC score, obtained by averaging each model’s AUC score when evaluated on the validation set across all 3 folds, is displayed below for the twenty models: 

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

**Final Selection: f = 7, K[1] = 256, K[2] = 256 with AUC = 0.9461 **

Many authors have found it useful, for CNNs in which filters are learned via unsupervised learning, to provide as much information as possible to the classifier by setting the number of clusters learned by K-Means (and therefore the number of convolutional filters) for each layer as large as possible. The table verifies that, in general, performance improves as the total number of filters (i.e. K[1]+K[2]) increases. In particular, many of the best performing models contained the maximum number of filters permitted by the grid in the second convolutional layer (K[2]=512) and first convolutional layer (K[1]=256). 

The final hyperparameter combination was chosen in the interest of preserving computational resources and maintaining symmetry with the baseline model: by using similar hyperparameters, a similar architecture, and similar total trainable parameter count, comparison with the baseline model’s performance was facilitated. Note that both the CRNN and the baseline model have two convolutional layers – in order to further maintain symmetry, and since the table demonstrates no clear trend in performance improvement with respect to filter size, the filter size (f) for those convolutional layers was set to 7 for the CRNN. In addition, K[1] was maintained at the maximum value of 256 in the grid, but K[2] was restricted to a (slightly less optimal) value of 256 so that the total trainable parameter count of both the baseline model and CRNN are maintained at roughly the same magnitude. 

Results of Training
=========================

Performance statistics for the baseline model were obtained using stratified k-fold cross-validation, with k=10. Computations were once again conducted on a GPU via the FloydHub Cloud Service, with ~20 minutes total computation time. Results are displayed in the table below:

Baseline Model

| Mean Training Set AUC Score | Mean Test Set AUC Score | 
|-----------------------------|-------------------------|
| 99.0212%                    | 98.5120%                | 

Performance statistics were likewise obtained for the CRNN, implementing the final hyperparameter combination chosen, using k-fold cross-validation with k=10 and ~8 hours total computation time. The results are displayed below:

Convolutional Recurrent Neural Network

| Mean Training Set AUC Score | Mean Test Set AUC Score | 
|-----------------------------|-------------------------|
| 99.0240%                    | 98.5517%                | 

The CRNN is therefore able to achieve a ~0.002% increase in performance with respect to the training set and ~0.04% increase in performance with respect to the test set. At first glance, the longer computation time, more complex architecture, and minimal increase in AUC score may lead to the conclusion that the baseline model is the more optimal of the two. However, there are several important points to note about the practical implementation of training for the CRNN, which prohibited the model from achieving optimal performance. To limit the computational resources required, only 10000 samples in the dataset were used in the unsupervised learning procedure. In other words, patches were extracted from only 10000 random samples from the training set for the K-Means algorithm to learn filters for the first convolutional layer, the same 10000 samples were used to train the autoencoder, and the same 10000 samples were used by K-Means to learn filters for the second convolutional layer. Ideally, a much larger number of samples would have been used. 

The advantage of unsupervised learning (as used in the CRNN) over supervised learning (as used in the baseline model) is, after all, the ability to use the significantly greater quantity of available unlabeled data to learn abstract, generalizable patterns. The learned convolutional filters were therefore not as generalizable or informative as the ideal and were dependent on the particular 10000 random samples given to K-Means for clustering from the training set. 

![pipeline](https://github.com/cchinchristopherj/Right-Whale-Convolutional-Recurrent-Neural-Network/blob/master/Images/pipeline.png)

In addition, in an ideal scenario, the number of filters in the second and/or first convolutional layers would have been set to a greater value such as 512, enabling an increase in features given to the classifier. A larger set of model hyperparameters could have also been further optimized and the number of epochs of no improvement for the early stopping technique extended to a larger value such as 10, which would have allowed the instances of the CRNN to train for more epochs and achieve higher AUC scores. 

An experiment, for example, was conducted in which the number of epochs of no improvement for early stopping was set to 5 (instead of the default of 1 used in the k-fold cross-validation procedure, which typically terminated training before epoch 100 and yielded the AUC scores reported in the preceding table). With the number of epochs of no improvement set to 5, model performance on the validation set continuously improved even through epoch 200, demonstrating the potential of the CRNN to achieve significantly higher AUC scores than the baseline model given greater computational resources. 

This potential of the CRNN to surpass the performance of the baseline model can most likely be attributed to its use of a Bidirectional LSTM instead of fully-connected layers. One could argue that the Bidirectional LSTM more concisely captures informative features from the dataset to give to the output classification layer: recall that the fully-connected layers in the baseline model take flattened image matrix volumes as input, thereby losing spatial information and context, while the Bidirectional LSTM in the CRNN takes time series of frequency vectors as input, thereby allowing it to learn important long-range dependencies in the spectrograms. 

Correct Usage
=========================

Due to size restrictions, it was not possible to upload the pre-defined datasets I used for training, tuning, and cross-validation. However, if you would like to replicate the process I used to obtain the preceding results, the "crnn_training.py", "crnn_tuning.py", and "crnn_cv.py" provide guidance for running the three procedures on your own training, validation, and test sets. 

The training set from the original Kaggle competition "train_2.zip" can be found [here](https://www.kaggle.com/c/the-icml-2013-whale-challenge-right-whale-redux/data) and downloaded to the desired directory on your computer.
