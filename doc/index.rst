Welcome to Bag of Words's documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Classification Algorithms
============

Random Forest
**********************
In order to optimize the Random Forest classifier, we played with the following parameters:

* n_estimators which is the number of trees. Increasing this parameter, can increasing the performance but most important, it makes the predictions more stable. We choose 700 trees.
* n_jobs which is the number of processors to run in parallel for both fit and predict. We choose -1. 
* max_depth which is the maximum depth of the tree. We choose 5.
* max_features which is the number of features to try for each tree. When we increase this parameters, we also increase the performance. But we have to be concerned about the fact that if we take too many features, we decrease the diversity of each tree. We use ‘auto’ which is the similar to ‘sqrt’. 

Naive Bayes
**********************
This classifier is based on the Bayes’ Theorem. To optimize, we implement a search grid. 

* For bag of word, we use the model MultinomialNB because we count how often word occurs in the reviews. We choose alpha equal to 0.1
* For word2vec : we use the model BernoulliNB because we use feature vector with binary values. We choose alpha equal to 1.2

Logistic regression
**********************
To optimize the Logistic regression we used the following parameters:

* As penalty we used an L2 regularization to improve the generalization performance.
* Dual formulation is set to True given the L2 penalty
* The tolerance for stopping criteria was set to 0.0001
* A bias intercept was added

Simple feedforward neural network
**********************
This is a fully-connected feed-forward neural network. The network was optimized using word2vec as the vectorizer, with the simple averaging method.
 
The following characteristics of the network were set in advance:

* The two hidden layers used the Rectified Linear Units nonlinear activation function (ReLU).
* The output layer consisted of two nodes and used the sigmoid activation function.
* The optimizer was stochastic gradient descent.
* The initialisation weights were randomly drawn from a uniform distribution.

The following characteristics of the network were optimized:

* Batch size: we compared batch sizes of 16, 32, 64 and 128 samples. We selected 32 as it was the best compromise between speed and accuracy.
* Learning rate: we compared learning rates of 0.001, 0.01, 0.05 and 0.25. We obtained practically the same result for 0.01 and 0.05. The results for 0.001 and 0.25 were much were. We selected a learning rate of 0.01.
* Number of nodes in the two hidden layers. We compared the following configurations: 10/10,  12/12, 16/10, 10/16. These numbers were chosen such that the total number of parameters to be estimated by the model was always at least three times smaller than the number of available training samples (also when applying 3-fold cross-validation). The results were very similar, but we selected the 16/10 configuration as it was marginally better than the others.

Convolutional neural network
**********************
This Neural Network was optimized by implementing the following:

* A filter of size 90, a convolutional window size 6, and a case sensitive padding, that uses a bias vector, and a nonlinear activation function ReLU (Rectified Linear Units).
* A max pooling layer that reads five hundred words, and strides two words.
* A flattening layer which converts all the arrays into a single linear vector.
* A dense layer that is an artificial neural network classifier with size of 250 units, a nonlinear activation function ReLU (Rectified Linear Units), that uses a bias vector.
* A dropout layer that consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting. The network becomes less sensitive to the specific weights of neurons. This parameter is set to 0.1
* A final dense layer that is an artificial neural network classifier with size 1, and a sigmoid activation function that uses a bias vector.
* We compile this neural network using a binary cross entropy cost function, with an Adam optimizer that has a slow learning rate set to 0.0021.
* We implement an early stopping function to with patience set to 3 over 10 epochs to fit the model.


.. autoclass:: bagofwords.ReviewPreprocessor
   :members:

.. autofunction:: bagofwords.clean_up_reviews


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
