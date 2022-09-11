# Review

Exercises:

- Logistic Regression

  - [03a_LAB_Logistic_Regression_Error_Metrics.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/logistic-regression/03a_LAB_Logistic_Regression_Error_Metrics.ipynb)
  - [lab_jupyter_logistic_regression.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/logistic-regression/lab_jupyter_logistic_regression.ipynb)

- K Nearest Neighbor

  - [03b_LAB_KNN.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/knn/03b_LAB_KNN.ipynb)
  - [lab_jupyter_knn.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/knn/lab_jupyter_knn.ipynb)

- Support Vector Machine

  - [03c_DEMO_SVM.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/svm/03c_DEMO_SVM.ipynb)
  - [lab_jupyter_svm.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/svm/lab_jupyter_svm.ipynb)

- Decision Trees

  - [03d_LAB_Decision_Trees.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/decision-trees/03d_LAB_Decision_Trees.ipynb)
  - [lab_jupyter_decisiontree.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/decision-trees/lab_jupyter_decisiontree.ipynb)

- Ensemble Models

  - [03e_DEMO_Bagging.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/03e_DEMO_Bagging.ipynb)
  - [Bagging.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/Bagging.ipynb)
  - [03f_LAB_Boosting_and_Stacking.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/03f_LAB_Boosting_and_Stacking.ipynb)
  - [Ada_Boost.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/Ada_Boost.ipynb)
  - [Gradient_Boosting.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/Gradient_Boosting.ipynb)
  - [Stacking.ipynb](https://colab.research.google.com/github/iliyaML/ibm-machine-learning/blob/main/supervised-learning-classification/ensemble-models/Stacking.ipynb)

- lab_jupyter_model-explanations.ipynb
- lab_jupyter_imbalanced_data.ipynb

## Classification Problems

The two main types of supervised learning models are:

- Regression models, which predict a continuous outcome
- Classification models, which predict a categorical outcome.

The most common models used in supervised learning are:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machines
- Decision Tree
- Neural Networks
- Random Forests
- Boosting
- Ensemble Models

With the exception of logistic regression, these models are commonly used for both regression and classification. Logistic regression is most common for dichotomous and nominal dependent variables.

## Logistic Regression

Logistic regression is a type of regression that models the probability of a certain class occurring given other independent variables.It uses a logistic or logit function to model a dependent variable. It is a very common predictive model because of its high interpretability.

Exercises:

- 03a_LAB_Logistic_Regression_Error_Metrics.ipynb
- lab_jupyter_logistic_regression.ipynb

## Classification Error Metrics

A confusion matrix tabulates true positives, false negatives, false positives and true negatives. Remember that the false positive rate is also known as a type I error. The false negatives are also known as a type II error.

Accuracy is defined as the ratio of true postives and true negatives divided by the total number of observations. It is a measure related to predicting correctly positive and negative instances.

Recall or sensitivity identifies the ratio of true positives divided by the total number of actual positives. It quantifies the percentage of positive instances correctly identified.

Precision is the ratio of true positive divided by total of predicted positives. The closer this value is to 1.0, the better job this model does at identifying only positive instances.

Specificity is the ratio of true negatives divided by the total number of actual negatives. The closer this value is to 1.0, the better job this model does at avoiding false alarms.

The receiver operating characteristic (ROC) plots the true positive rate (sensitivity) of a model vs. its false positive rate (1-sensitivity).

The area under the curve of a ROC plot is a very common method of selecting a classification methods.T

he precision-recall curve measures the trade-off between precision and recall.

The ROC curve generally works better for data with balanced classes, while the precision-recall curve generally works better for data with unbalanced classes.

## K Nearest Neighbor Methods for Classification

K Nearest Neighbor methods are useful for classification. The elbow method is frequently used to identify a model with low K and low error rate.

These methods are popular due to their easy computation and interpretability, although it might take time scoring new observations, it lacks estimators, and might not be suited for large data sets.

Exercises:

- 03b_LAB_KNN.ipynb
- lab_jupyter_knn.ipynb

## Support Vector Machines

The main idea behind support vector machines is to find a hyperplane that separates classes by determining decision boundaries that maximize the distance between classes.

When comparing logistic regression and SVMs, one of the main differences is that the cost function for logistic regression has a cost function that decreases to zero, but rarely reaches zero. SVMs use the Hinge Loss function as a cost function to penalize misclassification. This tends to lead to better accuracy at the cost of having less sensitivity on the predicted probabilities.

Regularization can help SVMs generalize better with future data.

By using gaussian kernels, you transform your data space vectors into a different coordinate system, and may have better chances of finding a hyperplane that classifies well your data.SVMs with RBFs Kernels are slow to train with data sets that are large or have many features.

Exercises:

- 03c_DEMO_SVM.ipynb
- lab_jupyter_svm.ipynb

## Decision Trees

Decision trees split your data using impurity measures. They are a greedy algorithm and are not based on statistical assumptions.

The most common splitting impurity measures are Entropy and Gini index.Decision trees tend to overfit and to be very sensitive to different data.

Cross validation and pruning sometimes help with some of this.

Great advantages of decision trees are that they are really easy to interpret and require no data preprocessing.

Exercises:

- 03d_LAB_Decision_Trees.ipynb
- lab_jupyter_decisiontree.ipynb

## Ensemble Models

### Ensemble Based Methods and Bagging

Tree ensembles have been found to generalize well when scoring new data. Some useful and popular tree ensembles are bagging, boosting, and random forests. Bagging, which combines decision trees by using bootstrap aggregated samples. An advantage specific to bagging is that this method can be multithreaded or computed in parallel. Most of these ensembles are assessed using out-of-bag error.

Exercises:

- 03e_DEMO_Bagging.ipynb
- Bagging.ipynb

### Random Forest

Random forest is a tree ensemble that has a similar approach to bagging. Their main characteristic is that they add randomness by only using a subset of features to train each split of the trees it trains. Extra Random Trees is an implementation that adds randomness by creating splits at random, instead of using a greedy search to find split variables and split points.

Exercises:

- Ramdom_forest.ipynb

### Boosting

Boosting methods are additive in the sense that they sequentially retrain decision trees using the observations with the highest residuals on the previous tree. To do so, observations with a high residual are assigned a higher weight.

Exercises:

- 03f_LAB_Boosting_and_Stacking.ipynb
- Ada_Boost.ipynb

### Gradient Boosting

The main loss functions for boosting algorithms are:

- 0-1 loss function, which ignores observations that were correctly classified. The shape of this loss function makes it difficult to optimize.
- Adaptive boosting loss function, which has an exponential nature. The shape of this function is more sensitive to outliers.
- Gradient boosting loss function. The most common gradient boosting implementation uses a binomial log-likelihood loss function called deviance. It tends to be more robust to outliers than AdaBoost.

The additive nature of gradient boosting makes it prone to overfitting. This can be addressed using cross validation or fine tuning the number of boosting iterations. Other hyperparameters to fine tune are:

- learning rate (shrinkage)
- subsample
- number of features

Exercises:

- Gradient_Boosting.ipynb

### Stacking

Stacking is an ensemble method that combines any type of model by combining the predicted probabilities of classes. In that sense, it is a generalized case of bagging. The two most common ways to combine the predicted probabilities in stacking are: using a majority vote or using weights for each predicted probability.

Exercises:

- Stacking.ipynb

## Modeling Unbalanced Classes

Classification algorithms are built to optimize accuracy, which makes it challenging to create a model when there is not a balance across the number of observations of different classes. Common methods to approach balancing the classes are:

- Downsampling or removing observations from the most common class
- Upsampling or duplicating observations from the rarest class or classes
- A mix of downsampling and upsampling

## Modeling Approaches for Unbalanced Classes

Specific algorithms to upsample and downsample are:

- Stratified sampling
- Random oversampling
- Synthetic oversampling, the main two approaches being Synthetic Minority Oversampling Technique (SMOTE) and Adaptive Synthetic sampling (ADASYN)
- Cluster Centroids implementations like NearMiss, Tomek Links, and Nearest Neighbors

Exercises:

- lab_jupyter_model-explanations.ipynb
- lab_jupyter_imbalanced_data.ipynb
