---
layout: post
title: My machine learning notes
subtitle: A brief summary to capture key concepts of machine learning algorithms
tags: [machinelearning]
comments: false
mathjax: true
---

## 1. Linear Regression

### 1.1 Expression

$$h_{\Theta}(X)=X\Theta$$

where

- $X$: matrix in n_samples x m+1_features
- $\Theta$: column vector of m+1_coefficients
- $h$: mapping function

The external n_samples x 1_feature column array contains numbers of ones so that interception ($\theta_{0}$) is fitted.

### 1.2 Cost function

$$J(\Theta)=\frac{1}{2n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)^{2}$$

and its partial derivatives

$$\frac{\partial}{\partial \theta_{j}}J(\Theta)=\frac{1}{n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)X^{(i,j)}, j=0,1,2...m$$

where

- $\theta_{j}$: $j$th element in $\Theta$
- $X^{(i)}$: the $i$th row of matrix $X$
- $X^{(i,j)}$: the $i$th-row-$j$th-column element in matrix $X$

Our goal is to obtain $min\ J(\Theta)$.

### 1.3 Batch Gradient Descent Algorithm

Update $\theta_{j}$ and compute $J(\Theta)$ until converge
$$\theta_{j}=\theta_{j}-\alpha \cdot \frac{\partial}{\partial \theta_{j}}J(\Theta), j=0,1,2...m$$

where

- $\alpha$: learning rate

Important:

- all $\theta_{j}$ should update **simultaneously**, meaning the updated values should be assigned to temperary variables first
- all examples are used in every step, namely **batch**

### 1.4 Normal Equation (A partial least method)

The general idea of normal equation is assuming minimization is achieved when partial derivatives are equal to 0:

$$\frac{\partial}{\partial \theta_{j}}J(\Theta)=0, j=0,1,2...m$$

The solution is

$$\Theta=(X^{T}X)^{-1}X^{T}y$$

Important:

- Inverse of the matrix may not exist, e.g. when number of features is much larger than that of samples, or there's redundant features in the dataset. In this case try removing some features or use regularization.
- With most implementations computing a matrix inverse grows by $O(n^{3})$, so not good if feature number is large.

### 1.5 Concepts and Definitions

- *Batch gradient descent*: compute with all the training data in each step
- *feature scaling*: normalization, zero-mean, standardization... to make it more effective to converge

## 2. Logistic Regression (A classification algorithm)

### 2.1 Expression

$$h_{\Theta}(X)=\frac{1}{1+e^{-f_{\Theta}(X)}}$$

where

- $f_{\Theta}(X)$: regress function. $f_{\Theta}(X)=X\Theta$ if it's linear.

Logistic Regression projects linear regression results with sigmoid function $g(z)=\frac{1}{1+e^{-z}}$ so that the outputs are contineous and nested within `[0, 1]` with satifactory stiffness, which are suitable for two-class classification problems.

### 2.2 Cost Function

Using max likelyhood estimation, cost function of Logistic Regression can be derived:

$$J(\Theta)=-\frac{1}{n}\left[\sum_{i=1}^{n}y^{(i)}log(h_{\Theta}(X^{(i)}))+(1-y^{(i)})log(1-h_{\Theta}(X^{(i)}))\right]$$

### 2.3 Minimize Cost with Gradient Descent Algorithm

If $f_{\Theta}(X)$ is linear, we can update $\theta_{j}$ and compute $J(\Theta)$ until converge

$$\theta_{j}=\theta_{j}-\alpha \cdot \frac{1}{n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)X^{(i,j)}, j=0,1,2...m$$

Routines and rules for Linear Regression are also applicable here.

### 2.4 Other Minimization Methods

- Conjugate gradient
- BFGS (Broyden-Fletcher-Goldfarb-Shanno)
- L-BFGS (Limited memory - BFGS)

Properties:

- No need to pick learning rate
- Generally faster than gradient descent
- Algorithms are complex

### 2.5 Multi-class classification

One vs. all method:

1. Train a logistic regression classifier $h_{\Theta}^{(i)}(X)$ for each class i to predict the probability that $y=i$.
2. On a new input, $x$, to make a prediction, pick the class i that maximizes the probability that $h_{\Theta}^{(i)}(x)=1$.

## 3. Regularization

### 3.1 Concepts and Definitions

- Overfitting: also known as high variance, which may fail to generalize
- Underfitting: also known as high bias
- Regularization: penalize parameters to overcome overfitting

### 3.2 Regularized Linear Regression

For Linear Regression, after adding regularization term, we have

$$J(\Theta)=\frac{1}{2n}\left[\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)^{2}+\lambda\cdot\sum_{j=0}^{m}\theta_{j}^{2}\right]$$
$$\frac{\partial}{\partial \theta_{j}}J(\Theta)=\frac{1}{n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)X^{(i,j)}+\frac{\lambda}{n}\cdot\theta_{j}, j=0,1,2...m$$

where regularization parameter $\lambda$ controls a trade off between variance and bias, and it should be carefully tuned.

- If $\lambda$ is too large, we penalize all parameters in $\Theta$ and make them close to 0. This ends up underfitting results.
- If $\lambda$ is too small, this will probably end up overfitting results when there are many features in the training data.
- The interception term $\theta_{0}$ is not to be penalized.

Then for gradient descent algorithm, update $\theta_{j}$ simultaneously with following until converge

$$\theta_{0}=\theta_{0}-\alpha \cdot \frac{1}{n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)$$
$$\theta_{j}=(1-\frac{\alpha\lambda}{n})\cdot\theta_{j}-\alpha \cdot \frac{1}{n}\sum_{i=1}^{n}\left(h_{\Theta}(X^{(i)})-y^{(i)}\right)X^{(i,j)}, j=1,2,3...m$$

For normal equation (also known as ridge regression), $\Theta$ is solved by

$$\Theta=(X^{T}X+\lambda I)^{-1}X^{T}y$$

where $I$ is m+1 sized square indentity matrix.

### 3.3 Regularized Logistic Regression

The gradient descent alogrithm implementation of regularized Logistic Regression is the same as regularized Linear Regression, except that their cost functions are different.

$$J(\Theta)=-\frac{1}{n}\left[\sum_{i=1}^{n}y^{(i)}log(h_{\Theta}(X^{(i)}))+(1-y^{(i)})log(1-h_{\Theta}(X^{(i)}))\right]+\frac {\lambda}{2n}\cdot\sum_{j=1}^{m}\theta_{j}^{2}$$

## 4. Neural Networks - Learn New Features On Their Own

### 4.1 Concepts and Schemes

![Neural Network](neural_networks1.png "Scheme")

where

- $x_{i}$: $i$th input
- $a_{j}^{i}$: $j$th node in $i$th layer
- $x_{0}$, $a_{0}$: bias term added for each layer before activation
- $\Theta_{jk}^{i}$: $j$th-row-$k$th-column element in weights array of $i$th layer
- $g$: activation function
- $h_{\Theta}(x)$: final node

### 4.2 Cost Function

$$J(\Theta)=-\frac{1}{n}\left[\sum_{i=1}^{n}\sum_{k=1}^{K}y^{(i)}_{k}log(h_{\Theta}(X^{(i)})_{k})+(1-y^{(i)}_{k})log(1-h_{\Theta}(X^{(i)})_{k})\right]+\frac {\lambda}{2n}\cdot\sum_{l=1}^{L-1}\sum_{i=1}^{s_{l}}\sum_{j=1}^{s_{l+1}}\left(\Theta_{ji}^{(l)}\right)^{2}$$

where

- $\Theta_{ij}^{(l)}$: Parameter used to transform $j$th input from $l$th layer to $i$th output of $l+1$th layer
- $n$: number of samples
- $K$: number of outputs (classes)
- $y^{(i)}$: $i$th target
- $h_{\Theta}(X^{(i)})_{k}$: hypophesis function that project $i$th input to $k$th output
- $\lambda$: regularization parameter
- $L$: number of layers
- $s_{l}$: count of elements in $l$th layer

### 4.3 Back Propagation Algorithm

It minimize cost function in following manner:

1. Generate outputs, e.g. $a^{(L)}$ using randomly picked $\Theta$ (use values between 0 and 1).
2. Compute errors between outputs and real targets. i.e. $\delta^{(L)}=y-a^{(L)}$.
3. Compute errors for $L-1$th layer (property of sigmoid function):

   $$\delta^{(L-1)}=(\Theta^{(L-1)})^{T}\delta^{(L)}.*[a^{(L-1)}.*(1-a^{(L-1)})]$$

4. Repeat above to calculate errors for all layers.
5. Generate error summation:

   $$\Delta_{i j}^{(l)}:=\Delta_{i j}^{(l)}+a_{j}^{(l)} \delta_{i}^{(l+1)}$$

   where $\Delta$ is initiated with matrix of zeros.

6. Repeat above to calculate for all training samples.
7. Compute

   $$D_{i j}^{(l)}:=\frac{1}{m} \Delta_{i j}^{(l)}+\lambda \Theta_{i j}^{(l)} \text { if } j \neq 0$$

   and

   $$D_{i j}^{(l)}:=\frac{1}{m} \Delta_{i j}^{(l)} \quad \text { if } j=0$$

8. Then the partial derivatives that useful for minimizing cost can be calculated by

   $$\frac{\partial}{\partial \Theta_{i j}^{(l)}} J(\Theta)=D_{ij}^{(l)}$$

9. Gradient Descent Algorithm: update $\Theta$ with the partial derivative term until cost function converges.

### 4.4 Some rules

1. Neural Network architecture
   - Number of inputs: equal to number of features in the dataset
   - Number of outputs: equal to number of classes/groups
   - Number of hidden layer: usually the larger the better, but also more computationally expensive
   - Number of nodes in hidden layer:
     - should probably keep the same number for all layers
     - might be 1.5 to 2 folds of number of inputs
2. Training
   - Random initiation of $\Theta$: small random values near zero.

## 5. Support Vector Machine

### 5.1 Concepts and definitions

- SVM is a large margin classifier. Compared to Logistic Regression, it decides separation boundary with larger minimum distance from any of the training data.
- It's sensitive to outliers.
- Faster than Neural Networks.
- More robust than Logistic Regression.

### 5.2 Cost function

1. With a linear kernel

   $$J(\Theta)=C\left[\sum_{i=1}^{n}y^{(i)}cost_{1}(\Theta^{T}X^{(i)})+(1-y^{(i)})cost_{0}(\Theta^{T}X^{(i)}))\right]+\frac {1}{2}\cdot\sum_{j=1}^{m}\theta_{j}^{2}$$

   It's a form of $J(\Theta)=CA+B$, where

   - $cost_{1}(\Theta^{T}X^{(i)})$: part of $A$ when $y^(i)=1$, approximately to 0 if $\Theta^{T}X^{(i)} \geq 1$
   - $cost_{0}(\Theta^{T}X^{(i)})$: part of $A$ when $y^(i)=0$, approximately to 0 if $\Theta^{T}X^{(i)} \leq -1$
  
   $\Theta^{T}X^{(i)}$ equals to dot product of the two vectors and can be write as $p^{i}\cdot||\Theta||$, where

   - $p^{i}$: projection of $X^{(i)}$ to $\Theta$
   - $||\Theta||$: vector length of $\Theta$. $B=\frac {1}{2} \cdot ||\Theta||^{2}$
  
   So, the minimization of $J(\Theta)$ becomes minimization of $B$, which can be achieved via maximization of $\sum_{i=1}^{n}|p^{i}|$.

1. With a Gaussian kernel

   When the decision boundary is non-linear, we need to adopt a non-linear kernel like Gaussian kernel to measure similarity.

   Then we can transfer original data $X$ into $F$ by following

   1. Take all $n$ training examples as landmarks, feature scaling should be applied first if not;
   2. For each example $x^{(i)}$, calculate similarites to each landmark $l^{(j)}$ as new features, it's $n$ features in total;
      $$f^{(i)}_{j}=e^{-\frac{||x^{(i)}-l^{(j)}||^{2}}{2\sigma^{2}}}, j=1,2,3...n$$
      $$f^{(i)}_{0}=1$$
   3. Apply SVM with linear kernel to the new data.
  
### 5.3 Some Basics

- large $C$ gives a hypothesis of overfitting (low bias high variance) and vice versa.
- small $\sigma^{2}$ also gives a hypothesis of overfitting.
- huge (e.g. 50 000+) numbers of training data will be slow to run if use Gaussian kernel.

## 6. K-means Clustering

### 6.1 Concepts and definitions

- unsupervised learning
- K: number of clusters

### 6.2 Algorithm details

Steps:

1. Randomly allocate K points as centroids of the clusters.
2. For each observation, determine the index of closest centroid and assign the observation to certain cluster.
3. For the determined K clusters, calculate their means and use as K new centroids.
4. Repeat step 2 and step 3 until converge (the centroid locations merely change over loop).

The cost function is

$$J=\frac{1}{n}\cdot\sum_{i=1}^{n}||x^{(i)}-\mu_{c^{i}}||^{2}$$

where

- $n$: number of observations
- $\mu_{c^{i}}$: centroid of cluster $c^{i}$ where $x^{(i)}$ is assigned.

### 6.3 To be noted

- K-means algorithm may converge to local optima depending on the initiation of the K centroids. To overcome this and get global optima, we run the algorithm multiple times (~50-1000) with different random centroid initiations and choose the result with minimal cost.
- Value of K can be ambiguous and is hard to choose automatically. A plot of $J$ versus K may help, but not always.

## 7. Principal Component Analysis (PCA)

### 7.1 Concepts and definitions

- dimensionality reduction
  - speed up algorithms
  - reduce space usage
  - get rid of redundent features
  - easy to visualize
  
- PCA
  - Tries to find a lower dimensional surface so the sum of squares onto that surface is minimized, whereas linear regression minimizes vertical distance
  - Features are treated equally
  - Practically, apply mean normalization and feature scaling (standardization) beforehead

### 7.2 Algorithm detail

Suppose that dataset is organized as matrix $X$ in n_observations x m_features, to reduce dimensionality from m to k, do the following:

1. Compute covariance matrix
   $$\Sigma=\frac{1}{n-1}X^{T}X$$
2. Compute eigenvectors of $\Sigma$ with one of
   - singular value decompisition
     $$[U,S,V]=svd(\Sigma)$$
     - eigenvectors are hosted as columns of $U$, choose first k columns as loadings ($L$)
     - eigenvalues are diagonal values of $S$, which stands for variances explained by certain eigenvector (also known as latents)
   - normal eigenvalues method
3. Compute $XL$ as reduced scores ($X_{p}$)

### 7.3 Notes and Advices

- choose k
  - total ratios of first k latents is reasonably large, e.g. 80% or more
  - test k until it fufills
  $$\frac{||X-X_{p}L^{T}||^{2}}{||X||^{2}} \leq 0.01$$
- bad application of PCA: prevent overfitting
- replace $X$ with its transpose after scaling and apply PCA if feature number m is much larger than observation number n. decomposition of matrix has complexity of $O(n^{3})$
- when used with other learning algorithms, scaling and decomposition should performed only on training data, then cross validation and test data use the scaling values and loadings to generate reduced features.

## 8. Anomaly Detection

### 8.1 Non-multivariate Case

Algorithm using normal distribution function

1. Choose $m$ indicative features for anomaly detection
2. Determine means $\mu$ and deviations $\sigma^{2}$ for the features
   $$\mu_{j}=\frac{1}{n}\sum_{i=1}^{n}x^{(i)}_{j}$$
   $$\sigma^{2}_{j}=\frac{1}{n}\sum_{i=1}^{n}(x^{(i)}_{j}-\mu_{j})^{2}$$
3. Given a new example $x$, compute
   $$p(x;\mu,\sigma^{2})=\prod_{j=1}^{m}\frac{1}{\sqrt{2\pi}\sigma_{j}}e^{-\frac{(x_{j}-\mu_{j})^{2}}{2\sigma_{j}^{2}}}$$
4. Predict anomaly if $p(x;\mu,\sigma^{2})<\epsilon$, where $\epsilon$ is the threshold value for determination.

### 8.2 Multivariate Case

Here we replace $\sigma^{2}$ with covariance matrix $\Sigma$ when taking variable dependences under consideration:

1. Choose $m$ indicative features for anomaly detection
2. Determine means $\mu_{1\times m}$ and covariance $\Sigma_{m \times m}$ for the features
   $$\Sigma=X^{T}X$$
   where $X$ is the training dataset in $n\_samples \times m\_features$
3. Given a new example $x_{1\times m}$, compute
   $$p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}}e^{(-1/2(x-\mu)\Sigma^{-1}(x-\mu)^T)}$$
   where $|\Sigma|$ is the determinant of $\Sigma$
4. Predict anomaly if $p(x;\mu,\Sigma)<\epsilon$, where $\epsilon$ is the threshold value for determination.

### 8.3 Comparisons and Choices

- Non-multivariate
  - assume features are independent, so need to create new features if original features have strong relations
  - compuationally cheaper
  - scales much better even number of features is large
  - fine with small dataset
  - probably used more often
  
- Multivariate
  - no need to create new features
  - less computationally efficient
  - $\Sigma$ not always has inverse, e.g. $m>>n$ or there are redundant features in the dataset

## 9. Big Data Implementation

### 9.1 Why Large Scale Datasets Matter

- high performance in most cases
- computationally expensive, however

### 9.2 Gradient Descent Algorithms

- Batch Gradient Descent: update $\theta_{j}$ with all $n$ examples in each iteration
- Stochastic Gradient Descent: update $\theta_{j}$ with 1 example in each iteration
  1. randomly shuffle all examples
  2. update $\theta_{j}$ example by example in each iteration
  3. loop over all exmaples 1-10 times (maybe)
  
- Mini-batch Gradient Descent: update $\theta_{j}$ with $p$ examples in each iteration
  1. set every $p$ examples as mini-batch
  2. update $\theta_{j}$ with the mini-batches in each iteration
  
## 10. Decision tree and its ensembling variants

> Invariant to scaling of inputs, so no careful normalization required.

### 10.1 DTs

1. ID3 Tree
   - Entropy: $H(D)=-\sum_{i=1}^{|y|}p_{i}log_{2}(p_{i})$
   - Entropy on condition $a$: $H(D|a)=\sum_{v=1}^{|y^{a}|} \frac{|D^{v}|}{D}H(D^{v})$
   - Gain: $Gain(D|a)=H(D)-H(D|a)$

   Choose $a$ with $\argmax Gain(D|a)$.

   > Gain favors features with more available values.

2. C4.5 Tree
   - Intrinsic value of feature $a$: $IV(a)=-\sum_{v=1}^{|y^{a}|} \frac{|D^{v}|}{D} log_{2} \frac{|D^{v}|}{D}$
   - Gain ratio of feature $a$: $Gain_ratio(D|a)=\frac {Gain(D|a)}{IV(a)}$

   > Gain_ratio favors features with less available values

   To overcome this, C4.5 adopts the following strategy:
   1. find features with $Gain(D|a)$ above average level
   2. choose $a$ from these features with $\argmax Gain_ratio(D|a)$

3. CART
   - $Gini(D)=1-\sum_{i=1}^{|y|}p_{i}^{2}$
   - $Gini\_index(D|a)=\sum_{v=1}^{|y^{a}|} \frac{|D^{v}|}{D}Gini(D^{v})$

   Choose $a$ with $\argmin Gini\_index(D|a)$.

   > CART gets weights in each leaf, and can be used in either classification task (Gini) or regression task (MSE).

4. Pruning: handle overfitting

   - pre-pruning: top-down strategy. Apply geenralization evaluation before and after building a node. Build the node if generalization got better, drop it if not. may cause underfitting.
   - post-pruning: bottom-up strategy. Firstly train a tree. For each node from further edge of the tree, apply generalization evaluation before and after cutting the node. Drop the node if generalization got better after cutting. Cycle the evaluation until nothing to be done. low underfitting risk, but much more computationally expensive.

### 10.2 ensembling

1. Boosting

   > Boosting is an ensemble technique where new models are added to correct the errors made by existing models. low bias high variance

   - Adaboost: use $H(x)=\sum_{t=1}^{T}\alpha_{t}h_{t}(x)$ to minimize exponential loss.
     1. train a base model.
     2. adjust sample distribution based on error of the base model, train a new model with the new data.
     3. repeat step 2 until idle
     4. line up the models
   - Gradient boost DT

   > New models are added to minimize loss function along negative gradient direction determined from existing model that trained in previous iteration.

     - XGBoost: CART-based, loss-targeted, GD-boosted
       - Model: $\hat{y}_{i}=\sum_{k=1}^{K} f_{k}\left(x_{i}\right), \quad f_{k} \in \mathcal{F}$
       - Determine: $f_{t}\left(x_{i}\right)=\hat{y}_{i}^{(t)} - \hat{y}_{i}^{(t-1)}$ in round $t$
       - Goal: $Obj^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(x_{i}\right)\right)+\Omega\left(f_{t}\right)$. With $l$=square loss and Talor expansion, we have
         $$Obj^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}_{i}^{(t-1)}\right)+g_{i} f_{t}\left(x_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(x_{i}\right)\right]+\Omega\left(f_{t}\right)$$
         where
         $$g_{i}=\partial_{\hat{y}^{(t-1)}}\left(\hat{y}^{(t-1)}-y_{i}\right)^{2}=2\left(\hat{y}^{(t-1)}-y_{i}\right)$$
         $$h_{i}=\partial_{\hat{y}^{(t-1)}}^{2}\left(y_{i}-\hat{y}^{(t-1)}\right)^{2}=2$$
     - LightGBM
  
2. Bagging

   > high bias low variance

## References

1. Stanford's machine learning course presented by Professor Andrew Ng.
2. 机器学习, 周志华, 清华大学出版社, 2016
3. dmlc/xgboost, `https://github.com/dmlc/xgboost`
