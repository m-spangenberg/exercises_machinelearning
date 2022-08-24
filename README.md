# Machine Learning

Below are my notes on machine learning theory from [Coursera](https://www.coursera.org/specializations/machine-learning-introduction), [Google](https://developers.google.com/machine-learning/crash-course/), [SciKit-Learn](https://scikit-learn.org/), [Stanford Lectures](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv), and [Practical Deep Learning for Coders](https://course.fast.ai/).

* Supervised Learning
  * Linear Regression Models
  * Gradient Descent
  * Naive Bayes
  * Nearest Neighbors
  * Decision Trees
  * Classification
* Unsupervised Learning
  * Clustering
  * Anomaly Detection
  * Dimensionality Reduction

## Supervised Learning

In supervised machine learning, we're learning to create models that combine inputs to produce useful predictions on data, often previously unseen data. 

### Univariate Linear Regression

This is Linear Regression with one variable, for instance: The price of houses given their size.

#### Standard Notation & Terminology

See more [machine learning glossary](https://developers.google.com/machine-learning/glossary)

* **Features** are the input variables describing our data
  * Typically represented by the variables {x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>, ..., x<sub>n</sub>}
  * x = "input" variable feature
* **label** is the output variable we're predicting
  * Typically represented by the variable y
  * y = "output" target feature
* m = number of training examples
* **Example** is one piece or instance of data
  * a **Labeled Example** has {features, label}: (x,y)
  * These labeled examples are used to train our model.
  * (x,y) = single training example
  * (x<sup>(i)</sup>,y<sup>(i)</sup>) = the i<sup>th</sup> training example
* an **Unlabeled Example** has {features, ?}: (x,?)
  * Used for making predictions on new data (inference)
* **Model** maps examples to predicted labels
  * Defined by internal parameters, which are learned
  * w, b = parameters, weights or coefficients
  * y-hat or ŷ or y' = prediction or estimate
* **Training** means creating or learning the model. 
  * a model gradually learns the relationships between features and labels
* **Inference** means applying the trained model to unlabeled examples
* A **regression** model predicts continuous values
  * What is the value of a house?
* A **classification** model predicts discrete values
  * Is this an image of a dog, or a cat?

#### Linear Regression Model

Our training set is fed features (x) and targets (y) and our supervised algorithm will produce some function (a hypothesis, which is our model). The job of this function is to take a new feature (input variable) and produce an estimate, also called y-hat, which is a predicted value of y in a regression equation.

f<sub>w,b</sub>(x) = wx+b

f is the function that takes x's input and depending on the values of w and b, f will output a prediction of y-hat. A simpler notation is f(x), which in this context, denotes the same thing as f sub-w,b of x.

w and b are called the `parameters` of our model. In machine learning, parameters are the variables we can adjust during training in order to improve the model's performance, these parameters are also often referred to as `coefficients` or `weights`.

ŷ<sup>(i)</sup> = f<sub>w,b</sub>(x<sup>(i)</sup>) = wx<sup>(i)</sup>+b

The formula above shows our predicted value, y-hat for the i<sup>th</sup> training sample

#### Cost Function Formula

The question is: how do we find values for parameters w and b so that y-hat is close to the true target y<sup>i</sup>?

When figuring out the cost-function, we calculate what is called the 'squared error' or 'L<sub>2</sub> Loss' by subtracting our target feature from our prediction. We then work out the square of this deviation for every target-predictor pair in the training set, and finally sum all the squared errors of training samples in the data set up to m. It is important remember that we must compute the `average square error` instead of the total square error, we do this by dividing by m, but because of convention we use 'divides by 2 times m' which will make some of our later calculations neater. If we write our cost function as J of w,b, w and b are then our tunable parameters or weights, which we can use to reduce the cost of J of w,b. This process of minimizing loss is called **empirical risk minimization**.

Mean Squared Error Cost Function:

J(w,b) = $\frac{1}{2m}$ $\sum_{i=1}^{m}$ (ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>

In machine learning there exists different cost functions for different applications, but the `squared error cost function` is the by far the most commonly used for linear regression and seems to give good results for many applications.

#### Cost Function Intuition

To recap, here's what we've seen about the cost function so far:

* model: we want to model a straight line to a dataset with
  
f<sub>w,b</sub>(x) = wx+b

* parameters: depending on the values chosen for w,b, we get different fit lines
* cost function: to measure how well our model fits the training data we have a cost function
  
J(w,b) = $\frac{1}{2m}$ $\sum_{i=1}^{m}$ (ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>

* goal: to try to minimize J as a function of w and b

### Gradient Descent

Gradient Descent is an algorithm which is used extensively in machine learning, from linear regression to deep learning models, and is one of the most important building blocks in machine learning. Essentially, We have the cost function J(w,b) that we want to minimize, and it turns out we can use gradient decent to do just this to find the smallest possible cost value for J. We start off with some initial guesses for our parameters w and b, then we keep changing w and b until the cost of J settles at or near a minimum, descending downhill if you like, towards what is commonly referred to as the local minima. It's also possible for there to be more than one local minimum.

#### Implementing Gradient Descent

On each **gradient step**, w, the parameter, is updated to the old value of w minus Alpha times the term d/dw of the cost function J of wb. We are simply taking modifying our parameter w by taking the current value of w and adjusting it a small amount.

w = w-α $\frac{d}{dw}$ J(w,b)

b = b-α $\frac{d}{dw}$ J(w,b)

To break down the above equation from left to right:

* assign the product from the RHS to the LHS variable called w
* in this equation, Alpha or the symbol α, is called our learning rate
  * the learning rate is hoe aggressive the gradient descent step size is
* the [derivative term](https://en.wikipedia.org/wiki/Derivative) of the cost function J
  * the direction in which we want to step our gradient descent

We repeat the two steps shown in the equation until we reach a local minimum, also called convergence, which is when the values of w and b no longer change much in relation to their previous values. The key here is not to attempt to find the most efficient learning rate, but rather a learning rate that converges quickly enough without being too large and over-shooting the local minimum, or too small, and needing too much processing power.

#### Stochastic Gradient Descent

In gradient descent, a batch is the total number of examples you use to calculate the gradient in a single iteration and up to this point we have assumed the batch has been our entire dataset. When working at scale, data sets often contain billions or even hundreds of billions of examples, along with many redundant data-points.

It is safe to say enormous batches tend not to carry much more predictive value than large batches. What we ideally want it to get the right gradient on average for much less computation. To achieve this we can employ Stochastic Gradient Descent, the term "stochastic" means we are sampling one example comprising each batch at random. While SGD works, it can be quite noisy.

**Mini-batch Stochastic Gradient Descent** is a compromise between full-batch iteration and SGD where we sample between 10 and a 1000 examples chosen at random. Even though gradient descent so far has been focused on single features for simplicity's sake, it also works on multivariate feature sets.

* Could compute gradient over entire data set on each step, but this turns out to be unnecessary
* Computing gradient on small data samples works well
  * On every step, get a new random sample
* **Stochastic Gradient Descent**: one example at a time
* **Mini-Batch Gradient Descent**: batches of 10-1000
  * Loss & gradients are averaged over the batch

## Tensorflow

### What is TensorFlow?

TensorFlow [documentation](https://tensorflow.org/) for more details.

TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms.

TensorFlow toolkit hierarchy

* Estimators and tf.keras <-- high-level, object-oriented API
* tf.layers, tf.losses, tf.metrics, ... <-- reusable libraries for common model communication
* low-level API <-- extensive control
* CPU, GPU, TPU <-- TensorFlow code runs on these platforms

Along with TensorFlow, [NumPy](https://numpy.org/) is popularly used to simplify representing arrays and performing linear algebra operations along with [pandas](https://pandas.pydata.org/), which provides an easy way to represent datasets in memory.

## Convolutional Neural Networks for Vision Systems

### Image Classification

### Loss Functions and Optimization

### Intro to Neural Networks

### Convolutional Neural Networks (CNNs)

### Training Neural Networks I

### Training Neural Networks II

### Deep Learning Software

### CNN Architectures

### Recurrent Neural Networks

### Detection and Segmentation

### Visualizing and Understanding

### Generative Models

### Deep Reinforcement Learning

### Efficient Methods and Hardware for Deep Learning

### Adversarial Examples and Adversarial Training
