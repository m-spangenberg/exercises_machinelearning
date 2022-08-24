# Machine Learning

* Supervised Learning
  * Regression
  * Classification
* Unsupervised Learning
  * Clustering
  * Anomaly Detection
  * Dimensionality Reduction

## Supervised Learning

### Univariate Linear Regression

This is Linear Regression with one variable, for instance: The price of houses given their size.

#### Standard Notation & Terminology

* x = "input" variable feature
* y = "output" target feature
* m = number of training examples
* (x,y) = single training example
* (x<sup>(i)</sup>,y<sup>(i)</sup>) = the i<sup>th</sup> training example
* w, b = parameters, weights or coefficients
* y-hat or ŷ = prediction or estimate

#### Linear Regression Model

Our training set is fed features (x) and targets (y) and our supervised algorithm will produce some function (a hypothesis, which is our model). The job of this function is to take a new feature (input variable) and produce an estimate, also called y-hat, which is a predicted value of y in a regression equation.

f<sub>w,b</sub>(x) = wx+b

f is the function that takes x's input and depending on the values of w and b, f will output a prediction of y-hat. A simpler notation is f(x), which in this context, denotes the same thing as f sub-w,b of x.

w and b are called the `parameters` of our model. In machine learning, parameters are the variables we can adjust during training in order to improve the model's performance, these parameters are also often referred to as `coefficients` or `weights`.

ŷ<sup>(i)</sup> = f<sub>w,b</sub>(x<sup>(i)</sup>) = wx<sup>(i)</sup>+b

The formula above shows our predicted value, y-hat for the i<sup>th</sup> training sample

#### Cost Function Formula

The question is: how do we find values for parameters w and b so that y-hat is close to the true target y<sup>i</sup>?

When figuring out the cost-function, we calculate what is called the 'error' by subtracting our target feature from our prediction. We then work out the square of this deviation for every target-predictor pair in the training set, and finally sum all the squared errors of training samples in the data set up to m. It is important remember that we must compute the `average square error` instead of the total square error, we do this by dividing by m, but because of convention we use 'divides by 2 times m' which will make some of our later calculations neater. If we write our cost function as J of w,b, w and b are then our tunable parameters or weights, which we can use to reduce the cost of J of w,b.

Squared Error Cost Function:

J(w,b) = $\frac{1}{2m}$ $\sum_{i=1}^{m}$ (ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>

In machine learning there exists different cost functions for different applications, but the `squared error cost function` is the by far the most commonly used for linear regression and seems to give good results for many applications.

#### Cost Function Intuition

To recap, here's what we've seen about the cost function so far:

* model: we want to model a striaght line to a dataset with f<sub>w,b</sub>(x) = wx+b
* parameters: depending on the values chosen for w,b, we get different fit lines
* cost function: to measure how well our model fits the training data we have a cost function J(w,b) = $\frac{1}{2m}$ $\sum_{i=1}^{m}$ (ŷ<sup>(i)</sup> - y<sup>(i)</sup>)<sup>2</sup>
* goal: to try to minimize J as a function of w and b

#### Gradient Descent

Gradient Descent is an algorithm which is used extensively in machine learning, from linear regression to deep learning models, and is one of the most important building blocks in machine learning.

Overview:

We have the cost function J(w,b) that we want to minimize