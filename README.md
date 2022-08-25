# Machine Learning

Below are my notes on machine learning theory from [Coursera](https://www.coursera.org/specializations/machine-learning-introduction), [Google](https://developers.google.com/machine-learning/crash-course/), [SciKit-Learn](https://scikit-learn.org/), [Stanford](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv), and [Practical Deep Learning for Coders](https://course.fast.ai/).

- [Machine Learning](#machine-learning)
  - [Standard Notation](#standard-notation)
  - [Supervised Learning](#supervised-learning)
    - [**Classification**](#classification)
    - [**Regression**](#regression)
    - [**Linear Regression Model**](#linear-regression-model)
    - [**Cost Function Formula**](#cost-function-formula)
    - [**Cost Function Intuition**](#cost-function-intuition)
    - [**Generalization**](#generalization)
  - [Unsupervised Learning](#unsupervised-learning)
    - [Clustering](#clustering)
    - [Association](#association)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Regularization](#regularization)
    - [**How to Approach Overfitting?**](#how-to-approach-overfitting)
  - [Neural Network Architectures](#neural-network-architectures)
    - [Fully-Connected Feedforward Neural Networks](#fully-connected-feedforward-neural-networks)
    - [Recurrent Neural Networks](#recurrent-neural-networks)
    - [Convolutional Neural Networks](#convolutional-neural-networks)
  - [Data Gathering](#data-gathering)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Deep Learning](#deep-learning)
    - [**Motivation**](#motivation)
    - [**Neural Networks**](#neural-networks)
    - [**Forward Propagation (FNN)**](#forward-propagation-fnn)
    - [**Back Propagation (BP)**](#back-propagation-bp)
    - [**Learning Algorithm**](#learning-algorithm)
  - [Neural Network Terminology](#neural-network-terminology)
    - [**Activation Function**](#activation-function)
    - [**Loss Function**](#loss-function)
    - [**Optimizers**](#optimizers)
    - [**Parameters and Hyperparameters**](#parameters-and-hyperparameters)
    - [**Epochs, Batches, Batch Sizes, and Iterations**](#epochs-batches-batch-sizes-and-iterations)
  - [Gradient Descent](#gradient-descent)
    - [**Implementing Gradient Descent**](#implementing-gradient-descent)
    - [**Stochastic Gradient Descent**](#stochastic-gradient-descent)
  - [Errata](#errata)
    - [Toolkits and Libraries](#toolkits-and-libraries)
      - [TensorFlow](#tensorflow)
    - [Mathematics References](#mathematics-references)

## Standard Notation

See more [machine learning glossary](https://developers.google.com/machine-learning/glossary)

* **Features** are the input variables describing our data
  * Typically represented by the variables {$x_{1}x_{2}x_{...}x_{n}$}
  * $x$ = "input" variable feature
* **label** is the output variable we're predicting
  * Typically represented by the variable $y$
  * $y$ = "output" target feature
* $m$ = number of training examples
* **Example** is one piece or instance of data
  * a **Labeled Example** has {features, label}: (x,y)
  * These labeled examples are used to train our model.
  * $(x,y)$ = single training example
  * $(x^{(i)},y^{(i)})$ = the $i^{th}$ training example
* an **Unlabeled Example** has {features, ?}: (x,?)
  * Used for making predictions on new data (inference)
* **Model** maps examples to predicted labels
  * Defined by internal parameters, which are learned
  * $w, b$ = parameters, weights or coefficients
  * y-hat or $ŷ$ or $y'$ = prediction or estimate
* **Training** means creating or learning the model. 
  * a model gradually learns the relationships between features and labels
* **Inference** means applying the trained model to unlabeled examples
* **Sum** of common terms
  * Uppercase letter Sigma $\Sigma$, used to denote a sum of multiple terms
* **Standard Deviation**
  * The lowercase letter sigma $\sigma$ is used to represent [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation)
  * a measure of the amount of variation or dispersion of a set of values
* **Partial Derivative**
  * Cursive $\partial$ usually to denote a [partial derivative](https://en.wikipedia.org/wiki/Partial_derivative)
  * a derivative is the rate of change of a function with respect to a variable
  * a partial derivative is the derivative of a function of several variables with respect to change in just one of its variables
* **Gradient**
  * used in vector calculus
  * expressed as an upside down triangle known as a Del or nabla $\nabla$
  * The vector of partial derivatives with respect to all independent variables

## Supervised Learning

* **Applications of Supervised Learning**
  * Bioinformatics
  * Object Recognition
  * Spam Detection
  * Speech Recognition

Our journey begins with Supervised Learning, the most popular sub-branches of Machine Learning. Algorithms here, are designed to learn by example and the models it produces are trained on well-labeled data. We, the supervisors, are teaching systems to create models that combine labeled inputs called examples to produce useful predictions on data, often previously unseen data.

Each example is a pair consisting of:

* Input Feature (new data, typically a [vector](https://simple.wikipedia.org/wiki/Vector))
* Target Feature (our desired output)

**During training**, Supervised Learning algorithms search for patterns that correlate with the desired output.
**After training**, takes in unseen inputs and determines which label to classify it to.

At its most basic form, a supervised learning algorithm can be written as:

$$ŷ = f(x)$$

Where $ŷ$ (y-hat) is the predicted output determined by a mapping function $f$ based on an input value $x$. The function $f$ used to produce a prediction is created by the machine learning model during training. Let's take a look at the two sub-catagories of supervised learning: Classification and Regression.

### **Classification**

A classification algorithm will take input data and assign it classes or categories, predicting discrete values. For example: emails being spam or not spam are called `binary classification problems`. The model finds features in the data that correlate to a class and creates a `mapping function`, when provided with a new email, it will use this mapping function to classify it as either spam or not spam. Another popular example of a classification problem is human handwriting, where there are many variations of human handwriting, in both cursive and print.

* **Popular Classification Algorithms**
  * Linear Classifiers
  * Support Vector Machines
  * Decision Trees
  * K-Nearest Neighbour
  * Random Forest

### **Regression**

Regression is a predictive, statistical process, where the model tries to find the important relationship between independent and dependant variables. For instance the goal of a regressive algorithm could be to predict continuous values like infections, sales, or test scores.

* **Popular Regression Algorithms**
  * Linear Regression
  * Lasso Regression
  * Multivariate Regression

### **Linear Regression Model**

Often called **Univariate Linear Regression**, or Linear Regression with one variable, for instance: The price of houses given their size. Our training set is fed features $x$ and targets $y$ and our supervised algorithm will produce some function (a hypothesis, which is our model). The job of this function is to take a new feature (input variable) and produce an estimate, also called y-hat, which is a predicted value of $y$ in a regression equation.

$$f_{wb}(x) = wx+b$$

$f$ is the function that takes $x$'s input and depending on the values of $w$ and $b$, $f$ will output a prediction of $ŷ$ (y-hat). A simpler notation is $f(x)$, which in this context, denotes the same thing as $f_{w,b}(x)$.

w and b are called the `parameters` of our model. They are the `Slope` and the `y-intercept`, respectively. In machine learning, parameters are the variables we can adjust during training in order to improve the model's performance, these parameters are also often referred to as `coefficients` or `weights`.

$$ŷ^{(i)} = f_{w,b}(x^{(i)}) = wx^{(i)}+b$$

The formula above shows our predicted value, $ŷ$ (y-hat) for the $i^{th}$ training sample

Note: Models with single features are represented by [line of best fit](https://en.wikipedia.org/wiki/Simple_linear_regression), with two features, a plane and for more than two features, a [hyperplane](https://en.wikipedia.org/wiki/Hyperplane) is used.

### **Cost Function Formula**

The question is: how do we find values for parameters w and b so that y-hat is close to the true target y<sup>i</sup>?

When figuring out the cost-function, we calculate what is called the 'squared error' or 'L<sub>2</sub> Loss' by subtracting our target feature from our prediction. We then work out the square of this deviation for every target-predictor pair in the training set, and finally sum all the squared errors of training samples in the data set up to m. It is important remember that we must compute the `average square error` instead of the total square error, we do this by dividing by m, but because of convention we use 'divides by 2 times m' which will make some of our later calculations neater. If we write our cost function as J of w,b, w and b are then our tunable parameters or weights, which we can use to reduce the cost of J of w,b. This process of minimizing loss is called **empirical risk minimization**.

Mean Squared Error Cost Function:

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (ŷ^{(i)} - y^{(i)})^2$$

In machine learning there exists different cost functions for different applications, but the `squared error cost function` is the by far the most commonly used for linear regression and seems to give good results for many applications.

### **Cost Function Intuition**

To recap, here's what we've seen about the cost function so far:

* model: we want to model a straight line to a dataset with
  
$$f_{w,b}(x) = wx+b$$

* parameters: depending on the values chosen for w,b, we get different fit lines
* cost function: to measure how well our model fits the training data we have a cost function
  
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (ŷ^{(i)} - y^{(i)})^2$$

* goal: to try to minimize J as a function of w and b

### **Generalization**

Just fitting our training data is not enough to do machine learning, in machine learning we're after generalization. If we specialize in hyper specificity, also called `over-fitting`, we risk having a model that is unable to cope with changes in data, for instance: a spam filter so precisely tuned that it doesn't pick up any new forms of spam mail.

What is a "Good" Model?

* Theoretically:
  * Interesting field: [generalization theory](https://en.wikipedia.org/wiki/Generalization_(learning))
  * Based on ideas of measuring model simplicity / complexity
* Intuition: formalization of the principle of Ockam's Razor
  * The less complex a model is, the more likely that a good empirical result is not just due to the peculiarities of our sample (A model should be as simple as possible)
* Empirically:
  * Asking: will our model do well on new samples of data?
  * Evaluating: get a new sample of data-call it a test set
  * Good performance on the test set is a useful indicator of good performance on the new data in general:
    * If the test set is large enough
    * If we don't cheat by using the test set over and over

There are three basic assumptions in all of the above:

1. We draw examples **independently and identically (i.i.d.)** at random from the distribution
2. The distribution is **stationary**: it doesn't change over time
3. We always pull from the **same distribution**: including training, validation and test sets

It is important to remember that the above assumption **can** be violated, for instance in the case of 2. people can change their shopping behavior as seasons change, and case 3. tastes and fashions can change.
https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting

## Unsupervised Learning

Unsupervised Learning is a branch of Machine Learning that is used to manifest underlying patterns in data, and is often used as part of exploratory data analysis, where large data sets often hold surprising information that would have been otherwise ignored by researchers due to the complexity of the relationships between data-points or the amount of data that needs to be processed. For instance: Clustering DNA to study evolutionary biology, customers segmentation to build marketing strategies, or pharmaceutical drug-discovery by associating molecule properties to desired outcomes. It does not use labelled data, but rather relies on the data features.

* **Applications of Unsupervised Learning**
  * AirBnb, learns to recommends housing based on past customer interaction
  * Amazon, learns purchases and recommends new purchases through association rule mining/learning
  * Credit Card Fraud Detection, learns to detect fraud based on complex rule sets

### Clustering

Clustering is the process of grouping data into different clusters or groups. The goal of this clustering is to predict continuous values such as test scores or patterns in genetic data. Good clustering will contain data-points which are as similar to each other as possible. 

Within clustering there exists Partitional Clustering and Hierarchical Clustering. **Partitional Clustering** can only have each data point associated to a single cluster wheras **Hierarchical Clustering** can have clusters within clusters with data point that may belong to many clusters, and can be organized as a tree diagram.

* **Common Clustering Algorithms**
  * K-Means
  * Expectation Maximization
  * Hierarchical CLuster Analysis (HCA)

### Association

Association attempts to find relationships between different entities, for example when looking at the purchases of shoppers, we might be able to infer what else they will purchase based on common purchasing decisions made by many other shoppers. This is called [association rule learning](https://en.wikipedia.org/wiki/Association_rule_learning).

## Reinforcement Learning

* **Applications of Reinforcement Learning**
  * Robotics and Aircraft Motion Control
  * Business Strategy and Planning
  * Traffic Light Control Systems
  * Web System Configuration

Reinforcement Learning enables an `agent` to learn in an interactive environment by trial and error based on feedback from its own actions and experiences. Like supervised learning, it uses mapping between an input and output, but unlike supervised learning where feedback is a correct set of actions, reinforcement learning uses rewards and punishments as signals for positive and negative behaviour.

Unlike unsupervised learning, the goal of reinforcement is to find a suitable model that wil maximize the total cumulative reward for the agent. We call these goal oriented algorithms, for example: they will try to maximize points won in a game over many moves, where they are penalized for wrong decisions and rewarded for right ones. RL is usually modeled as a [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) or [Q-Learning](https://en.wikipedia.org/wiki/Q-learning).

* **Key Terms**
  * State - the current situation of the agent
  * Environment - the world the agent operates in
  * Agent - the model's decisions manifested
  * Policy - maps the agents state to the agents actions
  * Action - the agents effect on the environment
  * Reward - feedback from the environment

## Regularization

A core problem in Deep Learning is to create a model that performs well on training data AND new test data. The most common problem faced in Deep Learning as it turns out, is `overfitting`. This is a situation where your model performs exceptionally well on training data, but not on new data because the model is blind to new variations. Conversely, we also have scenarios where `underfitting` can occur, which is when our model is unable to draw reasonable predictions because it under estimates the data.

### **How to Approach Overfitting?**

* Dropout
  * Good results, Very popular
  * At every iteration, randomly removes nodes and their connections
* Augmentation
  * Some application do better with more data $\rarr$ Better Model
  * Synthesize data and add it to the training set
  * Good approach for classification, used widely in object recognition
* Early Stopping
  * Commonly used, simple to implement
  * Training error decreases steadily, but validation errors increase after a certain point
  * Stop training when validation starts to increase

## Neural Network Architectures

### Fully-Connected Feedforward Neural Networks

### Recurrent Neural Networks

### Convolutional Neural Networks

## Data Gathering

## Data Preprocessing

## Model Training

## Deep Learning

What is 'Deep Learning'? A subset of Machine Learning, which itself is part of the domain of Artificial Intelligence. Where Machine Learning involves teaching computers to recognize patterns in data, Deep Learning is a Machine Learning technique that learns features and tasks directly from data. The inputs are run through often extremely complex "neural networks", with many hidden layers, hence 'deep' learning. Deep Learning can be based in supervised, unsupervised, or reinforcement learning.

### **Motivation**

Why even use Deep Learning? Traditional ML, no matter how complex, will always be machine-like. It produces systems that require domain expertise and human intervention. The key idea in DL is that by feeding our data into a neural network, the system will teach itself, requiring less direct interaction by humans. These algorithms have existed for longer than most people are aware, and have only really come into their own because of the following factors:

* An abundance of data, so called 'Big Data' (Facebook, Google, et al.)
* More computational power (GPU's, TPU's and other custom processing units)
* New software architectures (Tensorflow, PyTorch, ML tool sets and libraries)

### **Neural Networks**

What is a Neural Network? Neural Networks are constructed from neurons, like neurons in the brain, they are interconnected and layered in networks. These networks take data as input and train themselves to find patterns in data. These Neural Networks then predict outputs for similar sets of data.

![Multi Layer Neural Network](https://upload.wikimedia.org/wikipedia/commons/c/c2/MultiLayerNeuralNetworkBigger_english.png "Multi Layer Neural Network - https://commons.wikimedia.org/wiki/File:MultiLayerNeuralNetworkBigger_english.png")

### **Forward Propagation ([FNN](https://en.wikipedia.org/wiki/Feedforward_neural_network))**

$$ŷ =  \sigma \sum_{i=1}^{n}x_{i}w_{i}+b_{i}$$

In forward propagation the following processes occur:

1. $x_{i}$ : The input layer, defined as neurons $x_{1}x_{2}x_{3}x_{...}x_{n}$, receives information
2. $w_{i}$ : The input neurons connect to the next layer through weighted channels $w_{1}w_{2}w_{3}w_{...}w_{n}$
3. $b_{n}$ : The inputs are multiplied by these weights and their sum sent as input to biased neurons in the hidden layer $b_{1}b_{2}b_{3}b_{...}b_{n}$
4. $\sigma$ : The total of the biased neuron, plus the weighted sum from the original input is then passed to a [non-linear activation function](https://en.wikipedia.org/wiki/Activation_function)
5. The activation function decides if the neuron can contribute to the next layer
6. $ŷ$ : The output layer is a form of probability or estimation where the neuron with the highest value determines what the output is

Some key insights:

* **Weight** of neurons tells us how important a neuron is related to other neurons
* **Bias** is akin to opinion between related neurons and shifts $\sigma$ [right or left](https://en.wikipedia.org/wiki/Scalar_(mathematics))


### **Back Propagation ([BP](https://en.wikipedia.org/wiki/Backpropagation))**

Back propagation is much like forward propagation, except here information here goes from output layer to hidden layers. Let's assume our NN produces a prediction, which can be either right or wrong. In Back Propagation, the NN evaluates its own performance with a [Loss Function](https://en.wikipedia.org/wiki/Loss_function) in order to quantify the deviation from the expected output. This deviation is what is fed back to the hidden layers, so weights and biases can be adjusted and the training process can improve.

### **Learning Algorithm**

* Initialize network's weights and biases with random values
* Supply the input neurons with data
* Compare predicted values with expected values and calculate loss
* Perform Back Propagation to propagate loss back through the network
* Update weights and biases based on the calculated loss from a gradient descent algorithm
* Iterate through previous steps until loss is minimized

## Neural Network Terminology

Some basic terminology used in Machine and Deep Learning.

### **Activation Function**

* Introduce 'Non-Linearity' in the Network
* Decides whether a neuron can contribute (activation threshold)
* Which function should we use?
  * a. **Step Function** that is either 0 or 1 does not work in every scenario
  * b. **Linear Function** is ok, but the derivative is a constant
    * This means the gradient has no relation with $x$
    * The activation function is nothing but a linear input of the first layer, not good!
  * c. **Sigmoid Function** defined as $a(x)=$ $1\over1+e^{-x}$
    * Non-linear in nature, so we can stack layers
    * Output will be in range 0 to 1 (analog outputs)
    * Tend to have steep $Y$ values and poor response at extremes
    * [Vanishing Gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
  * d. **Tanh Function**
    * Similar to Sigmoid
    * Derivative is steeper than Sigmoid
    * Also suffers from Vanishing Gradient problem
  * e. **ReLU Function** (Rectified Linear Unit) $R(z) = \max(0,z)$
    * Non-linear (stackable neuron layers) although isn't bounded :(
    * Sparse Activation is preferred
    * [Dying ReLU problem](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)
  * f. **Leaky ReLU Function**
* Binary Classifications try: **Sigmoid**
* If there is uncertainty try: **ReLU or modified ReLU**

### **Loss Function**

* A way to quantify the deviation of the predicted output by the neural network to the expected output
* Simply put: A mathematical way of calculating how 'wrong' the output of our neural network is
* Different types of loss functions exist
  * Regression: Squared Error, Huber Loss
  * Binary Classification: Binary Cross-Entropy, Hinge Loss
  * Multi-Class Classification: Multi-Class Cross Entropy, Kullback Divergence

### **Optimizers**

* During training, we adjust parameters (weights and biases) to minimize the loss function
  * HOW do we achieve this?
    * By updating the network based on the output of the loss function
    * Loss funtion guides the optimizer
    * Descending towards the lowest `local minimum` is our process of `reducing error`
* **Gradient Descent**
  * Most popular optimizer, essentially is back propagation
  * Iterative algorithm that starts at a random point
  * Calculates what a small change in each individual weight does to the loss function
  * Adjust each parameter based on its gradient (taking small steps)
  * Iterates until it reaches the lowest point (minimum)
  * What is a gradient?
    * $\nabla f(x,y) =$ $(\frac{\partial f}{\partial x}(x,y), \frac{\partial f}{\partial y}(x,y))$
    * The Gradient of a Function is the vector of partial derivatives with respect to all independent variables.
    * Always points in the direction of the steepest increase in the function
    * There exists a global and local minimum, to avoid the local minimum we must tweak the Learning Rate
* **Learning Rate**
  * A large learning rate will overshoot the global minimum
  * A small learning rate will take very long to converge on the global minimum
* Other Optimizers
  * **Stochastic Gradient Descent**
    * Like Gradient Descent, but only uses a random subset of the training examples
    * Uses batches of examples in each pass
    * Uses momentum to accumulate gradients
    * Less computationally intensive
  * **Adagrad**
    * Adapts Learning Rate to individual features
    * Some weights will have different learning rates
    * IDeal for sparse datasets with many input examples
    * Problem: Learning Rate tends to get small with time
  * **RMSprop**
    * Specialized version of Adagrad
    * Accumulates Gradients in a fixed window instead of using momentum
    * Similar to Adaprop
  * **Adam**
    * Stands for Adaptive Moment Estimation
    * Uses the concept of momentum
      * our way of telling the network whether we want past changes affect the new change
    * Used widely in practice
* Takeaways
  * There are many optimizers, easy to be overwhelmed by the complexity of choice
  * All of them have the same goal: Minimizing the loss function
  * Trial and error will help us develop an intuition for which ones are preferable

### **Parameters and Hyperparameters**

* What are **Model Parameters**?
  * Variables internal to the neural network
  * Estimated directly from the data
  * Responsible for defining the skill of our model
  * Required by the model when making predictions
  * Not set manually
  * Saved as part of the learned model
  * Exampels: **Weights**, **Biases**
* What are **Hyperparameters**?
  * Configurations external to the neural network model
  * Value cannot be estimated directly from data
  * No easy wat to find the best value, takes trial and error
  * When a Deep Learning algorithm is tuned, we make changes to hyperparameters.
  * All manually specified parameters are hyperparameters
  * Examples: **Learning Rate**, [C](https://stackoverflow.com/questions/12809633/parameter-c-in-svm-standard-to-find-best-parameter) (penalty/error) & $\sigma$ (standard deviation) in [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

### **Epochs, Batches, Batch Sizes, and Iterations**

* **Epoch**
  * When the ENTIRE dataset is passed forward and backward through the neural network only ONCE
  * In general, we use more than one epoch to help us produce more generalized models
  * Gradient descent is iterative, one epoch is simply not enough for good fitting to occur
  * Problem: Too many epochs could result in over-fitting where our model is fragile to new data
* **Batches and Batch Sizes**
  * Often our data sets are very large, so we divide our data into manageable batches
  * Batch Size is simply the total number of training examples in a single Batch
* **Iterations**
  * The number of batches needed to complete one epoch
    * 34000 training examples in 500 batches will take 68 iterations to complete 1 epoch

## Gradient Descent

Gradient Descent is an algorithm which is used extensively in machine learning, from linear regression to deep learning models, and is one of the most important building blocks in machine learning. Essentially, We have the cost function J(w,b) that we want to minimize, and it turns out we can use gradient decent to do just this to find the smallest possible cost value for J. We start off with some initial guesses for our parameters w and b, then we keep changing w and b until the cost of J settles at or near a minimum, descending downhill if you like, towards what is commonly referred to as the local minima. It's also possible for there to be more than one local minimum.

### **Implementing Gradient Descent**

On each **gradient step**, w, the parameter, is updated to the old value of w minus Alpha times the term d/dw of the cost function J of wb. We are simply taking modifying our parameter w by taking the current value of w and adjusting it a small amount.

$$w = w-α \frac{d}{dw} J(w,b)$$

$$b = b-α \frac{d}{dw} J(w,b)$$

To break down the above equation from left to right:

* assign the product from the RHS to the LHS variable called w
* in this equation, Alpha or the symbol α, is called our learning rate
  * the learning rate is hoe aggressive the gradient descent step size is
* the [derivative term](https://en.wikipedia.org/wiki/Derivative) of the cost function J
  * the direction in which we want to step our gradient descent

We repeat the two steps shown in the equation until we reach a local minimum, also called convergence, which is when the values of w and b no longer change much in relation to their previous values. The key here is not to attempt to find the most efficient learning rate, but rather a learning rate that converges quickly enough without being too large and over-shooting the local minimum, or too small, and needing too much processing power.

### **Stochastic Gradient Descent**

In gradient descent, a batch is the total number of examples you use to calculate the gradient in a single iteration and up to this point we have assumed the batch has been our entire dataset. When working at scale, data sets often contain billions or even hundreds of billions of examples, along with many redundant data-points.

It is safe to say enormous batches tend not to carry much more predictive value than large batches. What we ideally want it to get the right gradient on average for much less computation. To achieve this we can employ Stochastic Gradient Descent, the term "stochastic" means we are sampling one example comprising each batch at random. While SGD works, it can be quite noisy.

**Mini-batch Stochastic Gradient Descent** is a compromise between full-batch iteration and SGD where we sample between 10 and a 1000 examples chosen at random. Even though gradient descent so far has been focused on single features for simplicity's sake, it also works on multivariate feature sets.

* Could compute gradient over entire data set on each step, but this turns out to be unnecessary
* Computing gradient on small data samples works well
  * On every step, get a new random sample
* **Stochastic Gradient Descent**: one example at a time
* **Mini-Batch Gradient Descent**: batches of 10-1000
  * Loss & gradients are averaged over the batch

## Errata

### Toolkits and Libraries

#### TensorFlow

What is TensorFlow?

TensorFlow [documentation](https://tensorflow.org/) for more details.

TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning algorithms.

TensorFlow toolkit hierarchy

* Estimators and tf.keras <-- high-level, object-oriented API
* tf.layers, tf.losses, tf.metrics, ... <-- reusable libraries for common model communication
* low-level API <-- extensive control
* CPU, GPU, TPU <-- TensorFlow code runs on these platforms

Along with TensorFlow, [NumPy](https://numpy.org/) is popularly used to simplify representing arrays and performing linear algebra operations along with [pandas](https://pandas.pydata.org/), which provides an easy way to represent datasets in memory.

### Mathematics References

* Glossary of [mathematical symbols](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols)
* List of mathematical symbols [by subject](https://en.wikipedia.org/wiki/List_of_mathematical_symbols_by_subject) 
* List of mathematical [constants](https://en.wikipedia.org/wiki/List_of_mathematical_constants)