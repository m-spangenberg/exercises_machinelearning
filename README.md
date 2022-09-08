# Machine Learning

Below are my notes on machine learning and artificial intelligence from [Harvard CS50AI](https://learning.edx.org/course/course-v1:HarvardX+CS50AI+1T2020/home), [MIT OpenCourseware](https://www.youtube.com/watch?v=h0e2HAPTGF4), [Andrew Ng at Stanford](https://www.youtube.com/watch?v=jGwO_UgTS7I), [Coursera](https://www.coursera.org/specializations/machine-learning-introduction), [Google](https://developers.google.com/machine-learning/crash-course/), [SciKit-Learn](https://scikit-learn.org/), [Stanford CS231](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv), and [fast.ai](https://course.fast.ai/).

- [Machine Learning](#machine-learning)
  - [What is Machine Learning?](#what-is-machine-learning)
  - [Supervised Learning](#supervised-learning)
    - [**Classification**](#classification)
    - [**Regression**](#regression)
    - [**Linear Regression Model**](#linear-regression-model)
    - [**Vectorization**](#vectorization)
    - [**Cost Function Formula**](#cost-function-formula)
    - [**Cost Function Intuition**](#cost-function-intuition)
    - [**Cost Function Regularization**](#cost-function-regularization)
    - [**Generalization**](#generalization)
  - [Unsupervised Learning](#unsupervised-learning)
    - [**Clustering**](#clustering)
    - [**Association**](#association)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Regularization](#regularization)
    - [**How to Approach Overfitting?**](#how-to-approach-overfitting)
  - [Neural Network Architectures](#neural-network-architectures)
    - [**Fully-Connected Feedforward Neural Networks (FNN)**](#fully-connected-feedforward-neural-networks-fnn)
    - [**Recurrent Neural Networks (RNN)**](#recurrent-neural-networks-rnn)
    - [**Convolutional Neural Networks (CNN)**](#convolutional-neural-networks-cnn)
  - [Deep Learning](#deep-learning)
    - [**Motivation**](#motivation)
    - [**Neural Networks**](#neural-networks)
    - [**Forward Propagation (FP)**](#forward-propagation-fp)
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
  - [Common Image Ingest Workflows](#common-image-ingest-workflows)
  - [Deep Learning Model Deployment Workflows](#deep-learning-model-deployment-workflows)
  - [Building a Deep Learning Model](#building-a-deep-learning-model)
    - [**Data Gathering**](#data-gathering)
    - [**Data Preprocessing**](#data-preprocessing)
    - [**Model Training & Evaluation**](#model-training--evaluation)
    - [**Model Optimization**](#model-optimization)
  - [Additional Information](#additional-information)
    - [**Standard Notation**](#standard-notation)
    - [**Additional Learning Material**](#additional-learning-material)
    - [**Glossary**](#glossary)
    - [**Toolkits and Libraries**](#toolkits-and-libraries)
      - [**PyTorch**](#pytorch)
      - [**TensorFlow**](#tensorflow)
    - [**Dataset Aggregators**](#dataset-aggregators)

## What is Machine Learning?

In the late 50's, [Art Samuel](https://en.wikipedia.org/wiki/Arthur_Samuel) said Machine Learning was a "Field of study that gives computers the ability to learn without being explicitly programmed". We first have to ask ourselves, how are things learned? A human being has the ability to learn in two ways:

* **Memorization**
  * What we call **Declarative Knowledge**
  * Accumulation of individual facts
  * Limited by
    * Time to observe facts
    * Memory to store facts
* **Generalization**
  * What we call **Imperative Knowledge**
  * Deduce new facts from old facts
  * Limited by accuracy of deduction process
    * Essentially a predictive activity
    * Assumes that the past predict the future

Our interest is in extending generalization to the learning algorithm, so that it has the ability to make useful inferences about the world around it from the implicit patterns in its input data. The basic paradigm then is that we have data, which we feed to an algorithm, which produces a model, a sort of kernel of truth that is its understanding of a dataset, which can then be used to infer useful information from new data. This basic paradigm of predictions and classifications has two variations, the **supervised** and the **unsupervised** learning algorithm.

## Supervised Learning

* **Applications of Supervised Learning**
  * Bioinformatics
  * Object Recognition
  * Spam Detection
  * Speech Recognition

Our journey begins with Supervised Learning, the most popular sub-branches of Machine Learning. Algorithms here, are designed to learn by example and the models it produces are trained on well-labeled data. We, the supervisors, teach systems to create models that combine labeled inputs called examples to produce useful predictions on data, often previously unseen data.

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

The typical architecture of a classification model looks as follows. You see, for Classification problems, there exists **Binary** and **Multi-class** methods, in both cases the algorithms we employ will learn boundaries to separate our examples from other classes. 

We use Binary algorithms ([SVM](https://www.youtube.com/watch?v=efR1C6CvhmE), [logistic](https://www.youtube.com/watch?v=yIYKR4sgzI8) (without softmax), [Perceptron](https://www.youtube.com/watch?v=R2AXVUwBh7A)), for when we have classification problems where our examples must be assigned *exactly* one of two classes (for example we have a class a for honey bees, and everything else which is not a honey bee)

We use Multi-class algorithms ([NB](https://www.youtube.com/watch?v=O2L2Uv9pdDA), [kNN](https://www.youtube.com/watch?v=HVXime0nQeI), [DT](https://www.youtube.com/watch?v=_L39rN6gz7Y), [logistic](https://www.youtube.com/watch?v=yIYKR4sgzI8)) for when our example is assigned exactly one of more than two classes which can be partitioned into mutually exclusive regions.

| **Hyperparameter**                    | **Binary Classification**                   | **Multi-class Classification**  |
|---------------------------------------|---------------------------------------------|---------------------------------|
| **Input Layer Shape** (in_features)   | Same as number of features                  | _Same as binary classification_ |
| **Hidden Layer(s)**                   | Problem specific, min = 1, max = unlimited  | _Same as binary classification_ |
| **Neurons per Hidden Layer**          | Problem specific, generally 10 to 512       | _Same as binary classification_ |
| **Output Layer Shape** (out_features) | 1 (one class or the other)                  | 1 per class                     |
| **Hidden Layer Activation**           | **ReLU (rectified linear unit), etc**       | _Same as binary classification_ |
| **Output Activation**                 | **Sigmoid**                                 | **Softmax**                     |
| **Loss Function**                     | **Binary Crossentropy**                     | **Cross Entropy**               |
| **Optimizer**                         | **Stochastic Gradient Descent, Adam, etc**  | _Same as binary classification_ |
### **Regression**

Regression is a predictive, statistical process, where the model tries to find the important relationship between independent and dependant variables. For instance the goal of a regressive algorithm could be to predict continuous values like infections, sales, or test scores.

* **Popular Regression Algorithms**
  * Linear Regression
  * Lasso Regression
  * Multivariable Regression

### **Linear Regression Model**

There exists many forms of Linear Regression, let's take a look at the most basic type first.

* **Univariate Linear Regression**, where we have one feature (variable), for instance the price of a house given the size of the house.

Our training set is fed features $x$ and targets $y$ and our supervised algorithm will produce some function (a hypothesis, which is our model). The job of this function is to take a new feature (input variable) and produce an estimate, also called y-hat, which is a predicted value of $y$ in a regression equation.

$$f_{wb}(x) = wx+b$$

$f$ is the function that takes $x$'s input and depending on the values of $w$ and $b$, $f$ will output a prediction of $ŷ$ (y-hat). A simpler notation is $f(x)$, which in this context, denotes the same thing as $f_{w,b}(x)$.

w and b are called the `parameters` of our model. They are the `Slope` and the `y-intercept`, respectively. In machine learning, parameters are the variables we can adjust during training in order to improve the model's performance, these parameters are also often referred to as `coefficients` or `weights`.

$$ŷ^{(i)} = f_{w,b}(x^{(i)}) = wx^{(i)}+b$$

The formula above shows our predicted value, $ŷ$ (y-hat) for the $i^{th}$ training sample

Note: Models with single features are represented by [line of best fit](https://en.wikipedia.org/wiki/Simple_linear_regression), with two features, a plane and for more than two features, a [hyperplane](https://en.wikipedia.org/wiki/Hyperplane) is used.

Now let's take a look at a slightly more complicated form of Linear Regression.

* **Multiple Linear Regression**, with multiple features, for instance: the price of a house given the size, number of rooms, number of floors, and age.

The notation for this type of regression is similar to what we use for standard linear regression. $X_{j}$ to refer to features and $n$ to denote the total number of features. We also use $X^{(i)}$ to denote the vector of features that comprise the $i^{(th)}$ example in the dataset.

|           | **Size in Square Meters** | **Number of Bedrooms** | **Number of Floors** | **Age of building** | **Price in Dollars** |
|-----------|---------------------------|------------------------|----------------------|---------------------|----------------------|
|           | $X_{1}$ \| 2140           | $X_{2}$ \| 5           | $X_{3}$ \| 1         | $X_{4}$ \| 45       | $X_{5}$ \| 460000   |
| $i^{(2)}$ | 1416                      | 3                      | 2                    | 40                  |                      |

For this example $X^{(2)}$ is a row-vector that equals = [1416, 3, 2, 40] and if we want to refer to a specific variable value, we can use $X^{(1)}_{4}$, which equals 45.

To try and get a sense of how we might model the above example as a multi-variate equation, compared to simple Linear Regression, first observe the Linear Regression formula:

$$F_{w,b}(X)=wx+b$$

And then expand on this to make it compatible with multiple features:

$$F_{w,b}(X)=w_{1}x_{1}+w_{2}x_{2}+w_{3}x_{3}+w_{4}x_{4}+b$$

Now swap out some of those placeholder with more meaningful data, like so:

$$F_{w,b}(X)=(1.1*size)+(4*bedrooms)+(10*floors)+(-2*years)+80000$$

The numbers we've plugged into our equation are weights, for instance at $b$ we use the value 80 to denote the base price for a standard house, perhaps 80,000 is the absolute minimum needed to build a house that's up to municipal or state code in your area. The idea to drive home here, is that our model's features are modified by its weights and biases.

When we have an indeterminate amount of weights and features, a more concise way to write our formula would be to represent collections of similar elements, such as our weights, as a row-vectors. For this formula we can then write $w$ with a little arrow on top. We can do the same for $X$ and then add a dot $\cdot$ between them, this represents the dot-product of the vectors, $\vec{w}$ and $\vec{X}$. The dot-product is simply the sum of all the elements in a vector multiplied by the elements in the corresponding vector, like so: $w_{1}X_{1}+w_{2}X_{2}+w_{...}X_{...}+w_{n}X_{n}$

$$F_{\vec{w},b}(\vec{X})=\vec{w}\cdot\vec{X}+b$$

### **Vectorization**

Without vectorization, trying to derive the dot product with sequential calculation from vectors would be incredibly tedious. So using it makes our code shorter, and it runs much faster!

example: without vectorization, formula

$$F_{\vec{w},b}(\vec{X})=(\sum^{n}_{j=1}w_{j}X_{j})+b$$

example: python for-range loop, slow and inefficient
```py
f = 0
for j in range(0, n):
  f = f + w[j] * x[j]
f = f + b
```

example: python and numpy, mathe-magical!
```py
f = np.dot(w,x) + b
```

The reason `NumPy` is able to be so much faster than a pure Python implementation, is because our pure implementation is serialized, so one-step-after-another, which is highly inefficient. NumPy performs the same task in a highly `parallel` manner and leverages specialized hardware that resides on the CPU/GPU to add the resulting sum of each individual derivative term together. This is all to illustrate how vectorized code is the preferred method for implementing machine learning algorithms that are efficient and scale well with larger datasets.

example: implementing multiple linear regression in python with numpy
```py
w = np.array([0.5, 1.3, 1.6, ..., 2.0])
d = np.array([0.2, 0.5, 0.7, ..., 1.7])
w = w - 0.1 * d
```

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

### **Cost Function Regularization**



### **Generalization**

Just fitting our training data is not enough to do machine learning, in machine learning we're after generalization. If we specialize in hyper specificity, also called `over-fitting`, we risk having a model that is unable to cope with changes in data, for instance: a spam filter so precisely tuned that it doesn't pick up any new forms of spam mail, or one that produces `false-negatives`, and `false-positives`.

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

## Unsupervised Learning

Unsupervised Learning is a branch of Machine Learning that is used to manifest underlying patterns in data, and is often used as part of exploratory data analysis, where large data sets often hold surprising information that would have been otherwise ignored by researchers due to the complexity of the relationships between data-points or the amount of data that needs to be processed. For instance: Clustering DNA to study evolutionary biology, customers segmentation to build marketing strategies, or pharmaceutical drug-discovery by associating molecule properties to desired outcomes. It does not use labelled data, but rather relies on the data features.

* **Applications of Unsupervised Learning**
  * AirBnb, learns to recommends housing based on past customer interaction
  * Amazon, learns purchases and recommends new purchases through association rule mining/learning
  * Credit Card Fraud Detection, learns to detect fraud based on complex rule sets

### **Clustering**

Clustering is the process of grouping data into different clusters or groups. The goal of this clustering is to predict continuous values such as test scores or patterns in genetic data. Good clustering will contain data-points which are as similar to each other as possible. 

Within clustering there exists Partitional Clustering and Hierarchical Clustering. **Partitional Clustering** can only have each data point associated to a single cluster wheras **Hierarchical Clustering** can have clusters within clusters with data point that may belong to many clusters, and can be organized as a tree diagram.

* **Common Clustering Algorithms**
  * K-Means
  * Expectation Maximization
  * Hierarchical CLuster Analysis (HCA)

### **Association**

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

It turns out, with datasets, if you have a lot of features but not enough examples, your model will likely become overfit. Instead of using all features, it is better practice to select a few relevant features. This feature selection has a downside though, by limiting our selections to only some features, useful data might be lost.

### **How to Approach Overfitting?**

* Dropout
  * Good results, Very popular
  * At every iteration, randomly removes nodes and their connections
* Augmentation
  * Some application do better with more data result in better models
  * Synthesize data and add it to the training set
  * Good approach for classification, used widely in object recognition
* Early Stopping
  * Commonly used, simple to implement
  * Training error decreases steadily, but validation errors increase after a certain point
  * Stop training when validation starts to increase

## Neural Network Architectures

### **Fully-Connected Feedforward Neural Networks ([FNN](https://en.wikipedia.org/wiki/Feedforward_neural_network))**

By fully connected, we mean each neuron is connected to every neuron in the subsequent layer with no backwards connections. As we've found, each neuron contains an activation function that changes the output of neuron when given an input, and each type of non-linear activation function (sigmoid, tanh, and rectified linear unit) has their own pros and cons. We use them at various layers based on the problem they're each meant to solve. With these fundamentals we are capable of building a wide variety of fully-connected feed forward networks.

* Inputs
* Outputs
* Hidden Layers
* Neurons per Hidden Layer
* Activation Functions

All this allows us to build powerful deep neural networks capable of solving complex problems. The more neurons we add to each hidden layer, the wider the network becomes, and the more hidden layers we add to the network, the deeper it becomes. But there is always a complexity trade-off, and more neurons require larger amounts of computational resources. Because these networks are not linear in nature, the complexity can increase very quickly, consuming many resources and taking longer and longer to train.

### **Recurrent Neural Networks ([RNN](https://en.wikipedia.org/wiki/Recurrent_neural_network))**

* **Applications of RNNs**
  * Natural Language Processing
  * Sentiment Analysis
  * DNA Sequence Classification
  * Speech Recognition
  * Language Translation

Where Fully-Connected Feedforward Neural Networks break down is that they take a fixed-sized input and produces a fixed-sized output but are unfortunately unable to model every problem we face. Information about the past must be supplied to the network, because it can't handle sequential data points as it does not share parameters over time.

Sharing parameters gives the network the ability to look for a given feature everywhere in the sequence, rather than just a certain area. To achieve this we need a specific framework able to deal with the following:

* Deal with variable length sequences
* Maintain sequence order
* Keep track of long-term dependencies
* Share parameters across the sequence

This is where Recurrent Neural Networks come in. They can operate effectively on sequences of data with variable input length and use a feedback loop in the hidden layers allowing it to use knowledge of a previous state as input to make new predictions. We can liken this to giving the neural network a short term memory, allowing it to model sequential data.

* **How do we train an RNN?**
  * We can think of each time-step as a layer
  * Backpropagation through time (BTT) is applied for every sequence data point instead of the entire sequence
  * Gradients used to make adjustments to weights and biases, allowing it to learn
  * VGP renders RNNs unable to learn long-range dependencies

We're however faced with a problem, to have a short term memory, we must employ backpropagation which in turn causes the vanishing gradient problem. In the VGP, gradients of a layer are calculated based on the gradients of a previous layer and if the initial gradient is small, adjustments to the subsequent layers will be even smaller, giving rise to vanishing gradients.

To remedy the VGP problem, we can employ two variants of Recurring Neural Networks.

* **LSTM** - Long Short Term Memory
  * Input Gate
  * Output Gate
  * Forget Gate
* **GRNN** - Gated Recurrent Neural Network
  * Update Gate
  * Reset Gate

Both these variants are capable of learning long-term dependencies using mechanisms called `gates`. These gates are [tensor](https://en.wikipedia.org/wiki/Tensor) operations that can learn what to add or remove from the hidden state of the feedback loop.

### **Convolutional Neural Networks ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network))**

* **Applications of CNNs**
  * Computer Vision
  * Image Recognition
  * Image Processing
  * Image Segmentation
  * Video Analysis

A type of Deep Neural Network Architecture designed for specific tasks like image classification. They are inspired by the organization of neurons in the visual cortex of the human brain. They exceed at processing data like images, audio and video. Like a FNN, they are composed of an Input, Output and several Hidden neuron layers. A CNN derives its name from the hidden layers employed in its creation and can consist of:

* Convolutional Layers
* Pooling Layers
* Fully Connected Layers
* Normalization Layers

Instead of activation functions, convolution and pooling functions are used instead. The input fo the CNN is typically a 2D array of neurons, corresponding to pixels if we're doing image classification. The output is typically 1 dimensional. [Convolution](https://en.wikipedia.org/wiki/Convolution) is a technique that allows us to extract visual features from a 2D array in small chunks. Each neuron in a convolution layer is responsible for a small cluster of neurons in a preceding layer. The bounding box that determines a cluster of neurons is called a filter, also called a kernel. We can conceptualize the filter as moving across the image performing a function on individual regions of the image and then sending results to a corresponding neuron in the convolution layer. The convolution of two functions $f$ and $g$ is defined as follows, and is in fact the [dot product](https://en.wikipedia.org/wiki/Dot_product) of the input function and the kernel function.

$$(f*g)(i)=\sum_{j=1}^{m}g(j)\cdot f(i-j+m/2)$$

Pooling, also known as sub-sampling or down-sampling, is the next step. It's objective is to further reduce the numbers of neurons necessary in subsequent layers of the network, while still retaining relevant information. There exists two types of pooling: Max and Min pooling, where max pooling is used to pick the maximum value from the selected region and min the minimum.

**Computer Vision Classification Problems**

* Binary Classification <-- one type of data or another, banana _or_ apples
* Object Detection <-- specific object in an image or sequence of images
* Multi-Class Classification <-- one type of data our of many, banana, apples, boots, cars, planes
* Segmentation <-- sections of images with semantic masking

## Deep Learning

What is '[Deep Learning](https://en.wikipedia.org/wiki/Deep_learning)'? A subset of Machine Learning, which itself is part of the domain of Artificial Intelligence. Where Machine Learning involves teaching computers to recognize patterns in data, Deep Learning is a Machine Learning technique that learns features and tasks directly from data. The inputs are run through often extremely complex "neural networks", with many hidden layers, hence 'deep' learning. Deep Learning can be based in supervised, unsupervised, or reinforcement learning.

### **Motivation**

Why even use Deep Learning? Traditional ML, no matter how complex, will always be machine-like. It produces systems that require domain expertise and human intervention. The key idea in DL is that by feeding our data into a neural network, the system will teach itself, requiring less direct interaction by humans. These algorithms have existed for longer than most people are aware, and have only really come into their own because of the following factors:

* An abundance of data, so called 'Big Data' (Facebook, Google, et al.)
* More computational power (GPU's, TPU's and other custom processing units)
* New software architectures (Tensorflow, PyTorch, ML tool sets and libraries)

### **Neural Networks**

What is a Neural Network? Neural Networks are constructed from neurons, like neurons in the brain, they are interconnected and layered in networks. These networks take data as input and train themselves to find patterns in data. These Neural Networks then predict outputs for similar sets of data.

![Multi Layer Neural Network](https://upload.wikimedia.org/wikipedia/commons/c/c2/MultiLayerNeuralNetworkBigger_english.png "Multi Layer Neural Network - https://commons.wikimedia.org/wiki/File:MultiLayerNeuralNetworkBigger_english.png")

### **Forward Propagation (FP)**

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
  * Not set manually, adjusted by forward and back propagation
  * Saved as part of the learned model
  * Examples: **Weights**, **Biases**
* What are **Hyperparameters**?
  * Configurations external to the neural network model
  * Value cannot be estimated directly from data
  * Also called a model's tuning parameters
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

**Mini-batch Stochastic Gradient Descent** is a compromise between full-batch iteration and SGD where we sample between 10 and a 1000 examples chosen at random. Even though gradient descent so far has been focused on single features for simplicity's sake, it also works on multivariate feature sets. **VERY IMPORTANT**: *Try not to use minibatches larger than 32*

* Could compute gradient over entire data set on each step, but this turns out to be unnecessary
* Computing gradient on small data samples works well
  * On every step, get a new random sample
* **Stochastic Gradient Descent**: one example at a time
* **Mini-Batch Gradient Descent**: batches of 10-1000
  * Loss & gradients are averaged over the batch

## Common Image Ingest Workflows

* Gathering Data
  * Data Representation
    * Resizing
    * Cropping
    * Padding
    * Squishing
  * Data Augmentation -> produces variations of existing data for more robust training/testing
    * Random Resize Cropping
    * Transforms, Rotations
    * Blurring, Distortions
  * Train the model to clean data -> helps us to catch problems in the data early
    * Use confusion-matrix to plot top losses
    * ImageClassifierCleaner -> prune high error or low confidence outliers

## Deep Learning Model Deployment Workflows

* Gradio currently preferred 

## Building a Deep Learning Model

1. Gather Data
2. Preprocess Data
3. Train Model
4. Evaluate Model
5. Optimize Model

### **Data Gathering**

The choice of data depends on the type of problem we're trying to solve, and bad data implies a bad model. It is very important that we make assumptions about our data and then test to see if these assumptions are grounded in truth. We must not only aim to have a dataset of adequate size, but also of a high enough quality so our model has the best chance possible to succeed. A good rule of thumb when it comes to dataset size, is that we should ideally have 10x more data than we have model parameters. For Regression problems, 10 examples per predictor variable would be a good place to start from, and for something more complex, like Image Classification: 1000 images per class is reasonable. Besides dataset size, we must ask ourselves some questions about the quality of our data, how common are labelling errors? How noisy are our features? Poor quality will impact reliability.

### **Data Preprocessing**

There are some tried and tested methods to get the most out of our datasets, chiefly the splitting of datasets into subsets. We can then use these subsets for specific steps in our training, evaluation, and optimization loop. The reason we rely on subsets is because the process of developing a model requires tuning of the models hyperparameters, and this tuning is done with feedback from the validation set. We want our training, testing, and validation sets to be very similar to each other to eliminate skewing as much as possible.

* Splitting dataset into `subsets`
  * Train on training set
  * Test on testing set
  * Evaluate on validation set

The splitting or **Partitioning** of our dataset relies on two factors, the total number of samples in our data, and the types of models we are attempting to produce. We have to be careful to **randomize our data** before making this split, or else we might run into a scenario where we've accidentally split say; all passing test scores into the training set, and all failed scores into the test set. The gotcha here is we **DO NOT EVER** want to train on test data, because we will get unrealistically strong opinions about how good our model is. If we have 100% accuracy, or surprisingly low loss, the chances are good a mistake has been made, or data has leaked into our training data.

**How large should our splits be?** This involves two ideas which are in tension, the larger our training set, the better our learning model will be, but the larger our test set is, the higher our confidence will be that we're making accurate predictions. Models with many hyperparameters are easier to validate because of an excess of testable data in the set, but if we have a small dataset we might possibly need to make use of [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). There also exists edge cases where we might not need a validation set because the hyperparameters are difficult to tune. This train-test-validate ratio is dependant on your unique requirements, and you will develop an intuition for the appropriate splits as you build more models.

**Cross-validation** allows us to compare different machine learning methods and get a sense for how well they will work in practice, we achieve this by taking our data set and splitting it into a Training and Testing set, we then set aside the Testing set and use a randomly chosen portion of the Training set for Validation purposes. We then use our Training set to produce multiple splits of the Training and Validation sets. The main advantage of cross-validation, is that it helps us avoid over-fitting. In practice we use what is called K-Fold Cross-Validation to reduce variability in the data. To achieve this we can perform multiple rounds of cross-validation using different partitions and then average the results over all the rounds.

For time-based datasets, where we have data collected over multiple days, it's reasonable to split the data so that we train on the majority of the dataset, and then use for instance the last day out of 30, as our Test/Validate set. This ensures the test on the most recent data, but it is important to remember time-based splits like this work best with **very, very large datasets**.

**Formatting** is another important factor to consider when doing preprocessing. We might have data in a database, and need it as a CSV file, or have a JSON feed to parse and export to a key:value datastore. The requirements will always change, but the important aspect is that this data is rarely ever pristine, we will have missing data.

Dealing with **Missing Data** will take up the largest portion of your time when doing preprocessing. These missing values are typically represented as 'NaN' or 'Null', or are just not there at all, most algorithms can't deal with these indicators, and it requires manual or semi-manual input from us. We can deal with these missing values by eliminating features, at the risk of removing relevant information or we can input missing values by computing averages.

The other problem we might face is having too much data, resulting in using too many resources to perform simple tasks because we have redundant data-points, or where we must rely on **Sampling** in order to reduce the size of a very large dataset while being able to deliver a working model.

**Imbalanced data** is present in almost all real-world datasets. This is when classification data that is skewed which results in majority and minority classes, our model will then be biased to the majority class. To [mitigate these imbalances](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data) we can employ `down-sampling` and `up-weighting`, where we reduce the majority by some factor, which leads to faster convergence and reduced resource use. It is crucial though that the model must remain calibrated, we add up-weighting after down-sampling to keep the dataset in similar ratio.

This leads us to **Feature Scaling**, which relies on [data transformation](https://developers.google.com/machine-learning/data-prep/transform/introduction) techniques to bring [features into similar scale](https://en.wikipedia.org/wiki/Feature_scaling). One of these techniques is **Normalization**, where we rescale features to a range between 0 and 1 by applying [min-max scaling](https://en.wikipedia.org/wiki/Normalization_(statistics)) to each feature column and the other is **Standardization**, which is when we center the field at mean-zero with standard deviation of 1, we do this to prevent features with wider ranges from dominating the distance metric.

### **Model Training & Evaluation**

1. Feed the data to our model
2. Forward propagation occurs
3. Loss is compared against Loss Function
4. Parameters are adjusted based on back propagation
5. Test model against validation set

### **Model Optimization**

a. Hyperparameter Tuning
   * Show the model the entire dataset multiple times (Increased number of epochs)
   * Change how quickly we descend to the lowest global minimum (Adjust the learning rate)
   * Better define the initial conditions in order to reach our desired outcome more quickly

b. Address Overfitting
   * Get more data
     * usually the easier solution
     * helps with reaching generalization
   * Reduce the models size
     * reduce the number of learnable parameters
     * might lead to underfitting
   * Weight Regularization
     * Constrain network complexity by forcing weights to take only small values
     * Add Cost to the Loss Function associated with larger weights
       * L1 Regularization adds cost to the absolute value of the weight coefficient (L1 Norm)
       * L2 Regularization adds cost to the square value of the weights coefficient (L2 Norm)

c. Data Augmentation
   * Good way of increasing dataset points from existing data artificially
   * Flipping, Blurring, Zooming exposes the model to more variation, making it less fragile

d. Dropout
   * Randomly drop some neurons at each forward or backward iteration or pass
   * Neurons develop a co-dependency on each other during training which results in over-fitting

## Additional Information

### **Standard Notation**

As a primer for people like myself who are not as mathematically inclined I present some standard mathematical notation we can be prepared to encounter during our exploration of this topic. It is neither exhaustive or all encompassing, but should help to make the statistics, linear algebra, probability and calculus less opaque.

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

### **Additional Learning Material**

* StatQuest with Josh Starmer: Statistics Fundamentals - [Playlist](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
* MIT OpenCourseWare: Linear Algebra by Gilbert Strang - [Playlist](https://www.youtube.com/playlist?list=PLUl4u3cNGP61iQEFiWLE21EJCxwmWvvek)
* FreeCodeCamp: Linear Algebra Full College Course by Dr. Jim Hefferon - [Playlist](https://www.youtube.com/watch?v=JnTa9XtvmfI)
* Calculus with Professor Leonard - [Playlists](https://www.youtube.com/user/professorleonard57/playlists)
* The Essence of Calculus by Grant Sanderson - [Playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

### **Glossary**

* Glossary of [mathematical symbols](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols)
* List of mathematical symbols [by subject](https://en.wikipedia.org/wiki/List_of_mathematical_symbols_by_subject) 
* List of mathematical [constants](https://en.wikipedia.org/wiki/List_of_mathematical_constants)

### **Toolkits and Libraries**

#### **PyTorch**

What is PyTorch?

PyTorch [documentation](https://pytorch.org/)

Is a popular machine learning toolkit that allows us to write fast deep learning code in Python, capable of running on one or many GPUs. It comes bundled with many prebuilt deep learning models, and is capable of handling the entire data ingest stack: preprocessing our data, modeling our data, deploying our model in an application or on cloud infrastructure. It was originally designed and used by Facebook/Meta, but has since been open-sourced and is used by Tesla, OpenAI and Microsoft.

To get started with PyTorch locally using Pipenv

* Make sure you've got pipenv installed with `pip3 install pipenv`
* Check your version of CUDA using `nvidia-smi`
* Then head to PyTorch's [Start Locally](https://pytorch.org/get-started/locally/) guide, just swap out `pip3` for `pipenv`.
* Install Jupyter inside the venv, `pipenv install jupyter ipython ipykernel pip`
* Once you've installed PyTorch, add some useful data science tools to the venv: `pipenv install pandas numpy matplotlib scikit-learn`
* If you prefer to work in the browser, or run into VSCode flatpak issues: `pipenv run jupyter notebook`

See `exercises_pytorch` for more study notes on pytorch fundamentals, tensors, and workflow examples.

#### **TensorFlow**

What is TensorFlow?

TensorFlow [documentation](https://tensorflow.org/) for more details.

In TensorFlow APIs are arranged hierarchically, with the high-level APIs built on the low-level APIs. Machine Learning Researchers use the low-level APIs to create and explore new machine learning algorithms and the high-lvel API's help streamline workflows for Machine Learning Engineers to help them get jobs done more quickly.

Along with TensorFlow, [NumPy](https://numpy.org/) is popularly used to simplify representing arrays and performing linear algebra operations along with [pandas](https://pandas.pydata.org/), which provides an easy way to represent datasets in memory.

### **Dataset Aggregators**

* UCI [Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
* Kaggle [datasets](https://www.kaggle.com/datasets)
* Google [Dataset Search](https://datasetsearch.research.google.com/)
* Paperswithcode [Datasets](https://paperswithcode.com/datasets)
