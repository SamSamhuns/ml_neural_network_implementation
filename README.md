# Neural Network simple implementation

A simple implementation of a neural network in python. This is not how actual neural networks are implemented and this is an example for only learning purposes.

## Neuron

A neuron is a component of machine learning that takes *n* inputs, does math with the inputs and outputs a single value.

Simple neuron with two inputs.

<img src='img/two_input_neuron.png' />

Inputs x<sub>1</sub> and x<sub>2</sub>, multiplied by weights w<sub>1</sub> and w<sub>2</sub>.

    x1:-> x1 * w1
    x2:-> x2 * w2

Weighted inputs added together with a bias `b`. Then the sum is passed through an activation function, `y=ƒ(x1 * w1 + x2 * w2 + b)`.

The activation function turns an unbounded input into a useful predictable form. It compresses inputs in the domain (-&infin;, 	+&infin;) to (0, 1). Large negative numbers become ~0 while large positive numbers become ~1.

One example of the activation function is the sigmoid function. ƒ(x) = 1&frasl;<sub>(1+e<sup>-x</sup>)</sub>

## 1) An instance of a two input neuron
A two input neuron with weight, *w*=[2,0] and bias *b*=3. And an input of [3,4].

Using the dot product for concise computation.

    (w.x) + b = ((w1 * x1)+(w2 * x2)) + b
              = (6+0)+3
              = 9

            y = ƒ((w.x)+b) = f(9) = 0.999

The **logit** of the neuron is equal to the dot product of the input and weights vector added to the bias.

The process of passing inputs forward to get an output is called a **feedforward**.

```python
import numpy as np
# A simple two input neuron example

def sigmoid(x):
  '''Sigmoid function'''
  return 1/(1+np.exp(-x))

class Neuron:
  '''Simple two input neuron class'''
  def __init__(self, weights, bias):
    self.weights = weights
    self.bias = bias

  def feedforward(inputs):
    '''weights and input array must have same dimensions'''
    si = np.dot(self.weights, inputs) + self.bias
    return sigmoid(si)

weights = np.array([3,4,0,2,4])
bias = -6
inputs = np.array([1,3,2,1,1])

n1 = Neuron(weights, bias)
n1.feedforward(inputs) # 0.999999694097773
```

## 2) A neural network
It a collection of neurons that are connected together. A **Deep Neural Network** is a network with multiple hidden layers.

<img src='img/simple_neural_network.png' />

A **hidden layer** is any layer that is between the input (first) layer and output (last) layer.

```python
# Using the same implementation of Neuron class and sigmoid function from above
class NeuralNetwork:
  def __init__(self, weights, bias):
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, inputs):
    out_h1 = self.h1.feedforward(inputs)
    out_h2 = self.h2.feedforward(inputs)

    out_o1 = self.o1.feedforward(np.array[out_h1, out_h2])
    return out_o1

net1 = NeuralNetwork([0, 1], 0)
inpt = np.array([2, 3])
net1.feedforward(inpt) # 0.7216325609518421
```


## 3) Training a neural network (Mean Squared Error)

Given the following measurements:

| No. | height | weight | age| gender |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 151 | 47 | 63 | m |
| 2 | 139 | 36 | 63 | f |
| 3 | 136 | 31 | 65 | f |
| 4 | 156 | 53 | 41 | m |
| 5 | 145 | 41 | 51 | f |
| 6 | 163 | 62 | 35 | m |

We want our neural network to predict the gender based on the height and weight of the individual first.

Truncating the heights, weights, and ages by their arithmetic means for easier calculation. While males are represented with `1` and females as `0`.

| No. | height (reduce 48) | weight (reduce 45) | age (reduce 53) | gender |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 103 | 2 | 10 | 1 |
| 2 | 91 | -9 | 10 | 0 |
| 3 | 88 | -14 | 12 | 0 |
| 4 | 108 | -8 | -12 | 1 |
| 5 | 97 | -4 | -2 | 0 |
| 6 | 115 | 17 | -18 | 1 |

### Loss/Cost
The quantification of the performance measure of a model based on a function known as the **cost or loss** function.

Using the **Mean Square Error (MSE)** loss function:

<p align="center">
    <img src='/img/mse_loss_function.png' width='400' />
</p>

Here,
* *n* is the total number of samples.
* *y* is the predicted variable that is the gender of the individual.
* *y<sub>true</sub>* is the actual Gender while *y<sub>pred</sub>* is the predicted gender of the individual.
* **Error** = *y<sub>true</sub>* - *y<sub>pred</sub>*

The **MSE** is mean of all the squared errors for each sample. The smaller the loss, the better the ML model.

**Training a neural network** refers to reducing this loss or minimizing the *loss/cost function*.

### Example of loss calculation

Given the following values:

| No. | *y<sub>true</sub>* | *y<sub>pred</sub>* |
| :---: | :---: | :---: |
| 1 | 1 | 1 |
| 2 | 0 | 1 |
| 3 | 0 | 0 |
| 4 | 1 | 1 |
| 5 | 0 | 0 |
| 6 | 1 | 1 |

MSE = <sup>1</sup>&frasl;<sub>6</sub> (0+1+0+0+0+0) = 0.166

### Calculating MSE loss

```python
def mse(y_true, y_pred):
  ''' y_true and y_predict are numpy arrays,
  which represent the true and predicted values
  '''
  return ((y_true - y_pred)**2).mean()

  y_true = np.array([1,0,0,1,0,1])
  y_pred = np.array([1,1,0,1,0,1])
  mse(y_true, y_pred) # 0.166
```

## 4) Training a complete neural network

If we just look at sample 1 from our data.

| No. | height (reduce 48) | weight (reduce 45) | age (reduce 53) | gender |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 103 | 2 | 10 | 1 |

Here, calculating the MSE

*L* = <sup>1</sup>&frasl;<sub>1</sub> ∑<sup><sup>1</sup></sup><sub><sub>i=1</sub></sub> ( *y<sub>true</sub>* - *y<sub>pred</sub>* )<sup>2<sup>

*L* = ( 1 - *y<sub>pred</sub>* )<sup>2<sup>

The loss of a function can also be represented as function of weights and biases of the neurons involved.

<img src='/img/neural_network_weights.png' />

Now, loss can be represented as a multivariate function,

<p align='center'>
    <i>L (w1, w2, w3, w4, w5, w6, b1, b2, b2)</i>
</br></br>
</p>

Now, to minimize this loss function we have to observe how *L* might change when one of its parameters, such as *w1* is changed. For these calculations we can make use of the **partial derivate**, <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>.

We can rewrite this partial derivative in terms of <sup>*∂y<sub>pred</sub>* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> like:

<p align='center'>
    <b><sup><i>∂L</i> </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub> = <sup><i>∂L</i></sup>&frasl;<sub><i>∂y<sub>pred</sub></i> </sub> * <sup><i>∂y<sub>pred</sub></i>  </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub></b>
</br>
</br>
</p>

We calculated that *L* = ( 1 - *y<sub>pred</sub>* )<sup>2</sup>, so:

<p align='center'>
    <b><sup><i>∂L</i> </sup>&frasl;<sub><i>∂y<sub>pred</sub></i> </sub> = -2 ( 1 - <i>y<sub>pred</sub></i> )</b>
</br></br>
</p>

To calculate  <sup>*∂y<sub>pred</sub>*  </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>, given that *h1, h2*, and *o2* represent the outputs of the respective neurons,the final output:

*y<sub>pred</sub>* = *o<sub>1</sub>* = ƒ( *w<sub>7</sub>.h<sub>1</sub>* + *w<sub>8</sub>.h<sub>2</sub>* + *b<sub>3</sub>* )
where ƒ represents the sigmoid function. So:

<p align='center'>
  <sup><i>∂y<sub>pred</sub></i>  </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub> = <sup><i>∂y<sub>pred</sub></i>  </sup>&frasl;<sub> <i>∂h<sub>1</sub></i></sub> * <sup><i>∂h<sub>1</sub></i>  </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub>
  </br>
  <b><sup><i>∂y<sub>pred</sub></i>  </sup>&frasl;<sub> <i>∂h<sub>1</sub></i></sub> = <i>w<sub>7</sub></i> * ƒ<sup>'</sup>( <i>w<sub>7</sub>.h<sub>1</sub></i> + <i>w<sub>8</sub>.h<sub>2</sub></i> + <i>b<sub>3</sub></i> )</b>
  </br></br>
</p>

Doing the same **back propagation** calculation for <sup>*∂h<sub>1</sub>*  </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> :

<p align='center'>
  <i>h<sub>1</sub></i> = ƒ( <i>w<sub>1</sub>.x<sub>1</sub></i> + <i>w<sub>3</sub>.x<sub>2</sub></i> + <i>w<sub>5</sub>.x<sub>3</sub></i> + <i>b<sub>1</sub></i> )
  </br>
  <b><sup><i>∂h<sub>1</sub></i>  </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub> = <i>x<sub>1</sub></i> * ƒ<sup>'</sup>( <i>w<sub>1</sub>.x<sub>1</sub></i> + <i>w<sub>3</sub>.x<sub>2</sub></i> + <i>w<sub>5</sub>.x<sub>3</sub></i> + <i>b<sub>1</sub></i> )</b>
  </br></br>
</p>

Here, *x<sub>1</sub>* is the height, *x<sub>2</sub>* is weight and *x<sub>3</sub>* is the age. *ƒ<sup>'</sup>( x )* is the derivate of the sigmoid function:

<p align='center'>
  <i>ƒ( x )</i> = <sup><i>1</i>  </sup>&frasl;<sub> <i>1 + e<sup>-x</sup></i></sub></br>
  <i>ƒ<sup>'</sup>( x )</i> = - ( 1 + e<sup>-x</sup> )<sup>-2</sup> . ( - e<sup>-x</sup> ) = <sup>1</sup>&frasl;<sub> ( 1 + e<sup>-x</sup> )</sub> * ( 1 - <sup>1  </sup>&frasl;<sub> 1 + e<sup>-x</sup></sub> ) = <b>ƒ( x ) * (1 - ƒ( x ))</b>
  </br></br>
</p>

Finally we can calculate <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> using the following equation:

<b><sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>  = <sup>*∂L* </sup>&frasl;<sub> *∂y<sub>pred</sub>*</sub> \* <sup>*∂y<sub>pred</sub>* </sup>&frasl;<sub> *∂h<sub>1</sub>*</sub> \* <sup>*∂h<sub>1</sub>* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub></b>

### Example calculation of the partial derivative

| No. | height (reduce 48) | weight (reduce 45) | age (reduce 53) | gender |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 103 | 2 | 10 | 1 |

Assuming there is a single row in our dataset and initializing all weights to 1 and all biases to 0:

    h1 = ƒ( w1x1 + w3x2 + w5x3 + b1)
       = f ( 103 + 2 + 10 + 0 )
       = 0.99999999999999

    h2 = ƒ( w2x1 + w4x2 + w6x3 + b2) = 0.99999999999999

    o1 = ƒ( w7h1 + w8h2 + b3)
       = ƒ( 0.99999999999999 + 0.99999999999999 + 0 )
       = 0.88

The neural network predicts that *y<sub>pred</sub>* = 0.88, which is close to the true value 1(male) but not exactly 1.

Now, if we calculate <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> :

<p align='center'>
  <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>  = <sup>*∂L* </sup>&frasl;<sub> *∂y<sub>pred</sub>*</sub> \* <sup>*∂y<sub>pred</sub>* </sup>&frasl;<sub> *∂h<sub>1</sub>*</sub> \* <sup>*∂h<sub>1</sub>* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>
  </br></br>

  <sup>*∂L* </sup>&frasl;<sub> *∂y<sub>pred</sub>*</sub> = -2 ( 1 - *y<sub>pred</sub>* )
  = -2 ( 1 - 0.88 )
  = -0.24
  </br></br>

  <sup>*∂y<sub>pred</sub>* </sup>&frasl;<sub> *∂h<sub>1</sub>*</sub> = *w<sub>7</sub>* \* ƒ<sup>'</sup>( *w<sub>7</sub>.h<sub>1</sub>* + *w<sub>8</sub>.h<sub>2</sub>* + *b<sub>3</sub>* )
  </br>
  1 \* ƒ<sup>'</sup>(0.9999999 + 0.9999999 + 0)
  </br>
  ƒ(1.9999999) \*  (1 - ƒ(1.9999999))
  </br>
  0.105
  </br></br>

  <sup>*∂h<sub>1</sub>* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> =  *x<sub>1</sub>* \* ƒ<sup>'</sup>( *w<sub>1</sub>.x<sub>1</sub>* + *w<sub>3</sub>.x<sub>2</sub>* + *w<sub>5</sub>.x<sub>3</sub>* + *b<sub>1</sub>* )
  </br>
  = 103 \* ƒ<sup>'</sup>(103 + 2 + 10 + 0)
  </br>
  = 103 \* ƒ(115)  \* (1 - ƒ(115))
  </br>
  = 0.0000000000001
  </br></br>

  <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>  = -0.24 \* 0.105 \* 0.0000000000001
  </br>
  **-2.52e-15**

</p>

The negative partial derivative states that increasing *w<sub>1</sub>* would decrease *L* by a tiny fraction.

### Stochastic Gradient Training

The problem of fine tuning the weights and biases so as to minimize the function *L* is an optimization problem. We can use an algorithm called **stochastic gradient descent (SGD)** for this. SGD is just the following update equation:

<p align='center'>
    <i>w<sub>1</sub></i> = <i>w<sub>1</sub></i> - <i>η</i> <sup><i>∂L</i> </sup>&frasl;<sub> <i>∂w<sub>1</sub></i></sub>
</p>

*η* is a constant known as the **learning rate** that controls how fast we train our network. Choosing a big *η* might cause our model to overshoot the minima of the cost function and swing around it forever. Choosing too small of a value for *η* might cause our partial derivative slope to never reach this minima.

- If <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> is negative, <sub> *∂w<sub>1</sub>*</sub> will be increased which will reduce *L*.
- If <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub> is positive, <sub> *∂w<sub>1</sub>*</sub> will be decreased which will reduce *L*.

This process has to be repeated for each weight and bias in our network. This way our loss will slowly decrease and our network will improve.

#### Network Training Process

1. Choose **one** sample from our dataset. As Stochastic Gradient Descent works on one sample at a time.
2. Calculate all the partial derivates of the loss with respect to all the weights and biases. (e.g. <sup>*∂L* </sup>&frasl;<sub> *∂w<sub>1</sub>*</sub>)
3. Use the SGD update equation to update each weight and bias.

## Acknowledgments
*  Based on Victor Zhou's implementation at <https://victorzhou.com/blog/intro-to-neural-networks/>
