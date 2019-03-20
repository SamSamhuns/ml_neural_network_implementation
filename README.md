# Neural Network simple implementation

A simple implementation of a neural network in python. This is not how actual neural networks are implemented and this is an example for only learning purposes.

## Neuron

A neuron is a component of machine learning that takes *n* inputs, does math with the inputs and outputs a single value.

Simple neuron with two inputs.

<img src='img/two_input_neuron.png' />

Inputs x<sub>1</sub> and x<sub>2</sub>, multiplied by weights w<sub>1</sub> and w<sub>2</sub>.

    x1:-> x1 * w1
    x2:-> x2 * w2

Weighted inputs added together with a bias `b`. Then the sum is passed through an activation function, `y=f(x1 * w1 + x2 * w2 + b)`.

The activation function turns an unbounded input into a useful predictable form. It compresses inputs in the domain (-&infin;, 	+&infin;) to (0, 1). Large negative numbers become ~0 while large positive numbers become ~1.

One example of the activation function is the sigmoid function. 1&frasl;<sub>(1+e<sup>-x</sup>)</sub>

## 1) An instance of a two input neuron
A two input neuron with weight, *w*=[2,0] and bias *b*=3. And an input of [3,4].

Using the dot product for concise computation.

    (w.x) + b = ((w1 * x1)+(w2 * x2)) + b
              = (6+0)+3
              = 9

            y = f((w.x)+b) = f(9) = 0.999

The **logit** of the neuron is equal to the dot product of the input and weights vector added to the bias.

The demonstrated process of passing inputs forward to get an output is called a **feedforward**.

## 2) A neural network
It a collection of neurons that are connected together.

<img src='img/simple_neural_network.png' />

A **hidden layer** is any layer that is between the input (first) layer and output (last) layer.

## 3) Training a neural network (Mean Squared Error)

Given the following measurements:

| No. | height | weight | age| male |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 151.765 | 47.82 | 63 | 1 |
| 2 | 139.700 | 36.48 | 63 | 0 |
| 3 | 136.525 | 31.86 | 65 | 0 |
| 4 | 156.845 | 53.04 | 41 | 1 |
| 5 | 145.415 | 41.27 | 51 | 0 |
| 6 | 163.830 | 62.99 | 35 | 1 |

## 4) Training a complete neural network

## Acknowledgments
*  Based on Victor Zhou's implementation at <https://victorzhou.com/blog/intro-to-neural-networks/>
