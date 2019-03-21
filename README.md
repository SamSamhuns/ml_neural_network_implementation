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
<center><img src='/img/mse_loss_function.png' width='400' /></center>

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

MSE = <sup>1</sup>&frasl;<sub>6</sub> (0+1+0+0+0+0) = **0.166**

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

## Acknowledgments
*  Based on Victor Zhou's implementation at <https://victorzhou.com/blog/intro-to-neural-networks/>
