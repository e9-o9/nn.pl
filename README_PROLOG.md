# nn.pl - Neural Network Implementation in Pure Prolog

A minimal, pure Prolog implementation of feedforward neural networks with backpropagation training.

## Features

- **Pure Prolog implementation** - No external dependencies, works with standard SWI-Prolog
- **Feedforward neural networks** - Multi-layer perceptron architecture
- **Backpropagation training** - Gradient descent with configurable learning rate
- **Multiple activation functions** - Sigmoid, Tanh, and ReLU
- **Vector/Matrix operations** - Dot products, matrix multiplication, etc.
- **Loss functions** - Mean Squared Error (MSE)
- **Example problems** - XOR and simple regression examples included

## Requirements

- SWI-Prolog 7.0 or higher

## Installation

Simply clone the repository and load the `nn.pl` file:

```bash
git clone https://github.com/e9-o9/nn.pl
cd nn.pl
swipl
```

Then in the Prolog REPL:

```prolog
?- consult('nn.pl').
```

## Quick Start

### Creating a Network

```prolog
% Create a network with 2 inputs, 3 hidden neurons, and 1 output
?- create_network([2, 3, 1], Network, Weights).
```

### Making Predictions

```prolog
% Forward propagation
?- create_network([2, 3, 1], Network, _),
   predict([0.5, 0.8], Network, Output).
```

### Training a Network

```prolog
% Train on XOR problem
?- create_network([2, 4, 1], Network, _),
   TrainingData = [
       sample([0, 0], [0]),
       sample([0, 1], [1]),
       sample([1, 0], [1]),
       sample([1, 1], [0])
   ],
   train(TrainingData, Network, 0.5, 1000, TrainedNetwork),
   predict([1, 1], TrainedNetwork, Output).
```

## API Reference

### Network Creation

#### `create_network(+LayerSizes, -Network, -Weights)`
Creates a neural network with the specified architecture.

- **LayerSizes**: List of integers `[InputSize, Hidden1, Hidden2, ..., OutputSize]`
- **Network**: Returns the network structure
- **Weights**: Returns the initialized random weights

Example:
```prolog
?- create_network([2, 4, 1], Network, Weights).
```

### Forward Propagation

#### `forward(+Input, +Network, -Output)`
Performs forward propagation through the network.

- **Input**: List of input values
- **Network**: The network structure
- **Output**: List of output values

#### `predict(+Input, +Network, -Output)`
Alias for `forward/3`, makes a prediction using the trained network.

### Training

#### `train(+TrainingData, +Network, +LearningRate, +Epochs, -TrainedNetwork)`
Trains the network using backpropagation.

- **TrainingData**: List of `sample(Input, Target)` terms
- **Network**: Initial network
- **LearningRate**: Learning rate (typically 0.01 - 1.0)
- **Epochs**: Number of training epochs
- **TrainedNetwork**: Returns the trained network

Example:
```prolog
?- TrainingData = [sample([0, 0], [0]), sample([0, 1], [1])],
   train(TrainingData, Network, 0.5, 1000, TrainedNetwork).
```

#### `backprop(+Input, +Target, +Network, +LearningRate, -UpdatedNetwork)`
Performs one step of backpropagation for a single training example.

### Activation Functions

#### `sigmoid(+X, -Y)`
Sigmoid activation: `y = 1 / (1 + e^(-x))`

#### `tanh_activation(+X, -Y)`
Hyperbolic tangent activation

#### `relu(+X, -Y)`
ReLU activation: `y = max(0, x)`

### Loss Functions

#### `mse_loss(+Predicted, +Target, -Loss)`
Computes Mean Squared Error loss.

### Vector and Matrix Operations

#### `dot_product(+Vector1, +Vector2, -Result)`
Computes dot product of two vectors.

#### `add_vectors(+Vector1, +Vector2, -Result)`
Element-wise vector addition.

#### `matrix_vector_mult(+Matrix, +Vector, -Result)`
Matrix-vector multiplication.

## Examples

### XOR Problem

The XOR problem is a classic test for neural networks:

```prolog
?- consult('nn.pl').
?- example_xor.

Training XOR network...
Testing predictions:
  [0, 0] -> [0.05123...]
  [0, 1] -> [0.94567...]
  [1, 0] -> [0.95234...]
  [1, 1] -> [0.04891...]
```

### Simple Regression

Approximate a function (e.g., `f(x) = x^2`):

```prolog
?- consult('nn.pl').
?- example_regression.

Training regression network...
Testing predictions:
  f(0.0) = [0.0234...] (expected: 0.0)
  f(1.0) = [0.9876...] (expected: 1.0)
  f(1.5) = [2.1543...] (expected: 2.25)
```

### Custom Problem

```prolog
% Create a network for binary classification
?- create_network([5, 8, 2], Network, _),
   TrainingData = [
       sample([1, 0, 1, 0, 1], [1, 0]),
       sample([0, 1, 0, 1, 0], [0, 1])
   ],
   train(TrainingData, Network, 0.3, 500, TrainedNetwork),
   predict([1, 0, 1, 0, 1], TrainedNetwork, Output).
```

## Running Tests

```bash
swipl -q -l test_nn.pl -g run_tests -t halt
```

Or within the Prolog REPL:

```prolog
?- consult('test_nn.pl').
?- run_tests.
```

## Architecture

The implementation consists of several key components:

1. **Network Structure**: Networks are represented as `network(LayerSizes, Weights)` terms
2. **Weights**: Each layer has a list of neurons, each neuron has `neuron(Weights, Bias)`
3. **Forward Propagation**: Recursive computation through layers with sigmoid activation
4. **Backpropagation**: Gradient computation and weight updates using chain rule
5. **Training**: Iterative epoch-based training with full batch updates

### Data Structures

- **Network**: `network(LayerSizes, Weights)`
- **Neuron**: `neuron(Weights, Bias)`
- **Training Sample**: `sample(Input, Target)`
- **Gradient**: `gradient(WeightGradients, BiasGradient)`

## Limitations

- Currently only supports sigmoid activation in backpropagation (tanh and relu are available but not fully integrated)
- Full batch training only (no mini-batch or stochastic gradient descent)
- No regularization (L1/L2)
- No momentum or adaptive learning rates
- Performance is limited compared to optimized libraries (this is for educational purposes)

## Performance Tips

- Use smaller networks for faster training
- Adjust learning rate based on problem (0.1 - 0.5 works well for most problems)
- More hidden neurons can help with complex problems but slow down training
- Training time grows with: network size, dataset size, and number of epochs

## Educational Purpose

This implementation is designed for:
- Learning how neural networks work
- Understanding backpropagation in detail
- Experimenting with network architectures
- Teaching Prolog programming
- Prototyping simple ML problems

For production use, consider established libraries like TensorFlow, PyTorch, or similar.

## License

See COPYRIGHT.txt in the repository.

## Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [SWI-Prolog Documentation](https://www.swi-prolog.org/pldoc/)
