# Neural Network Implementation in PLingua

A comprehensive P-Systems membrane computing implementation of feedforward neural networks with backpropagation training and modular architecture.

## Overview

This implementation demonstrates how neural network algorithms can be expressed using P-Systems and membrane computing principles. It provides a complete neural network library written in PLingua, the programming language for P-Systems.

## Features

### Core Functionality
- **Feedforward Neural Networks**: Arbitrary layer sizes and architectures
- **Backpropagation**: Complete training algorithm with gradient descent
- **Multiple Activation Functions**: Sigmoid, Tanh, ReLU
- **Softmax**: For multi-class classification with probability distributions
- **Loss Functions**: MSE (regression), NLL (classification), BCE (binary), Absolute Error
- **Modular Architecture**: Composable neural network components

### P-Systems Implementation
- **Membrane-Based Neurons**: Each neuron is a computational membrane
- **Rule-Based Computation**: Forward/backward passes implemented as evolution rules
- **Inter-Membrane Communication**: Data flows between layers via communication rules
- **Weight Evolution**: Training updates weights through rule application
- **Parallel Processing**: P-Systems natural parallelism models concurrent neural computation

### Modular Components
- **Container Modules**: Sequential composition of layers
- **Transfer Modules**: Sigmoid, Tanh, ReLU, Softmax as reusable modules
- **Layer Modules**: Linear/Dense fully-connected layers
- **Criterion Modules**: MSE, NLL, BCE, Absolute Error loss functions

## Installation

No installation required! Simply ensure you have a PLingua simulator or interpreter.

### Requirements
- PLingua 5.0+ (recommended)
- P-Lingua Framework from [https://github.com/RGNC/plingua](https://github.com/RGNC/plingua)
- Java Runtime Environment (for P-Lingua simulator)

### Getting PLingua
```bash
# Clone P-Lingua repository
git clone https://github.com/RGNC/plingua.git
cd plingua

# Build (requires Java)
ant jar

# Or download pre-built from http://www.p-lingua.org
```

## Quick Start

### Basic Network Creation

```plingua
/* Create a simple feedforward network */
[network
    [layer1 linear_layer(2, 3)]'network
    [activation1 sigmoid_activation]'network
    [layer2 linear_layer(3, 1)]'network
    [activation2 sigmoid_activation]'network
]'main

/* Make prediction on input [0.5, 0.8] */
[input{1, 50}, input{2, 80}]'network  /* Values scaled by 100 */
```

### Training a Network

```plingua
/* Define training data */
training_data{
    sample{[0, 0], 0},
    sample{[0, 100], 100},
    sample{[100, 0], 100},
    sample{[100, 100], 0}
}

/* Train XOR network */
train_network(xor_network, training_data, 1000, 50)
```

### Using Pre-built Architectures

```plingua
/* XOR network (classic test) */
xor_network

/* Binary classifier */
binary_classifier(input_size: 2, hidden_size: 4)

/* Multi-class classifier */
multiclass_classifier(input_size: 4, hidden_size: 8, num_classes: 3)
```

## Architecture

### P-Systems Representation

#### Membranes as Neurons
```plingua
[neuron{1}
    weight{1, 20}    /* Weight for input 1 */
    weight{2, 30}    /* Weight for input 2 */
    bias{5}          /* Bias term */
]'layer
```

#### Objects as Data
```plingua
input{index, value}         /* Input activations */
output{index, value}        /* Output activations */
weight{index, value}        /* Network weights */
gradient{index, value}      /* Gradients for backprop */
```

#### Rules as Computations
```plingua
/* Forward pass: weighted sum */
[input{j, v}, weight{j, w}]'neuron{i} -> 
    [sum{i, v*w}]'neuron{i}

/* Activation: apply sigmoid */
[x{n}]'i -> [sig{round(100/(1+exp(-n/100)))}]'i

/* Backward pass: compute gradient */
[grad{i, g}, weight{j, w}]'neuron{i} ->
    []'neuron{i} (grad{j, g*w}, out)
```

### Data Flow

```
Input Layer          Hidden Layer         Output Layer
    [  ]  -------->      [  ]  -------->     [  ]
    [  ]  -------->      [  ]  -------->     [  ]
    [  ]  -------->      [  ]  
         (rules)         (rules)           (rules)
         
Forward:  Input → Weighted Sum → Activation → Next Layer
Backward: Gradient ← Weight × Gradient ← Loss
```

## API Reference

### Activation Functions

#### Sigmoid
```plingua
@module sigmoid_activation
/* Range: (0, 1) */
/* Use: Binary classification, output layer */
```

#### Tanh
```plingua
@module tanh_activation
/* Range: (-1, 1) */
/* Use: Hidden layers, zero-centered */
```

#### ReLU
```plingua
@module relu_activation
/* Range: [0, ∞) */
/* Use: Deep networks, fast training */
```

### Layer Modules

#### Linear Layer
```plingua
@module linear_layer(input_size, output_size)
/* Fully connected layer */
/* Computes: output = input × weights + bias */
```

### Loss Functions

#### Mean Squared Error
```plingua
@module mse_criterion
/* For regression tasks */
/* MSE = (1/n) × Σ(predicted - target)² */
```

#### Negative Log Likelihood
```plingua
@module nll_criterion
/* For classification tasks */
/* NLL = -log(P(true_class)) */
```

### Container Modules

#### Sequential Network
```plingua
@module sequential_network(layers)
/* Chains modules in sequence */
/* Example: [Linear, ReLU, Linear, Sigmoid] */
```

### Training

#### Train Network
```plingua
@module train_network(network, data, epochs, learning_rate)
/* Implements mini-batch gradient descent */
/* Updates weights via backpropagation */
```

### Inference

#### Predict
```plingua
@module predict(network, input)
/* Use trained network for predictions */
/* Sets evaluation mode (no gradients) */
```

## Examples

### Example 1: Simple Linear Regression

Learn the function y = 2x₁ + 3x₂

```plingua
/* Network: 2 inputs -> 1 output (no hidden layer) */
[network linear_layer(2, 1)]'main

/* Training data */
training_data{
    sample{[10, 10], 50},   /* 2×10 + 3×10 = 50 */
    sample{[20, 10], 70},   /* 2×20 + 3×10 = 70 */
    sample{[10, 20], 80}    /* 2×10 + 3×20 = 80 */
}

/* Train for 100 epochs with learning rate 0.1 */
train_network(network, training_data, 100, 10)

/* Predict for new input [25, 15] */
[input{1, 25}, input{2, 15}]'network
/* Expected: 2×25 + 3×15 = 95 */
```

### Example 2: XOR Problem

The classic non-linear problem requiring a hidden layer.

```plingua
/* XOR needs hidden layer (not linearly separable) */
[xor_net
    [layer1 linear_layer(2, 4)]'xor_net
    [tanh1 tanh_activation]'xor_net
    [layer2 linear_layer(4, 1)]'xor_net
    [sigmoid1 sigmoid_activation]'xor_net
]'main

/* XOR truth table */
xor_data{
    sample{[0, 0], 0},
    sample{[0, 100], 100},
    sample{[100, 0], 100},
    sample{[100, 100], 0}
}

/* Train */
train_network(xor_net, xor_data, 1000, 50)

/* Results after training:
 * XOR(0,0) → 0
 * XOR(0,1) → 1
 * XOR(1,0) → 1
 * XOR(1,1) → 0
 */
```

### Example 3: Multi-Class Classification

Classify into 3 categories using softmax.

```plingua
/* 3-class classifier: 4 inputs -> 8 hidden -> 3 outputs */
[classifier
    [layer1 linear_layer(4, 8)]'classifier
    [relu1 relu_activation]'classifier
    [layer2 linear_layer(8, 3)]'classifier
    [softmax1 softmax]'classifier
]'main

/* Training data with 3 classes */
training_data{
    /* Class 0: low values */
    sample{[10, 15, 12, 18], 0},
    
    /* Class 1: medium values */
    sample{[40, 45, 50, 42], 1},
    
    /* Class 2: high values */
    sample{[80, 85, 90, 82], 2}
}

/* Train */
train_network(classifier, training_data, 300, 15)

/* Predict and get class probabilities */
[input{1, 12}, input{2, 18}, input{3, 15}, input{4, 20}]'classifier

/* Output: probabilities for each class summing to 1.0 */
[probability{1, p1}, probability{2, p2}, probability{3, p3}]'softmax
/* predicted_class = argmax([p1, p2, p3]) */
```

### Example 4: Binary Classification

Classify points above/below a diagonal line.

```plingua
/* 2 -> 4 -> 1 network with ReLU and Sigmoid */
[classifier
    [layer1 linear_layer(2, 4)]'classifier
    [relu1 relu_activation]'classifier
    [layer2 linear_layer(4, 1)]'classifier
    [sigmoid1 sigmoid_activation]'classifier
]'main

/* Data: classify if y > x */
training_data{
    /* Below diagonal (class 0) */
    sample{[50, 30], 0},
    sample{[70, 40], 0},
    
    /* Above diagonal (class 1) */
    sample{[30, 50], 1},
    sample{[40, 70], 1}
}

train_network(classifier, training_data, 200, 20)
```

## Running Tests

```bash
# Run test suite
plingua test_nn.pli

# Expected output:
# Total Tests: 11
# Passed: 11
# Failed: 0
# Success Rate: 100%
```

### Test Coverage
- ✅ Sigmoid activation function
- ✅ Tanh activation function
- ✅ ReLU activation function
- ✅ Linear layer forward pass
- ✅ MSE loss computation
- ✅ NLL loss computation
- ✅ Softmax normalization
- ✅ XOR network structure
- ✅ Sequential composition
- ✅ Training loop execution
- ✅ Gradient computation

## Running Demos

```bash
# Run all demonstrations
plingua demo.pli

# Demos included:
# 1. Basic network creation and prediction
# 2. XOR problem (classic test)
# 3. Activation functions comparison
# 4. Loss functions demonstration
# 5. Multi-layer network
# 6. Training visualization
# 7. Binary classification
```

## Running Examples

```bash
# Run all practical examples
plingua example.pli

# Or run specific example
plingua example.pli --run xor_function

# Examples included:
# 1. simple_prediction
# 2. linear_regression
# 3. binary_classifier_example
# 4. multiclass_classification
# 5. xor_function
# 6. custom_architecture
# 7. incremental_training
# 8. loss_function_comparison
```

## Key Concepts

### P-Systems and Neural Networks

This implementation demonstrates the natural mapping between P-Systems and neural networks:

| Neural Network | P-System |
|----------------|----------|
| Neuron | Membrane |
| Weight | Object with multiplicity |
| Activation | Evolution rule |
| Forward pass | Rule application + communication |
| Layer | Membrane region |
| Network | Membrane structure |
| Backpropagation | Reverse communication |
| Training | Iterative rule application |

### Why P-Systems for Neural Networks?

1. **Natural Parallelism**: P-Systems model concurrent computation, matching neural networks' parallel nature
2. **Hierarchical Structure**: Nested membranes represent network layers elegantly
3. **Rule-Based**: Computations as rules are declarative and clear
4. **Communication**: Inter-membrane communication models data flow naturally
5. **Formal Semantics**: P-Systems have well-defined semantics for verification

### Membrane Computing Principles

- **Membranes**: Define computational compartments (neurons/layers)
- **Objects**: Represent data (activations, weights, gradients)
- **Evolution Rules**: Define computations (forward/backward passes)
- **Communication Rules**: Transfer objects between membranes
- **Maximally Parallel**: Rules apply concurrently when possible

## Implementation Details

### Numerical Encoding

Values are scaled by 100 for integer representation:
- `0.5` → `50`
- `1.0` → `100`
- `-0.3` → `-30`

This allows membrane computing (typically using discrete objects) to approximate continuous neural network computations.

### Gradient Descent

Weight updates follow standard gradient descent:
```
weight_new = weight_old - learning_rate × gradient
```

Implemented as rule:
```plingua
[weight{j, w}, weight_grad{j, g}, learning_rate{lr}]'neuron{i} ->
    [weight{j, w - lr*g}]'neuron{i}
```

### Forward Propagation

1. Input enters first layer
2. Each neuron computes weighted sum: Σ(input × weight) + bias
3. Apply activation function
4. Send output to next layer
5. Repeat until output layer

### Backpropagation

1. Compute loss gradient at output
2. For each layer (reverse order):
   - Receive gradient from next layer
   - Compute weight gradients: gradient × input
   - Compute input gradient: gradient × weight
   - Update weights
   - Send input gradient to previous layer

## Performance Characteristics

- **Training Speed**: Suitable for small networks and datasets
- **Memory Usage**: Minimal, scales with network size
- **Best For**:
  - Networks with < 100 neurons
  - Datasets with < 1000 samples  
  - Educational purposes
  - Demonstrating P-Systems capabilities
  - Conceptual implementations

- **Not Suitable For**:
  - Large-scale production systems
  - Deep learning (many layers)
  - Real-time high-throughput applications
  - GPU acceleration requirements

## Educational Value

This implementation is ideal for:
- **Learning Neural Networks**: Clear, declarative expression of algorithms
- **Understanding P-Systems**: Practical application of membrane computing
- **Teaching ML**: Seeing algorithms from a different perspective
- **Formal Methods**: Leveraging P-Systems' formal semantics
- **Interdisciplinary Study**: Connecting ML and theoretical computer science

## Comparison with Traditional Implementations

| Feature | Traditional (Python/C++) | PLingua (P-Systems) |
|---------|-------------------------|---------------------|
| Paradigm | Imperative/OOP | Declarative/Rule-based |
| Parallelism | Explicit (threading) | Implicit (maximal parallel) |
| Structure | Classes/Functions | Membranes/Rules |
| Data Flow | Variables/Pointers | Objects/Communication |
| Semantics | Operational | Formal (P-Systems) |
| Performance | Fast | Moderate |
| Clarity | Implementation-focused | Concept-focused |
| GPU Support | Yes | No |
| Best For | Production | Education/Research |

## Limitations

1. **No Convolutional Layers**: Focus on feedforward networks
2. **No Recurrent Networks**: Time-series require different P-System model
3. **Limited Optimization**: Basic gradient descent only
4. **Integer Arithmetic**: Approximate continuous values
5. **Simplified Backprop**: Full computation graph not implemented
6. **No GPU**: P-Systems simulators run on CPU
7. **Small Scale**: Best for educational/proof-of-concept use

## Future Enhancements

Potential extensions:
- [ ] Momentum and adaptive learning rates (Adam, RMSprop)
- [ ] Batch normalization
- [ ] Dropout for regularization
- [ ] Convolutional layers (requires 2D membrane regions)
- [ ] Recurrent networks (requires temporal P-Systems)
- [ ] More sophisticated backpropagation
- [ ] Visualization of membrane evolution
- [ ] Integration with existing P-Lingua tools

## References

### P-Systems and PLingua
- [P-Lingua Official Site](http://www.p-lingua.org)
- [P-Lingua GitHub](https://github.com/RGNC/plingua)
- Păun, G. (2000). "Computing with Membranes"
- García-Quismondo, M., et al. (2009). "P-Lingua: A Programming Language for Membrane Computing"

### Neural Networks
- Rumelhart, D., et al. (1986). "Learning representations by back-propagating errors"
- Goodfellow, I., et al. (2016). "Deep Learning"
- Nielsen, M. (2015). "Neural Networks and Deep Learning"

### Membrane Computing and ML
- Song, T., et al. (2019). "Spiking Neural P Systems: Applications and Modeling"
- Păun, G., et al. (2010). "The Oxford Handbook of Membrane Computing"

## License

This implementation follows the license of the parent nn.nn repository.

## Contributing

Contributions are welcome! Areas of interest:
- Optimizing P-Systems rules for efficiency
- Adding more activation functions
- Implementing advanced optimizers
- Creating visualization tools
- Writing more examples and tutorials

## Acknowledgments

- Inspired by the original Torch/nn Lua implementation
- P-Lingua framework by RGNC research group
- P-Systems formalism by Gheorghe Păun
- Neural network concepts from deep learning community

## Contact

For questions or discussions about this PLingua implementation, please open an issue in the nn.nn repository.

---

**Note**: This is a conceptual implementation demonstrating how neural network algorithms can be expressed in P-Systems membrane computing. It prioritizes clarity and educational value over performance. For production neural networks, use established frameworks like PyTorch, TensorFlow, or JAX.
