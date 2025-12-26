%% Neural Network Implementation in Pure Prolog
%% A minimal, pure Prolog implementation of feedforward neural networks
%% with backpropagation training.
%%
%% Note: Some utility predicates (derivative functions, scalar_mult_vector)
%% are defined but not exported. They are available for internal use and
%% for users who want to extend the module. The module currently uses
%% sigmoid activation in backpropagation, but tanh and relu are provided
%% for forward propagation and can be integrated if needed.

:- module(nn, [
    % Network creation and structure
    create_network/3,
    init_weights/2,
    
    % Forward propagation
    forward/3,
    sigmoid/2,
    tanh_activation/2,
    relu/2,
    
    % Backpropagation and training
    train/5,
    backprop/5,
    
    % Utilities
    dot_product/3,
    matrix_vector_mult/3,
    add_vectors/3,
    mse_loss/3,
    
    % Prediction
    predict/3
]).

%% ============================================================================
%% Network Structure
%% ============================================================================

%% create_network(+LayerSizes, -Network, -NetworkWeights)
%% Creates a neural network structure with the specified layer sizes.
%% LayerSizes: list of integers [InputSize, Hidden1, Hidden2, ..., OutputSize]
%% Network: structure network(Layers, Weights)
%% NetworkWeights: initial random weights
create_network(LayerSizes, network(LayerSizes, Weights), Weights) :-
    init_weights(LayerSizes, Weights).

%% init_weights(+LayerSizes, -Weights)
%% Initialize random weights for all layers.
%% Weights are represented as a list of layer weight matrices.
init_weights([_], []) :- !.
init_weights([InputSize, OutputSize | Rest], [LayerWeights | RestWeights]) :-
    init_layer_weights(InputSize, OutputSize, LayerWeights),
    init_weights([OutputSize | Rest], RestWeights).

%% init_layer_weights(+InputSize, +OutputSize, -LayerWeights)
%% Initialize weights for a single layer.
%% LayerWeights: list of [Weights, Bias] for each neuron
init_layer_weights(InputSize, OutputSize, LayerWeights) :-
    length(LayerWeights, OutputSize),
    maplist(init_neuron_weights(InputSize), LayerWeights).

%% init_neuron_weights(+InputSize, -NeuronWeights)
%% Initialize weights for a single neuron.
init_neuron_weights(InputSize, neuron(Weights, Bias)) :-
    length(Weights, InputSize),
    maplist(random_weight, Weights),
    random_weight(Bias).

%% random_weight(-Weight)
%% Generate a random weight between -1 and 1.
random_weight(W) :-
    random(R),
    W is (R * 2.0) - 1.0.

%% ============================================================================
%% Activation Functions
%% ============================================================================

%% sigmoid(+X, -Y)
%% Sigmoid activation function: y = 1 / (1 + e^(-x))
sigmoid(X, Y) :-
    Y is 1.0 / (1.0 + exp(-X)).

%% sigmoid_derivative(+Y, -Derivative)
%% Derivative of sigmoid: y * (1 - y)
sigmoid_derivative(Y, D) :-
    D is Y * (1.0 - Y).

%% tanh_activation(+X, -Y)
%% Hyperbolic tangent activation function
tanh_activation(X, Y) :-
    Y is tanh(X).

%% tanh_derivative(+Y, -Derivative)
%% Derivative of tanh: 1 - y^2
tanh_derivative(Y, D) :-
    D is 1.0 - (Y * Y).

%% relu(+X, -Y)
%% ReLU activation function: y = max(0, x)
relu(X, Y) :-
    (X > 0 -> Y = X ; Y = 0).

%% relu_derivative(+X, -Derivative)
%% Derivative of ReLU
relu_derivative(X, D) :-
    (X > 0 -> D = 1 ; D = 0).

%% ============================================================================
%% Vector and Matrix Operations
%% ============================================================================

%% dot_product(+Vector1, +Vector2, -Result)
%% Compute dot product of two vectors
dot_product([], [], 0).
dot_product([X|Xs], [Y|Ys], Result) :-
    dot_product(Xs, Ys, Rest),
    Result is Rest + (X * Y).

%% add_vectors(+Vector1, +Vector2, -Result)
%% Element-wise addition of two vectors
add_vectors([], [], []).
add_vectors([X|Xs], [Y|Ys], [Z|Zs]) :-
    Z is X + Y,
    add_vectors(Xs, Ys, Zs).

%% scalar_mult_vector(+Scalar, +Vector, -Result)
%% Multiply vector by scalar
scalar_mult_vector(_, [], []).
scalar_mult_vector(S, [X|Xs], [Y|Ys]) :-
    Y is S * X,
    scalar_mult_vector(S, Xs, Ys).

%% matrix_vector_mult(+Matrix, +Vector, -Result)
%% Multiply matrix by vector (list of dot products)
matrix_vector_mult([], _, []).
matrix_vector_mult([Row|Rows], Vector, [Result|Results]) :-
    dot_product(Row, Vector, Result),
    matrix_vector_mult(Rows, Vector, Results).

%% ============================================================================
%% Forward Propagation
%% ============================================================================

%% forward(+Input, +Network, -Output)
%% Perform forward propagation through the network
forward(Input, network(_, Weights), Output) :-
    forward_layers(Input, Weights, _, Output).

%% forward_layers(+Input, +Weights, -Activations, -Output)
%% Forward propagation through all layers, keeping track of activations
forward_layers(Input, [], [Input], Input).
forward_layers(Input, [LayerWeights|RestWeights], [Input|RestActivations], Output) :-
    forward_layer(Input, LayerWeights, LayerOutput),
    forward_layers(LayerOutput, RestWeights, RestActivations, Output).

%% forward_layer(+Input, +LayerWeights, -Output)
%% Forward propagation through a single layer
forward_layer(Input, LayerWeights, Output) :-
    maplist(forward_neuron(Input), LayerWeights, Output).

%% forward_neuron(+Input, +NeuronWeights, -Output)
%% Compute output of a single neuron with sigmoid activation
forward_neuron(Input, neuron(Weights, Bias), Output) :-
    dot_product(Weights, Input, WeightedSum),
    PreActivation is WeightedSum + Bias,
    sigmoid(PreActivation, Output).

%% ============================================================================
%% Loss Functions
%% ============================================================================

%% mse_loss(+Predicted, +Target, -Loss)
%% Mean squared error loss
mse_loss(Predicted, Target, Loss) :-
    squared_errors(Predicted, Target, Errors),
    sum_list(Errors, Sum),
    length(Errors, N),
    Loss is Sum / N.

%% squared_errors(+Predicted, +Target, -Errors)
squared_errors([], [], []).
squared_errors([P|Ps], [T|Ts], [E|Es]) :-
    E is (P - T) * (P - T),
    squared_errors(Ps, Ts, Es).

%% ============================================================================
%% Backpropagation
%% ============================================================================

%% backprop(+Input, +Target, +Network, +LearningRate, -UpdatedNetwork)
%% Perform backpropagation and update weights
backprop(Input, Target, network(Layers, Weights), LearningRate, network(Layers, UpdatedWeights)) :-
    % Forward pass with activations
    forward_with_activations(Input, Weights, Activations, Output),
    
    % Calculate output error
    subtract_vectors(Output, Target, OutputError),
    
    % Backward pass
    reverse(Weights, RevWeights),
    reverse(Activations, RevActivations),
    backward_layers(RevWeights, RevActivations, OutputError, [], WeightGradients),
    
    % Update weights
    update_weights(Weights, WeightGradients, LearningRate, UpdatedWeights).

%% forward_with_activations(+Input, +Weights, -Activations, -Output)
forward_with_activations(Input, Weights, Activations, Output) :-
    forward_layers(Input, Weights, Activations, Output).

%% backward_layers(+RevWeights, +RevActivations, +Error, +AccGrad, -Gradients)
backward_layers([], _, _, Gradients, Gradients).
backward_layers([LayerWeights|RestWeights], [CurrentAct, PrevAct|RestActs], Error, AccGrad, Gradients) :-
    % Compute gradients for current layer
    compute_layer_gradients(LayerWeights, PrevAct, CurrentAct, Error, LayerGrad, PropError),
    
    % Recursively process previous layers
    backward_layers(RestWeights, [PrevAct|RestActs], PropError, [LayerGrad|AccGrad], Gradients).
backward_layers([LayerWeights|_], [CurrentAct], Error, AccGrad, [LayerGrad|AccGrad]) :-
    % Base case: first layer (use CurrentAct as input since there's no previous layer)
    compute_layer_gradients_first(LayerWeights, CurrentAct, Error, LayerGrad).

%% compute_layer_gradients(+LayerWeights, +Input, +Output, +Error, -LayerGrad, -PropagatedError)
compute_layer_gradients(LayerWeights, Input, Output, Error, LayerGrad, PropError) :-
    % Compute gradient for each neuron with corresponding output and error
    compute_neuron_gradients(LayerWeights, Input, Output, Error, LayerGrad),
    propagate_error(LayerWeights, Error, PropError).

%% compute_layer_gradients_first(+LayerWeights, +Output, +Error, -LayerGrad)
compute_layer_gradients_first(LayerWeights, Output, Error, LayerGrad) :-
    compute_neuron_gradients(LayerWeights, [], Output, Error, LayerGrad).

%% compute_neuron_gradients(+Neurons, +Input, +Outputs, +Errors, -Gradients)
compute_neuron_gradients([], _, [], [], []).
compute_neuron_gradients([neuron(_Weights, _Bias)|RestNeurons], Input, [O|RestOut], [E|RestErr], [gradient(WeightGrads, BiasGrad)|RestGrad]) :-
    sigmoid_derivative(O, Derivative),
    Delta is E * Derivative,
    (Input = [] -> WeightGrads = [] ; maplist(mult_delta(Delta), Input, WeightGrads)),
    BiasGrad = Delta,
    compute_neuron_gradients(RestNeurons, Input, RestOut, RestErr, RestGrad).

%% mult_delta(+Delta, +Input, -Gradient)
mult_delta(Delta, Input, Grad) :-
    Grad is Delta * Input.

%% propagate_error(+LayerWeights, +Error, -PropagatedError)
propagate_error(LayerWeights, Error, PropError) :-
    extract_weights(LayerWeights, WeightMatrix),
    transpose_multiply(WeightMatrix, Error, PropError).

%% extract_weights(+NeuronWeights, -WeightMatrix)
extract_weights([], []).
extract_weights([neuron(Weights, _)|Rest], [Weights|RestWeights]) :-
    extract_weights(Rest, RestWeights).

%% transpose_multiply(+Matrix, +Vector, -Result)
transpose_multiply(Matrix, Vector, Result) :-
    transpose(Matrix, Transposed),
    matrix_vector_mult(Transposed, Vector, Result).

%% transpose(+Matrix, -Transposed)
transpose([[]|_], []).
transpose(Matrix, [Row|Rows]) :-
    maplist(nth0(0), Matrix, Row),
    maplist(list_tail, Matrix, RestMatrix),
    transpose(RestMatrix, Rows).

list_tail([_|T], T).

%% subtract_vectors(+V1, +V2, -Result)
subtract_vectors([], [], []).
subtract_vectors([X|Xs], [Y|Ys], [Z|Zs]) :-
    Z is X - Y,
    subtract_vectors(Xs, Ys, Zs).

%% update_weights(+Weights, +Gradients, +LearningRate, -UpdatedWeights)
update_weights([], [], _, []).
update_weights([LayerW|RestW], [LayerG|RestG], LR, [UpdatedLayer|UpdatedRest]) :-
    update_layer_weights(LayerW, LayerG, LR, UpdatedLayer),
    update_weights(RestW, RestG, LR, UpdatedRest).

%% update_layer_weights(+LayerWeights, +LayerGradients, +LR, -UpdatedWeights)
update_layer_weights([], [], _, []).
update_layer_weights([neuron(W, B)|Rest], [gradient(GW, GB)|RestG], LR, [neuron(NewW, NewB)|UpdatedRest]) :-
    update_vector(W, GW, LR, NewW),
    NewB is B - (LR * GB),
    update_layer_weights(Rest, RestG, LR, UpdatedRest).

%% update_vector(+Weights, +Gradients, +LR, -UpdatedWeights)
update_vector([], [], _, []).
update_vector([W|Ws], [G|Gs], LR, [NewW|NewWs]) :-
    NewW is W - (LR * G),
    update_vector(Ws, Gs, LR, NewWs).

%% ============================================================================
%% Training
%% ============================================================================

%% train(+TrainingData, +Network, +LearningRate, +Epochs, -TrainedNetwork)
%% Train network on dataset for specified epochs
train(_, Network, _, 0, Network) :- !.
train(TrainingData, Network, LearningRate, Epochs, TrainedNetwork) :-
    Epochs > 0,
    train_epoch(TrainingData, Network, LearningRate, UpdatedNetwork),
    NewEpochs is Epochs - 1,
    train(TrainingData, UpdatedNetwork, LearningRate, NewEpochs, TrainedNetwork).

%% train_epoch(+TrainingData, +Network, +LearningRate, -UpdatedNetwork)
%% Train network on all samples in one epoch
train_epoch([], Network, _, Network).
train_epoch([sample(Input, Target)|Rest], Network, LearningRate, UpdatedNetwork) :-
    backprop(Input, Target, Network, LearningRate, IntermediateNetwork),
    train_epoch(Rest, IntermediateNetwork, LearningRate, UpdatedNetwork).

%% ============================================================================
%% Prediction
%% ============================================================================

%% predict(+Input, +Network, -Output)
%% Make a prediction using the trained network
predict(Input, Network, Output) :-
    forward(Input, Network, Output).

%% ============================================================================
%% Example Usage and Tests
%% ============================================================================

%% Example: XOR problem
example_xor :-
    % Create network: 2 inputs, 2 hidden neurons, 1 output
    create_network([2, 2, 1], Network, _),
    
    % Training data for XOR
    TrainingData = [
        sample([0, 0], [0]),
        sample([0, 1], [1]),
        sample([1, 0], [1]),
        sample([1, 1], [0])
    ],
    
    % Train network
    format('Training XOR network...~n'),
    train(TrainingData, Network, 0.5, 1000, TrainedNetwork),
    
    % Test predictions
    format('Testing predictions:~n'),
    predict([0, 0], TrainedNetwork, Out1),
    format('  [0, 0] -> ~w~n', [Out1]),
    predict([0, 1], TrainedNetwork, Out2),
    format('  [0, 1] -> ~w~n', [Out2]),
    predict([1, 0], TrainedNetwork, Out3),
    format('  [1, 0] -> ~w~n', [Out3]),
    predict([1, 1], TrainedNetwork, Out4),
    format('  [1, 1] -> ~w~n', [Out4]).

%% Example: Simple regression
example_regression :-
    % Create network: 1 input, 3 hidden neurons, 1 output
    create_network([1, 3, 1], Network, _),
    
    % Training data: approximate f(x) = x^2
    TrainingData = [
        sample([0.0], [0.0]),
        sample([0.5], [0.25]),
        sample([1.0], [1.0]),
        sample([1.5], [2.25]),
        sample([2.0], [4.0])
    ],
    
    % Train network
    format('Training regression network...~n'),
    train(TrainingData, Network, 0.1, 2000, TrainedNetwork),
    
    % Test predictions
    format('Testing predictions:~n'),
    predict([0.0], TrainedNetwork, P1),
    format('  f(0.0) = ~w (expected: 0.0)~n', [P1]),
    predict([1.0], TrainedNetwork, P2),
    format('  f(1.0) = ~w (expected: 1.0)~n', [P2]),
    predict([1.5], TrainedNetwork, P3),
    format('  f(1.5) = ~w (expected: 2.25)~n', [P3]).

%% Run all examples
run_examples :-
    format('~n=== Neural Network Examples ===~n~n'),
    example_xor,
    format('~n'),
    example_regression.
