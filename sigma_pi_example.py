import numpy as np
import itertools

class SigmaPiNeuron:
    """
    A Sigma-Pi neuron that computes weighted products of input pairs.
    """
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        # Weights for individual terms and pairwise products
        self.num_terms = n_inputs + (n_inputs * (n_inputs - 1)) // 2
        self.weights = np.random.randn(self.num_terms)
        self.bias = np.random.randn()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # Generate terms: individual inputs + all pairwise products
        terms = list(x)
        for i, j in itertools.combinations(range(len(x)), 2):
            terms.append(x[i] * x[j])
        # Compute activation (sigma-pi aggregation)
        activation = np.dot(terms, self.weights) + self.bias
        return self.sigmoid(activation)

class SigmaPiNetwork:
    """
    A Sigma-Pi neural network with customizable layers.
    """
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(1, len(layer_sizes)):
            layer = [SigmaPiNeuron(layer_sizes[i-1]) for _ in range(layer_sizes[i])]
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = np.array([neuron.forward(x) for neuron in layer])
        return x


def print_description2():
    description = """
 --- Sigma-Pi Neural Network Description ---

 1. Sigma-Pi Neuron:
    - Each neuron takes 'n_inputs' inputs.
    - It computes a weighted sum of:
        a) Each individual input (sigma part).
        b) All pairwise products of inputs (pi part).
    - Number of terms = n_inputs + (n_inputs choose 2)
      For example, if n_inputs = 2:
        Number of terms = 2 + (2*1)/2 = 3 terms:
          - Input 1
          - Input 2
          - Input 1 * Input 2
    - Each term has a corresponding weight.
    - A bias term is added.
    - The neuron output is the sigmoid activation of the weighted sum.

 2. Sigma-Pi Network:
    - The network consists of multiple layers.
    - Each layer has a specified number of Sigma-Pi neurons.
    - The input to each neuron is the output vector from the previous layer.
    - The forward pass:
        For each layer:
          - Compute outputs of all neurons in the layer.
          - Outputs become inputs to the next layer.
    - The final output is produced by the last layer.

 3. Example:
    - Network architecture: [2, 4, 1]
      - Input layer: 2 inputs
      - Hidden layer: 4 Sigma-Pi neurons
      - Output layer: 1 Sigma-Pi neuron
    - Sample input: [0.5, -0.5]
    - Output: a single value between 0 and 1 (due to sigmoid activation)

 4. Key points:
    - The Sigma-Pi neuron captures multiplicative interactions between inputs.
    - This allows modeling of more complex relationships than standard neurons.
    - Weights and biases are randomly initialized at creation.

 --- End of Description ---
 """
    print(description)




def print_description():
    description = """
--- Sigma-Pi Neural Network Description ---

1. Sigma-Pi Neuron:
   - Each neuron takes 'n_inputs' inputs.
   - It computes a weighted sum of:
       a) Each individual input (sigma part).
       b) All pairwise products of inputs (pi part).
   - Number of terms = n_inputs + (n_inputs choose 2)
     For example, if n_inputs = 4:
       Number of terms = 4 + (4*3)/2 = 10 terms:
         - Input 1
         - Input 2
         - Input 3
         - Input 4
         - Input 1 * Input 2
         - Input 1 * Input 3
         - Input 1 * Input 4
         - Input 2 * Input 3
         - Input 2 * Input 4
         - Input 3 * Input 4
   - Each term has a corresponding weight.
   - A bias term is added.
   - The neuron output is the sigmoid activation of the weighted sum.

2. Sigma-Pi Network:
   - The network consists of multiple layers.
   - Each layer has a specified number of Sigma-Pi neurons.
   - The input to each neuron is the output vector from the previous layer.
   - The forward pass:
       For each layer:
         - Compute outputs of all neurons in the layer.
         - Outputs become inputs to the next layer.
   - The final output is produced by the last layer.

--- End of Description ---
"""
    print(description)


if __name__ == "__main__":
    print_description()
    # Your existing code here...





# Example Usage
if __name__ == "__main__":
    # Define network architecture: [input_size, hidden_size, output_size]
    network = SigmaPiNetwork([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

    # Sample input
    input_data = np.array([0.5, -0.5])
    output = network.forward(input_data)

    print(f"Input: {input_data}")
    print(f"Output: {output}")


