import numpy as np
import matplotlib.pyplot as plt


# reshape to match the output format

def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))

def sigmoid_derivative(arr):
  s = sigmoid(arr)
  return s * (1 - s)

class NeuralNet():

  def __init__(self, layer_sizes=[2, 3, 4, 3, 1], lr=0.01):
    self.layer_sizes = layer_sizes # number of neurons per layer. The first value is the input dim
    # our input size is a column vector of length 2. therefore, when we do W1 * input, the columns must equal 2, and so on. 
    self.weights = [
      np.random.randn(self.layer_sizes[i], self.layer_sizes[i-1]) # the key is that the second input (i-1) ensures that the matrix multiplication can happen
      for i in range(1, len(self.layer_sizes))
    ]
    self.biases = [
      np.random.randn(self.layer_sizes[i], 1)
      for i in range(1, len(self.layer_sizes))
    ]
    self.lr = lr
  
  def forward(self, X):
    self.As = []
    self.Zs = []
    self.m = X.shape[1]
    A = X.T # transpose X for matrix ops
    for idx, (W, b) in enumerate(zip(self.weights, self.biases)):
      Z = W @ A + b # (3, 2) * (2, 10) -> (3, 10), then broadcast the bias
      A = sigmoid(Z)
      self.As.append(A)
      self.Zs.append(Z)
    
    return self.As[-1]

  
  def cost(self, y_hat, y):
    losses = - ((y * np.log(y_hat)) + (1-y) * np.log(1-y_hat)) # binary CE loss

    m = y_hat.reshape(-1).shape[0] # just get the number of entries in y_hat

    return 1/m * np.sum(np.sum(losses, axis=1))
  
  def backprop(self, X, Y):
    assert Y.shape[1] == 1, f"Y.shape[1] should be 1 to match dims and make column vector, got {Y.shape[0]}"
    self.y_hat = self.forward(X)

    Y = Y.T
    dC_dA3 = -(Y/self.y_hat - (1-Y)/(1-self.y_hat))
    dA3_dZ3 = sigmoid_derivative(self.Zs[-1])
    
    # element-wise because both are (1, m)
    dC_dZ3 = dC_dA3 * dA3_dZ3 
    
    # NOTE: this only works for sigmoid activation, it's a simplification of the above
    # dC_dZ3 = self.y_hat - Y.T
    
    dZ3_dW3 = self.As[-2].T # (10, 1)
    dC_dW3 = 1/self.m * (dC_dZ3 @ dZ3_dW3)
    dC_dB3 = 1/self.m * np.sum(dC_dZ3, axis=1, keepdims=True)

    self.weight_derivs = [dC_dW3] # stored backwards here
    self.bias_derivs = [dC_dB3]

    delta = dC_dZ3

    # time to calculate dC/dW2, dC/dW1, etc.
    for i in range(len(self.weights) - 2, -1, -1):
      # delta = dC/dZ, so dC/dZ_i = (W_(i+1).T @ dC/dZ_(i+1)) * dA_i/dZ_i
      delta = (self.weights[(i+1)].T @ delta) * sigmoid_derivative(self.Zs[i])
      if i == 0:
        # we calculate using input X rather than a previous activation
        dC_dW = (1/self.m) * (delta @ X)
      else:
        dC_dW = (1/self.m) * (delta @ self.As[i - 1].T)
      
      dC_dB = (1/self.m) * np.sum(delta, axis=1, keepdims=True)
        
      self.weight_derivs.insert(0, dC_dW)
      self.bias_derivs.insert(0, dC_dB)

    # decrement by LR
    for i, (dW, dB) in enumerate(zip(self.weight_derivs, self.bias_derivs)):
      self.weights[i] -= dW * self.lr
      self.biases[i] -= dB * self.lr

def get_soccer_problem(n_samples):
  np.random.seed(42)

  goals_per_game = np.random.uniform(0, 3, n_samples)
  pass_accuracy = np.random.uniform(60, 95, n_samples)

  draft_probability = (
      0.1 +  # base probability
      0.5 * (goals_per_game > 1) +  # bonus for good scoring
      0.3 * (pass_accuracy > 75) +    # bonus for good passing
      0.1 * np.random.normal(0, 1, n_samples)  # random noise
  )

  draft_probability = np.clip(draft_probability, 0, 1)
  drafted = (np.random.uniform(0, 1, n_samples) < draft_probability).astype(int)
  X = np.column_stack([goals_per_game, pass_accuracy])
  y = drafted.reshape(-1, 1) # reshape to (100, 1)

  plt.figure(figsize=(10, 6))
  plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], 
            c='red', marker='o', label='Not Drafted', alpha=0.6)
  plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], 
            c='blue', marker='s', label='Drafted', alpha=0.6)
  plt.xlabel('Goals per Game')
  plt.ylabel('Pass Accuracy (%)')
  plt.title('Soccer Player Draft Status')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.savefig("graph.jpg")

  return X, y

def get_separable_data(n_samples):
  # Simple linearly separable data

  x1 = np.random.uniform(-2, 2, n_samples)
  x2 = np.random.uniform(-2, 2, n_samples)

  noise = 0.1 * np.random.normal(0, 1, n_samples)
  decision_value = x1 + x2 + noise

  # Create binary labels
  drafted = (decision_value > 0).astype(int)

  X = np.column_stack([x1, x2])
  y = drafted.reshape(-1, 1)

  plt.figure(figsize=(10, 6))
  plt.scatter(X[y.flatten()==0, 0], X[y.flatten()==0, 1], 
            c='red', marker='o', label='Not selected', alpha=0.6)
  plt.scatter(X[y.flatten()==1, 0], X[y.flatten()==1, 1], 
            c='blue', marker='s', label='Selected', alpha=0.6)

  # Add the decision boundary line
  x_line = np.linspace(-2, 2, 100)
  y_line = -x_line  # x1 + x2 = 0 -> x2 = -x1
  plt.plot(x_line, y_line, 'k--', alpha=0.5, label='Decision Boundary')

  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('Simple Linearly Separable Data')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.xlim(-2.5, 2.5)
  plt.ylim(-2.5, 2.5)
  plt.savefig("graph.jpg")

  return X, y

def get_accuracy(y_hat, y):
  y_pred = (y_hat >= 0.5).astype(int)
  accuracy = np.mean(y_pred == y)
  return accuracy

if __name__ == "__main__":
  np.random.seed(42)

  nn = NeuralNet(lr=0.001)

  max_iterations = 1000

  X, y = get_separable_data(n_samples=10)

  for _ in range(max_iterations):
    nn.backprop(X, y) 
    # evaluate the cost
    out = nn.forward(X)
    cost = nn.cost(nn.y_hat, y.T)
    accuracy = get_accuracy(out, y.T)
    print("cost", cost, "accuracy", accuracy)