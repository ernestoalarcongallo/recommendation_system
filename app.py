import engines.collaborativeFilteringLR as engine
import pandas as pd

ENGINES_PATH = './engines'
DATA_SOURCE_PATH = './dataSource/'
R_FILE = 'R.csv'
THETA_FILE = 'Theta.csv'
X_FILE = 'X.csv'
Y_FILE = 'Y.csv'

def loadData():
  """ Load the data from the data source in csv format
  
  Attributes
  ----------

  X: {array-like}, shape = [n_projects, n_features]
  Theta: {array-like}, shape = [n_employees, n_features]
  Y: {array-like}, shape = [n_projects, n_employees]
  R: {array-like}, shape = [n_projects, n_employees]
    Contains the users rated projects, where R(i, j) = 1 if 
    the project i-th was ratd by the user j-th.

  """
  
  X = pd.read_csv(DATA_SOURCE_PATH + X_FILE, header=None)
  Theta = pd.read_csv(DATA_SOURCE_PATH + THETA_FILE, header=None)
  R = pd.read_csv(DATA_SOURCE_PATH + R_FILE, header=None)
  Y = pd.read_csv(DATA_SOURCE_PATH + Y_FILE, header=None)

  return X.values, Theta.values, R.values, Y.values

def main():
  X, Theta, R, Y = loadData()
  cofi = engine.CollaborativeFilteringLR()
  J, grad = cofi.costFunctionAndGradients(X, Theta, R, Y)
  print('The cost function is:\n{}'.format(J))
  print('The X gradients are:\n{}'.format(grad[0]))
  print('The Theta gradients are:\n{}'.format(grad[1]))
  [X_grad, Theta_grad] = cofi.fit(X, Theta, R, Y)

  J, grad = cofi.costFunctionAndGradients(X_grad, Theta_grad, R, Y)
  print('====================================')
  print('The cost function is:\n{}'.format(J))
  print('The X gradients are:\n{}'.format(grad[0]))
  print('The Theta gradients are:\n{}'.format(grad[1]))

if __name__ == '__main__':
  main()
