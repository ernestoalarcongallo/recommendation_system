import numpy as np

class CollaborativeFilteringLR:
  """ Collaborative Filtering algorithm using Linear Regression """

  def __init__(self, eta=0.05, n_iter=100):
    self.eta = eta
    self.n_iter = n_iter
  
  def costFunctionAndGradients(self, X, Theta, R, Y):
    """ Collaborative filtering (Linear Regression) cost function

      Parameters
      ----------
      X: {array-like}, shape = [n_projects, n_features]
      Theta: {array-like}, shape = [n_employees, n_features]
      Y: {array-like}, shape = [n_projects, n_employees]
      R: {array-like}, shape = [n_projects, n_employees]
        Contains the users rated projects, where R(i, j) = 1 if 
        the project i-th was ratd by the user j-th.

      Returns
      -------
      J: integer
        The cost obtained
      X_grad: {array-like}, shape = [num_projects, num_features]
        Contains the partial derivatives with respect to each element of X
      Theta_grad: {array-like}, shape = [num_employees, num_features]
        Contains the partial derivatives with respect to each element of Theta

    """

    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    prediction = np.dot(X,Theta.transpose())
    J = (1/2) * sum(sum(np.multiply(np.square(np.subtract(prediction, Y)), R)))

    X_grad = np.dot(np.multiply(prediction, R), Theta)
    Theta_grad = np.dot(np.multiply(prediction, R).transpose(), X)

    return J, [X_grad, Theta_grad]
  
  def fit(self):

