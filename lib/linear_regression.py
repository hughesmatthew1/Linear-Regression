import numpy as np

# Class for linear regression model
class LinearRegression:
    """
    Ordinary Least Squares Linear Regression Model
    Fit a linear model with coefficients theta = (theta_1, ..., theta_p)
    which minimize RSS between observed targets (y) and predicted targets (y_hat)

    Parameters
    ----------

    method : string, default = "normal"
        Specifies the solver method for the model
        Options:
            "normal" - Normal Equation
            "bgd" - Batch Gradient Descent
            "sgd" - Stochastic Gradient Descent
            "mbgd" - Mini-Batch Gradient Descent

    alpha : float, default = 0.2
        Learning rate
        Determines size of each step in gradient descent   
    

    Attributes
    ----------
    METHOD_MAP : dict(str, function)
        Dictionary mapping accepted method strings to method functions
    theta : array size (num_features, 1)
        Array of calculated weights that define the model's fit
    

    """

    def __init__(self, method = "normal"):
        self.method = method
        self.theta = None

    def __str__(self):
        """
        String function returns textual summary of model
        """
        s_overview = f"Linear Regression Model ({self.method.capitalize()})\nHyperparameters:\n"

        s_theta = f"Theta:\n"
        if self.theta is None:
            s_theta += "    Model unfitted"
        else:
            for i in range(len(self.theta)):
                s_theta += "    theta_{}: {:.2f}\n".format(i, self.theta[i][0])

        return s_overview + s_theta


    def preprocess(self, X):
        """
        Adds leading column of 1's to the dataset
        Used to calculate model bias (constant term)

        args
        --------
        X : array (num_samples, num_features)
            Training Data

        returns
        --------
        Preprocessed data : array (num_samples, num_features + 1)
            Training data with leading bias term initialized to 1
        """

        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


    def normal(self, X, Y):
        """
        Calculates weights with Normal Equation: Theta = (X_T * X)^-1 * X_T * Y
        No iterations
        Default method for this model
        """
        self.theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)

    def fit(self, X, Y):
        """
        Fit linear model to training data X and training targets Y
        according to model's hyperparameters defined at __init__
        
        Args
        ----------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values

        """
        X = self.preprocess(X)
        self.METHOD_MAP[self.method](self, X, Y)


    def predict(self, X):
        """
        Predict Y_hat from input vector X
        Return model's predictions based on the precomputed value(s) of theta
        
        Args
        ----------
        X : numpy array (num_samples, num_features)
            Data to predict values for 
        
        returns
        --------
        Y_hat : array (num_samples, 1)
            Predictions

        """
        X = self.preprocess(X)
        return np.dot(X, self.theta)

    # Attribute Definitions
    METHOD_MAP = {
        "normal": normal
    }
