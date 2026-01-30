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

    alpha : float, default = 0.05
        Learning rate
        Determines size of each step in gradient descent 
        ALPHA = initial learning rate
        alpha_t = learning rate at epoch t

    decay : float, default = 0.001
        Learning rate decay
        Determines the rate at which learning rate slows
    
    epochs: int, default = 300
        Specieifes the number of epochs in training for iterative methods
        I.e. how many rounds of gradient descent
        Ignored for "normal" method

    Attributes
    ----------
    METHOD_MAP : dict(str, function)
        Dictionary mapping accepted method strings to method functions
    theta : array size (num_features, 1)
        Array of calculated weights that define the model's fit
    num_features : int
        Stores the number of features the model is fitted to (includes preprocessed bias term) 
        Generally denoted as n
    num_examples : int
        Stores the number of examples the model is fitted on 
        Generally denoted as m
    t : int
        Completed number of epochs
    

    """

    def __init__(self, method = "normal", alpha = 0.05, decay = 0.001, epochs = 300):
        self.method = method

        # Hyperparameters
        self.ALPHA = alpha
        self.alpha_t = alpha
        self.decay = decay
        self.epochs = epochs

        self.theta = None
        self.t = 0
        self.num_features = None
        self.num_examples = None


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


    def inverse_time_decay(self):
        """
        Inverse time decay learning rate schedule.
        alpha_t = alpha / (1 + decay * t)
        Sets alpha_t to the appropriate learning rate

        Args
        ----
        t : int
            Current epoch
        """
        self.alpha_t = self.ALPHA / (1 + self.decay * self.t)


    def preprocess(self, X):
        """
        Adds leading column of 1's to the dataset
        Used to calculate model bias (constant term)

        args
        --------
        X : array (num_samples, num_features)
            Training Data

        Returns
        --------
        Preprocessed data : array (num_samples, num_features + 1)
            Training data with leading bias term initialized to 1
        """
        
        if X.shape[1] == self.num_features:
            return X
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


    def cost(self, X, Y):
        """
        Calculates cost of model on dataset (MSE)

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        
        Returns
        --------
        Cost : float
            MSE of model (current theta) on training data
            MSE = 1/n * sum (y_hat-y)^2
        """

        return (1 / self.num_examples) * np.sum(np.square(self.predict(X) - Y))


    def gradient(self, X, Y):
        """
        Calculates gradient of cost function wrt theta

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        
        Returns
        --------
        gradient : numpy array (num_features, 1)
            Array with each feature's gradient wrt theta

        """
    
        return (2 / self.num_examples) * np.dot(X.T, (self.predict(X) - Y))


    def normal(self, X, Y):
        """
        Calculates theta with Normal Equation: Theta = (X_T * X)^-1 * X_T * Y
        No iterations
        Default method for this model

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        """
        
        self.theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)


    def bgd(self, X, Y):
        """
        Calculates theta with Batch Gradient Descent
        Iterative gradient descent on full dataset
        method = "bgd"

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        """

        # Randomize theta
        self.theta = np.random.randn(self.num_features, 1)

        for t in range(self.epochs):
            # Calculate cost
            cost = self.cost(X, Y)

            # Calculate Gradient
            gradient = self.gradient(X, Y)

            # Apply Gradient Descent 
            self.theta-= (self.alpha_t) * (gradient)

            # Adjust Learning Rate
            self.inverse_time_decay()


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
        # Reconfigure model information
        self.num_features = X.shape[1] + 1
        self.num_examples = X.shape[0]

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
        
        Returns
        --------
        Y_hat : array (num_samples, 1)
            Predictions

        """
        X = self.preprocess(X)

        return np.dot(X, self.theta)

    # Attribute Definitions
    METHOD_MAP = {
        "normal": normal,
        "bgd" : bgd
    }

