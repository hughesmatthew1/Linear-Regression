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

    eta : float, default = 0.05
        Learning rate
        Determines size of each step in gradient descent 
        ETA = initial learning rate
        eta = learning rate at epoch t

    decay : float, default = 0.0
        Learning rate decay
        Determines the rate at which learning rate slows
        By default, do not use learning decay

    lmbda : float, default = 0.0
        Regularization coefficient

    alpha : float, default = 1.0
        ElasticNet coefficient
        1.0 = L1 (Lasso) Regularization
        0.0 = L2 (Ridge) Regularization

    mb_prop : float, default = 0.2
        Specifies the proportion of examples to be used in each batch of mbgd
        Default = 0.2 => 20% of examples per batch
    
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

    def __init__(self, method = "normal", eta = 0.05, decay = 0.0, lmbda = 0.0, alpha = 0.0, mb_prop = 0.2, epochs = 300):
        self.method = method

        # Hyperparameters
        self._ETA = eta
        self._eta = eta
        self._decay = decay
        self._lambda = lmbda
        self._alpha = alpha
        self._mb_prop = mb_prop
        self._epochs = epochs

        # Initialization
        self._theta = None
        self._t = 0
        self._num_features = None
        self._num_examples = None


    def __str__(self):
        """
        String function returns textual summary of model
        """
        s_overview = f"Linear Regression Model ({self.method.capitalize()})\nHyperparameters:\n"

        s_theta = f"Theta:\n"
        if self._theta is None:
            s_theta += "    Model unfitted"
        else:
            for i in range(len(self._theta)):
                s_theta += "    theta_{}: {:.2f}\n".format(i, self._theta[i][0])

        return s_overview + s_theta


    def inverse_time_decay(self):
        """
        Inverse time decay learning rate schedule.
        eta_t = eta / (1 + decay * t)
        Sets eta_t to the appropriate learning rate
        With default decay, eta_t remains constant

        Args
        ----
        t : int
            Current epoch
        """

        self._eta = self._ETA / (1 + self._decay * self._t)


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
        
        if X.shape[1] == self._num_features:
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

        return (1 / self._num_examples) * np.sum(np.square(self.predict(X) - Y))


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
    
        return (2 / self._num_examples) * np.dot(X.T, (self.predict(X) - Y))


    def regularization(self):
        """
        Calculates regularization value using elastic net regularization
        
        Returns
        --------
        regularization value : numpy array (num_features, 1)
            Array with each feature's regularization value wrt theta
        """

        l1_reg = (self._alpha) * np.sign(self._theta) # lasso
        l2_reg = (1 - self._alpha) * self._theta # ridge

        return l1_reg + l2_reg


    def descent(self, gradient, regularization):
        """
        Apply descent (update parameters)
        
        Args
        --------
        gradient : numpy array (num_features, 1)
            gradient vector wrt theta
        regularization : numpy array (num_features, 1)
            regularization vector wrt theta
        """

        self._theta -= self._eta * (gradient + self._lambda * regularization)


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
        
        self._theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y)


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
        self._theta = np.random.randn(self._num_features, 1)

        for epoch in range(self._epochs):
            # Calculate cost
            cost = self.cost(X, Y)

            # Calculate Gradient
            gradient = self.gradient(X, Y)

            # Calculate Regularization
            regularization = self.regularization()

            # Apply Gradient Descent 
            self.descent(gradient, regularization)

            # Adjust Learning Rate
            self._t += 1
            self.inverse_time_decay()


    def sgd(self, X, Y):
        """
        Calculates theta with Batch Gradient Descent
        Iterative gradient descent on one example per epoch
        method = "bgd"

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        """
        
        # Randomize theta
        self._theta = np.random.randn(self._num_features, 1)

        for epoch in range(self._epochs):
            # Each epoch randomly order the training data
            ordering = np.random.permutation(self._num_examples)
            for i in ordering:
                # Calculate cost on given example
                cost = self.cost(X[i:i+1], Y[i:i+1])

                # Calculate Gradient on given example
                gradient = self.gradient(X[i:i+1], Y[i:i+1])

                # Calculate Regularization
                regularization = self.regularization()

                # Apply Gradient Descent 
                self.descent(gradient, regularization)

                # Adjust Learning Rate
                self._t += 1
                self.inverse_time_decay()

 
    def mbgd(self, X, Y):
        """
        Calculates theta with Batch Gradient Descent
        Iterative gradient descent on a batch of examples per epoch
        method = "bgd"

        Args
        --------
        X : numpy array (num_samples, num_features)
            Training data
        Y : numpy array (num_samples, 1)
            Target values
        """

        batch_size = int(np.round(self._num_examples * self._mb_prop))

        # Randomize theta
        self._theta = np.random.randn(self._num_features, 1)

        for epoch in range(self._epochs):
            # Each epoch randomly order the training data
            ordering = np.random.permutation(self._num_examples)
            for i in range(0, self._num_examples, batch_size):
                j = ordering[i]
                # Calculate cost on given example
                cost = self.cost(X[j:j+batch_size], Y[j:j+batch_size])

                # Calculate Gradient on given example
                gradient = self.gradient(X[j:j+batch_size], Y[j:j+batch_size])
                
                # Calculate Regularization
                regularization = self.regularization()

                # Apply Gradient Descent 
                self.descent(gradient, regularization)

                # Adjust Learning Rate
                self._t += 1
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
        self._num_features = X.shape[1] + 1
        self._num_examples = X.shape[0]

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

        return np.dot(X, self._theta)

    # Attribute Definitions
    METHOD_MAP = {
        "normal": normal,
        "bgd" : bgd,
        "sgd" : sgd,
        "mbgd": mbgd
    }

