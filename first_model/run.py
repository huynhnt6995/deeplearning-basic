# test function f(x, y, z, t) = 3x + 4y + 5z + 6t


from sklearn.linear_model import LinearRegression 
from random import randint

TRAIN_INPUT = []
TRAIN_OUTPUT = []

for i in range(1, 100):
    x = randint(0, 100)
    y = randint(0, 100)
    z = randint(0, 100)
    t = randint(0, 100)
    _x = [x, y, z, t]
    _y = 3*x + 4*y + 5*z + 6*t

    TRAIN_INPUT.append(_x)
    
    TRAIN_OUTPUT.append(_y)

  
predictor = LinearRegression(n_jobs =-1) 
  
# Fill the Model with the Data
predictor.fit(X = TRAIN_INPUT, y = TRAIN_OUTPUT) 

# Random Test data
X_TEST = [[-1, -2, 3, 4]] 

# Predict the result of X_TEST which holds testing data
outcome = predictor.predict(X = X_TEST) 

# Predict the coefficients
coefficients = predictor.coef_

# Print the result obtained for the test data
print('Output : {}\nCoefficients : {}'.format(outcome, coefficients))

# this will out: result output = 50 and Coefficients function: 3x+4y+5z+6t