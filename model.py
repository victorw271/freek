from sklearn.linear_model import LinearRegression
from random import randint
import matplotlib.pyplot as plt


# Generate training set
from random import randint
TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 100

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2*b) + (3*c)
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

# Train model
predictor = LinearRegression(n_jobs=-1)
predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

# Get output
X_TEST = [[10, 20, 30]]
outcome = predictor.predict(X=X_TEST)
coefficients = predictor.coef_

print('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))

plt.figure()
plt.bar(1, outcome)
plt.title("Outcome value")
plt.xlabel("Outcome")
plt.ylabel("Value")
plt.show()

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))
