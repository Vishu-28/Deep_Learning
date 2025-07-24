#simple linear regression
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

X=np.linspace(0, 10, 1000)
y= 3*X + 7 + (3*np.random.randn(1000))

#architecture
model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))

#compile
model.compile(optimizer='sgd',loss='mse',metrics=['mae'])

#train
#model.fit(X, y, epochs=10)
model.fit(X, y, epochs=200,batch_size=32)

#evaluate
model.evaluate(X, y)

#predict
y_pred = model.predict(X)

#plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Keras')
plt.scatter(X, y, color='red', label='original data')
plt.plot(X, y_pred, color='black', label='predicted line')
plt.legend()
plt.savefig('linear_regression.png')
plt.show()

