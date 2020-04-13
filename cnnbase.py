
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv("data/sign_mnist_train.csv")
test = pd.read_csv("data/sign_mnist_test.csv")

x_train = train.values[:, 1:]
y_train = train.values[:, 0:1]

x_test = test.values[:, 1:]
y_test = test.values[:, 0:1]

#reshape data to fit CNN model
x_train = x_train.reshape(27455,28,28,1)
x_test = x_test.reshape(7172,28,28,1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Split in train and validate
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(25, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10)

# test
loss, acc = model.evaluate(x_test, y_test)
print("Test accuracy:", acc)
 



#x = train.label.value_counts()
#y = train.label.std()
#df = train.groupby('label').nunique()
#print(y)   
    
#print("Y_train shape",np.shape(y_train))

#print(np.shape(x_train[2]))
#print(y_train[2])
#pixels = x_train[2].reshape((28,28))
#plt.imshow(pixels)
#plt.imsave('C.png', pixels)
#y_pred = model.predict(x_test)
#print(np.shape(y_pred))