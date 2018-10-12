from utils.mnist_reader import load_mnist
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
import matplotlib.pyplot as plt
import numpy as np

# Used to load a saved model as json
from keras.models import model_from_json


"""
Use  tensorflow version 1.4.0 and keras version 2.0.8.
"""

"""
60000 datasets for training and 10000 test datasets.
"""
"""
Training Data used to tarin the model
Validation data used to tune the hyperparameters
test data used to test the data after the model has gone through initial vetting by validation set
"""

X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')

'''
This step contains normalization and reshaping of input.
For output, it is important to change number to one-hot vector.
Here we rescaled the image data so that each pixel lies in the interval [0, 1] instead of [0, 255].
It is always a good idea to normalize the input so that each dimension has approximately the same scale.

'''

fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2
                        "Dress",        # index 3
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6
                        "Sneaker",      # index 7
                        "Bag",          # index 8
                        "Ankle boot"]   # index 9


X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

'''
The format should be (batch, channels, height, width).
Each image has 28 x 28 resolution.
As all the images are in grayscale, the number of channels is 1.
If it was a color image, then the number of channels would be 3 (R, G, B).
'''


X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
print (X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

'''
Now, we need to one-hot encode the labels i.e. Y_train and Y_test.
In one-hot encoding an integer is converted to an array which contains
only one 1 and the rest elements are 0.
'''

number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

'''
Three steps to create a CNN
1. Convolution
2. Activation
3. Pooling
Repeat Steps 1,2,3 for adding more hidden layers

4. After that make a fully connected network
This fully connected network gives ability to the CNN
to classify the samples
'''

'''
we are using the Sequential model API to create a simple CNN model
repeating a few layers of a convolution layer followed by a pooling
layer then a dropout layer.
'''

# Create model in Keras
# This model is linear stack of layers

model = Sequential()
# This layer is used as an entry point into a graph.
# So, it is important to define input_shape.

model.add(
    InputLayer(
        input_shape=(1, 28, 28)
    ))

# Normalize the activations of the previous layer at each batch.

model.add(BatchNormalization())

# The output of this conv layer is an activation map.

# A classicial CNN Architecture
# input -> Conv -> ReLu - > Conv -> Relu -> Pool -> ReLU -> Conv -> Relu -> Pool -> FullyConnected

'''
attaching a fully connected layer to the end of the network. This layer basically takes an input volume
(whatever the output is of the conv or ReLU or pool layer preceding it)
and outputs an N dimensional vector where N is the number of classes that the program has to choose from.
'''

'''
A FC layer looks at what high level features most strongly correlate to a particular class and has particular
weights so that when you compute the products between the weights and the previous layer, you get the
correct probabilities for the different classes.
'''
# Next step is to add convolution layer to model.
model.add(
    Conv2D(
        32, (2, 2),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform'
    )
)

# Add max pooling layer for 2D data.

model.add(MaxPool2D(padding='same'))

# Add this same two layers to model.

model.add(
    Conv2D(
        32,
        (2, 2),
        padding='same',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform',
        input_shape=(1, 28, 28)
    )
)
model.add(MaxPool2D(padding='same'))

# It is necessary to flatten input data to a vector.
model.add(Flatten())

# Last step is creation of fully-connected layers.

model.add(
    Dense(
        128,
        activation='relu',
        bias_initializer=Constant(0.01),
        kernel_initializer='random_uniform',
    )
)

# Add output layer, which contains ten numbers.
# Each number represents cloth type.
model.add(Dense(10, activation='softmax'))

# Last step in Keras is to compile model.

"""
We use model.compile() to configure the learning process before training the model.
This is where you define the type of loss function, optimizer and the metrics evaluated
by the model during training and testing.
"""

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print (model.summary())

'''
This is very simply convolutional neural network. Model has only 37,946 params.
Now is time to train this network for image classification.
'''

model.fit(X_train,y_train,epochs=20,batch_size=32,validation_data=(X_test,y_test))
model.evaluate(X_test, y_test)
y_hat = model.predict(X_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
# figure = plt.figure(figsize=(20, 8))
# for i, index in enumerate(np.random.choice(X_test.shape[0], size=15, replace=False)):
#     ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
#     # Display each image
#     ax.imshow(np.squeeze(X_test[index]))
#     predict_index = np.argmax(y_hat[index])
#     true_index = np.argmax(y_test[index])
#     # Set the title for each image
#     ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index],
#                                   fashion_mnist_labels[true_index]),
#                                   color=("green" if predict_index == true_index else "red"))
#     print ("Plotting")

''' Saving the model '''

# serialize model to JSON

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

''' Load a model later '''
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))