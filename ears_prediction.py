import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

# constants
FOLD_ITERATIONS = 10

# load preprocessed data
X = pickle.load(open('X_ears_eth', 'rb'))
y = pickle.load(open('y_ears_eth', 'rb'))

# normalize data
X = normalize(X)

def build_model():
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(7))
    model.add(Activation('sigmoid'))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

loss_avg = 0.0
acc_avg = 0.0
model_callbacks = [tf.keras.callbacks.EarlyStopping(patience=2)]
for i in range(FOLD_ITERATIONS):
    # clear state
    tf.keras.backend.clear_session()
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # build model using data
    model = build_model()
    # train
    model.fit(X_train, y_train, batch_size=16, epochs=8, validation_split=0.1, callbacks=model_callbacks)
    # evaluate
    val_loss, val_acc = model.evaluate(X_test, y_test)
    # print results for this iteration
    print(f"Iteration: {i}, Val loss: {val_loss}, acc: {val_acc}")
    loss_avg += val_loss
    acc_avg += val_acc

print(f"Average: Loss: {loss_avg / 10.0}, Acc: {acc_avg / 10.0}")
