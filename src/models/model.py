from keras import Sequential, Input
from keras.layers import Dense
from keras import backend as bck

class Model:
    def __init__(self):
        pass
    
    @staticmethod
    def build():
        bck.clear_session()
        model = Sequential([
            Input(shape=(4,)),
            Dense(units=32, activation="tanh", kernel_initializer="he_uniform"),
            Dense(units=32, activation="tanh", kernel_initializer="he_uniform"),
            Dense(units=3, activation="softmax", kernel_initializer="he_uniform")
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model
        pass