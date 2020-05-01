from os import environ, path, makedirs

environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    Dropout,
    Softmax,
    Flatten,
)


class OCRModel:
    def __init__(self, input_shape, output_size):
        self.model = Sequential()
        self.model.add(
            Conv2D(
                48,
                kernel_size=5,
                strides=2,
                padding="valid",
                use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer=Constant(0),
                input_shape=input_shape,
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))
        self.model.add(
            Conv2D(
                64,
                kernel_size=5,
                strides=1,
                padding="valid",
                use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer=Constant(0),
            )
        )
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        # model.add(LeakyReLU())
        # model.add(Dropout(0.5))
        # model.add(
        #    Conv2D(
        #        128,
        #        kernel_size=5,
        #        strides=1,
        #        padding="valid",
        #        use_bias=True,
        #        kernel_initializer="glorot_normal",
        #        bias_initializer=Constant(0),
        #    )
        # )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Flatten())
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(3072))
        self.model.add(LeakyReLU())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_size))
        self.model.add(Softmax())

    def compile(
        self, lr=0.01, decay=0.0001, momentum=0.9, loss="mean_squared_logarithmic_error"
    ):
        sgd = SGD(lr=lr, decay=decay, momentum=momentum)
        self.model.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])

    def fit(self, imgs_array, labels_array, epochs=100):
        self.model.fit(
            imgs_array,
            labels_array,
            epochs=epochs,
            verbose=1,
            use_multiprocessing=True,
        )

    def evaluate(self, imgs_array, labels_array):
        return self.model.evaluate(imgs_array, labels_array)

    def saveModel(self, name):
        script_path = path.dirname(__file__) + "/../outputs/"
        if not path.exists(script_path):
            makedirs(script_path)
        model_json = self.model.to_json()
        with open(script_path + "/" + name + ".json", "w") as modelout:
            modelout.write(model_json)
        self.model.save_weights(script_path + "/" + name + ".h5")

    def loadModel(self, name):
        script_path = path.dirname(__file__) + "/../outputs/"
        with open(script_path + "/" + name + ".json", "r") as modelin:
            model_json = modelin.read()
        self.model = model_from_json(model_json)
        self.model.load_weights(script_path + "/" + name + ".h5")
