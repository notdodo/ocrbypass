from os import listdir, environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import numpy as np
from pprint import pprint
from keras.preprocessing.image import load_img, img_to_array
from keras.initializers import Constant
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    LeakyReLU,
    Dropout,
    Softmax,
    Flatten,
)


MAPPING_STRING = {
    "a": 0,
    "b": 1,
    "c": 2,
    "d": 3,
    "e": 4,
    "f": 5,
    "g": 6,
    "h": 7,
    "i": 8,
    "j": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "q": 16,
    "r": 17,
    "s": 18,
    "t": 19,
    "u": 20,
    "v": 21,
    "w": 22,
    "x": 23,
    "y": 24,
    "z": 25,
    "1": 26,
    "2": 27,
    "3": 28,
    "4": 29,
    "5": 30,
    "6": 31,
    "7": 32,
    "8": 33,
    "9": 34,
    "0": 35,
}

model = Sequential()
model.add(
    Conv2D(
        48,
        kernel_size=5,
        strides=2,
        padding="valid",
        use_bias=True,
        kernel_initializer="glorot_normal",
        bias_initializer=Constant(0),
        input_shape=(40, 30, 3),
    )
)
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(
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
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Flatten())
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(3072))
model.add(LeakyReLU())
model.add(Dropout(0.5))
model.add(Dense(36))
model.add(Softmax())

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=0.0001, momentum=0.9)
model.compile(
    optimizer=sgd, loss="mean_squared_logarithmic_error", metrics=["accuracy"]
)


def string_to_array(s):
    string_array = np.zeros(shape=(36))
    i = 0
    for c in s:
        string_array[mapping_string.get(c) + (36 * i)] = 1
        i += 1
    return string_array


def array_to_sring(arr):
    string = ""
    for c in arr:
        i = c % 36
        for key, value in mapping_string.items():
            if i == value:
                string += key
    return string


BASE_PATH = "./captchas/training/"
count_inputs = 0
max_inputs = 1432
imgs_array = np.zeros(shape=(max_inputs, 40, 30, 3))
labels_array = np.zeros(shape=(max_inputs, 36))
for filename in listdir(BASE_PATH):
    img = img_to_array(load_img(BASE_PATH + filename))
    imgs_array[count_inputs] = img
    char = filename[int(filename[6])]
    labels_array[count_inputs] = string_to_array(char)
    count_inputs += 1
    if count_inputs >= max_inputs:
        break

model.fit(imgs_array, labels_array, epochs=20000)

BASE_PATH = "./captchas/test/"
count_inputs = 0
max_inputs = 358
imgs_array = np.zeros(shape=(max_inputs, 40, 30, 3))
labels_array = np.zeros(shape=(max_inputs, 36))
for filename in listdir(BASE_PATH):
    img = img_to_array(load_img(BASE_PATH + filename))
    imgs_array[count_inputs] = img
    char = filename[int(filename[6])]
    labels_array[count_inputs] = string_to_array(char)
    count_inputs += 1
    if count_inputs >= max_inputs:
        break

print(model.evaluate(imgs_array, labels_array))

# serialize model to JSON
model_json = model.to_json()
with open("./model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("ocr_model_20k.h5")


TEST_IMG = "./captchas/test/8nnyy-2.png"
img_predict_array = np.zeros(shape=(1, 40, 30, 3))
img = img_to_array(load_img(TEST_IMG))
img_predict_array[0] = img

out_predict = model.predict(img_predict_array)
out = np.where(out_predict == np.amax(out_predict))[1].tolist()
print(array_to_sring(out))
assert array_to_sring(out) == "n"

TEST_IMG = "./captchas/test/8nnyy-0.png"
img_predict_array = np.zeros(shape=(1, 40, 30, 3))
img = img_to_array(load_img(TEST_IMG))
img_predict_array[0] = img

out_predict = model.predict(img_predict_array)
out = np.where(out_predict == np.amax(out_predict))[1].tolist()
print(array_to_sring(out))
assert array_to_sring(out) == "8"
