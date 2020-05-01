from tensorflow.keras.preprocessing.image import load_img, img_to_array
from os import path, listdir
from numpy import zeros
from ocr_libs.mapping import string_to_array


class ImageSet:
    def __init__(self, training, test, input_size, output_size):
        self.__TRAINING_PATH = path.abspath(training)
        self.__TEST_PATH = path.abspath(test)
        self.__count_inputs = 0
        self.__num_trainings = len([name for name in listdir(self.__TRAINING_PATH)])
        self.__num_tests = len([name for name in listdir(self.__TEST_PATH)])
        self.__training_shape = (self.__num_trainings,) + input_size
        self.__test_shape = (self.__num_tests,) + input_size
        self.__training_imgs_array = zeros(shape=self.__training_shape)
        self.__test_imgs_array = zeros(shape=self.__test_shape)
        self.__training_labels_array = zeros(shape=(self.__num_trainings, output_size))
        self.__test_labels_array = zeros(shape=(self.__num_tests, output_size))

    def load_training(self):
        for filename in listdir(self.__TRAINING_PATH):
            self.__training_imgs_array[self.__count_inputs] = img_to_array(
                load_img(self.__TRAINING_PATH + "/" + filename)
            )
            char = filename[int(filename[6])]
            self.__training_labels_array[self.__count_inputs] = string_to_array(char)
            self.__count_inputs += 1
            if self.__count_inputs >= self.__num_trainings:
                break
        self.__count_inputs = 0
        return (self.__training_imgs_array, self.__training_labels_array)

    def load_test(self):
        for filename in listdir(self.__TEST_PATH):
            self.__test_imgs_array[self.__count_inputs] = img_to_array(
                load_img(self.__TEST_PATH + "/" + filename)
            )
            char = filename[int(filename[6])]
            self.__test_labels_array[self.__count_inputs] = string_to_array(char)
            self.__count_inputs += 1
            if self.__count_inputs >= self.__num_tests:
                break
        self.__count_inputs = 0
        return (self.__test_imgs_array, self.__test_labels_array)
