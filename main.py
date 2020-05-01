from ocr_libs.model import OCRModel
from image_lib.image import ImageSet
import sys

if __name__ == "__main__":
    from numpy import arange

    for l in arange(0.65, 0.9, 0.08):
        for d in arange(0.0051, 0.009, 0.005):
            for m in arange(0.6, 0.9, 0.1):
                ocr = OCRModel((40, 30, 3), 36)
                ocr.compile(
                    lr=l, decay=d, momentum=m, loss="mean_squared_logarithmic_error",
                )
                img = ImageSet(
                    "./captchas/training/", "./captchas/test/", (40, 30, 3), 36
                )
                imgs_training, labels_training = img.load_training()
                imgs_test, labels_test = img.load_test()

                ocr.fit(imgs_training, labels_training, epochs=10000)
                loss, accuracy = ocr.evaluate(imgs_test, labels_test)
                print(accuracy)
                ocr.saveModel(
                    "ocr_model"
                    + "_"
                    + str(l)
                    + "_"
                    + str(d)
                    + "_"
                    + str(m)
                    + "_"
                    + str(accuracy)
                )

    # TEST_IMG = "./captchas/test/8nnyy-2.png"
    # img_predict_array = np.zeros(shape=(1, 40, 30, 3))
    # img = img_to_array(load_img(TEST_IMG))
    # img_predict_array[0] = img
    #
    # out_predict = model.predict(img_predict_array)
    # out = np.where(out_predict == np.amax(out_predict))[1].tolist()
    # print(array_to_sring(out))
    # assert array_to_sring(out) == "n"
    #
    # TEST_IMG = "./captchas/test/8nnyy-0.png"
    # img_predict_array = np.zeros(shape=(1, 40, 30, 3))
    # img = img_to_array(load_img(TEST_IMG))
    # img_predict_array[0] = img
    #
    # out_predict = model.predict(img_predict_array)
    # out = np.where(out_predict == np.amax(out_predict))[1].tolist()
    # print(array_to_sring(out))
    # assert array_to_sring(out) == "8"
