from os import path

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.platform import build_info as tf_build_info
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def info():
    print("tf version", tf.__version__)
    print("cuda version", tf_build_info.cuda_version_number)
    print("cudnn version", tf_build_info.cudnn_version_number)
    print(device_lib.list_local_devices())


def hello_world():
    w = tf.Variable(tf.ones(shape=(2, 2)), name="w")
    b = tf.Variable(tf.zeros(shape=2), name="b")

    @tf.function
    def forward(x):
        return w * x + b

    out_a = forward([1, 0])
    print(out_a)


def tf_layers_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.04),
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    train_data = tf.ones(shape=(1, 28, 28, 1))
    test_data = tf.ones(shape=(1, 28, 28, 1))

    train_out = model(train_data, training=True)
    print(train_out)

    test_out = model(test_data, training=False)
    print(test_out)

    # 훈련되는 전체 변수
    len(model.trainable_variables)

    print(model.losses)


def run():
    data_generator = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest")

    dir_path = path.dirname(path.realpath(__file__))
    file_name = "cat.0.jpg"
    file_path = path.join(dir_path, "data", "train", "cats", file_name)
    target_path = path.join(dir_path, "data", "preview", "cats")

    img = load_img(file_path)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
    # 지정된 target_path 폴더에 저장합니다.
    i = 0
    for batch in data_generator.flow(x,
                                     batch_size=1,
                                     save_to_dir=target_path,
                                     save_prefix="cat",
                                     save_format="jpeg"):
        i += 1
        if i > 20:
            break  # 이미지 20장을 생성하고 종료


if __name__ == "__main__":
    info()
    hello_world()
    tf_layers_model()
    # run()
