from os import path

import tensorflow as tf
import tensorflow_datasets as tf_ds
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.python.client import device_lib
from tensorflow.python.platform import build_info as tf_build_info


# noinspection SpellCheckingInspection
def info():
    print("tf version", tf.version.VERSION)
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
    # noinspection SpellCheckingInspection
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


# 모델에 추가하기 위해 맞춤형 층을 만듭니다.
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(CustomLayer, self).__init__(*args, **kwargs)
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=input_shape[1:],
            dtype=tf.float32,
            initializer=tf.keras.initializers.ones(),
            regularizer=tf.keras.regularizers.l2(0.02),
            trainable=True)

    # call 메서드가 그래프 모드에서 사용되면
    # training 변수는 텐서가 됩니다.
    @tf.function
    def call(self, inputs, training=None):
        if training:
            return inputs + self.w
        else:
            return inputs + self.w * 0.5


def combine_variables_and_layers():
    custom_layer = CustomLayer()
    print(custom_layer([1]).numpy())
    print(custom_layer([1], training=True).numpy())

    train_data = tf.ones(shape=(1, 28, 28, 1))
    test_data = tf.ones(shape=(1, 28, 28, 1))

    # 맞춤형 층을 포함한 모델을 만듭니다.
    # noinspection SpellCheckingInspection
    model = tf.keras.Sequential([
        CustomLayer(input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
    ])

    train_out = model(train_data, training=True)
    test_out = model(test_data, training=False)

    print(train_out)
    print(test_out)


BUFFER_SIZE = 10  # 실전 코드에서는 더 큰 값을 사용합니다.
BATCH_SIZE = 64
NUM_EPOCHS = 5
STEPS_PER_EPOCH = 5


def train():
    data_sets, information = tf_ds.load(name='mnist', with_info=True, as_supervised=True)
    # noinspection SpellCheckingInspection
    mnist_train, mnist_test = data_sets['train'], data_sets['test']

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255

        return image, label

    train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    test_data = mnist_test.map(scale).batch(BATCH_SIZE)

    train_data = train_data.take(STEPS_PER_EPOCH)
    test_data = test_data.take(STEPS_PER_EPOCH)

    image_batch, label_batch = next(iter(train_data))
    print(image_batch)
    print(label_batch)

    image_batch, label_batch = next(iter(test_data))
    print(image_batch)
    print(label_batch)


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
    combine_variables_and_layers()
    train()
    # run()
