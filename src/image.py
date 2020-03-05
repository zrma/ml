from os import path

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


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
    for _ in data_generator.flow(x,
                                 batch_size=1,
                                 save_to_dir=target_path,
                                 save_prefix="cat",
                                 save_format="jpeg"):
        i += 1
        if i > 20:
            break  # 이미지 20장을 생성하고 종료