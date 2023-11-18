import cv2
import numpy as np
import os
import shutil


def rm_mkdir_my(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def add_noise(image, mean, stddev):
    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, stddev, image.shape)

    # 将高斯噪声添加到图片上
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    noisy_image = noisy_image.astype(np.uint8)

    return noisy_image

if __name__ == "__main__":
    # 获得所有数据集的目录
    dataset_folder = os.path.join(os.path.abspath(os.path.curdir), "pics", "std_dataset")
    dataset_names = os.listdir(dataset_folder)
    # 获得所有噪声数据集的目录
    noisy_dataset_folder = os.path.join(os.path.abspath(os.path.curdir), "pics", "noisy_dataset")

    for dataset_name in dataset_names:
        current_folder = os.path.join(dataset_folder, dataset_name)

        # 定义高斯噪声的均值和标准差
        mean = 0
        stddev = 20

        # 输出对应数据集加噪声之后保存的文件夹路径
        t_name = dataset_name + "_gn_s_" + str(stddev)
        dst_folder = os.path.join(noisy_dataset_folder, t_name)
        rm_mkdir_my(dst_folder)

        image_files = os.listdir(current_folder)
        for image_file in image_files:
            # 读取图片
            image = cv2.imread(os.path.join(current_folder, image_file))
            # 添加噪声
            noisy_image = add_noise(image, mean, stddev)

            # 保存添加噪声后的图片到输出文件夹
            dst_path = os.path.join(dst_folder, image_file)
            cv2.imwrite(dst_path, noisy_image)

        print("Added Gaussian noise to {}.".format(dst_folder))
