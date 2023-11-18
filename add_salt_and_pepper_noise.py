import cv2
import numpy as np
import os
import shutil


def rm_mkdir_my(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)


def add_noise(image, ratio):
    # 获取图像的高度和宽度
    height, width, channels = image.shape

    # 计算要添加的椒盐噪声点数
    num_noise_pixels = int(height * width * ratio / 2)

    # 随机生成噪声点的坐标
    salt_coords = [np.random.randint(0, height, num_noise_pixels), np.random.randint(0, width, num_noise_pixels), np.random.randint(0, channels, 1)]
    pepper_coords = [np.random.randint(0, height, num_noise_pixels), np.random.randint(0, width, num_noise_pixels), np.random.randint(0, channels, 1)]

    # 将椒盐噪声添加到图片上
    image[salt_coords[0], salt_coords[1], salt_coords[2]] = 255  # 设置椒噪声为白色
    image[pepper_coords[0], pepper_coords[1], salt_coords[2]] = 0  # 设置盐噪声为黑色

    return image

if __name__ == "__main__":
    # 获得所有数据集的目录
    dataset_folder = os.path.join(os.path.abspath(os.path.curdir), "pics", "std_dataset")
    dataset_names = os.listdir(dataset_folder)
    # 获得所有噪声数据集的目录
    noisy_dataset_folder = os.path.join(os.path.abspath(os.path.curdir), "pics", "noisy_dataset")

    for dataset_name in dataset_names:
        current_folder = os.path.join(dataset_folder, dataset_name)

        # 定义高斯噪声的均值和标准差
        ratio = 0.1

        # 输出对应数据集加噪声之后保存的文件夹路径
        t_name = dataset_name + "_sp_ratio_" + str(ratio)
        dst_folder = os.path.join(noisy_dataset_folder, t_name)
        rm_mkdir_my(dst_folder)

        image_files = os.listdir(current_folder)
        for image_file in image_files:
            # 读取图片
            image = cv2.imread(os.path.join(current_folder, image_file))
            # 添加噪声
            noisy_image = add_noise(image, ratio)

            # 保存添加噪声后的图片到输出文件夹
            dst_path = os.path.join(dst_folder, image_file)
            cv2.imwrite(dst_path, noisy_image)

        print("Added Salt-pepper noise to {}.".format(dst_folder))
