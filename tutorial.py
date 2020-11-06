from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()    # interactive mode

# 一共69张图，每张图68个landmark，每个landmark有x、y两个坐标
# landmarks_frame 是 69 rows x 137 columns 的 DataFrame 类型。
landmarks_frame = pd.read_csv("data/faces/face_landmarks.csv")
# print(landmarks_frame)

# 取csv档中的第65行数据看一看
n = 65
image_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]     # Dataframe.iloc 是按行号取某行数据，这里是取第65行的从第2列开始的所有数据。
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print("Image name: {}".format(image_name))
print("Landmarks shape: {}".format(landmarks.shape))
print("First 4 Landmarks: {}".format(landmarks[:4]))


def show_landmarks(image, landmarks):
    """show image with landmarks"""
    plt.imshow(image)    # plt.imshow() 把图像绘制到轴上
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='x', c='r')    # s: The marker size in points**2.
    # plt.pause(0.0001)


plt.figure()
# io.imread() 在 skimage 库下
show_landmarks(io.imread(os.path.join('data/faces/', image_name)), landmarks)

plt.show()


# Dataset class
class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            :param csv_file (string): Path to the csv file with annotation.
            :param root_dir (string): Directory with all the images.
            :param transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 获取图像路径名，从工程目录起始
        image_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        # skimage.io.imread(): 从文件中载入图片，返回ndarray类型，RGB图像返回 MxNx3, RGBA图像返回 MxNx4
        image = io.imread(image_name)
        landmarks = landmarks_frame.iloc[idx, 1:]
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


csv_file = 'data/faces/face_landmarks.csv'
root_dir = 'data/faces'
# 让我们实例化这个类并迭代数据样本。我们将打印前4个样本的size，并显示它们的landmarks。
face_dataset = FaceLandmarksDataset(csv_file=csv_file, root_dir=root_dir)

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]    # 目测这里用到了我们重载过的 FaceLandmarksDataset.__getitem(self, idx)__

    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)    # **把字典类型展开

    if i == 3:
        plt.show()
        break


# Transforms
# 从上面我们可以看到一个问题，就是样本的大小不一样。
# 大多数神经网络期望图像的大小是固定的。
# 因此，我们需要写一些预处理代码。让我们创建三个 transform :
#   Rescale: 缩放图像
#   RandomCorp: 从图像中随机裁剪。这是数据增强。
#   ToTensor: 将 numpy 图像转换为 torch 图像（我们需要交换 axis ）。
# 我们将把它们写成可调用的类，而不是简单的函数，这样变换的参数就不需要
# 在每次调用时传递。为此，我们只需要实现 __call__ 方法，如果需要，
# 还需要实现 __init__ 方法。然后我们可以使用这样的变换:
# tsfm = Transform(params)
# transformed_sample = tsfm(sample)
# 下面观察一下这些变换是如何应用在图像和 landmarks 上的。
class Rescale(object):
    """将样本中的图像重新缩放为给定尺寸。

    Args:
        output_size (tuple or int): 所需的输出大小。如果是元组，输出与output_size匹配。
            如果是int，则设置图片的较短边等于output_size的大小，并保持图像的长宽比不变。
    """

    def __init__(self, output_size):
        # assert 这个关键字称为断言。当这个关键字后面的条件为假时，程序自动崩溃并抛出AssertionError异常
        # 这里是要检查 output_size 是否是 int 或 tuple 两种类型的对象中的一个，如果不是则违反了output_size的规定。
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]   # image 是从 skimage.io.imread() 返回的一个 ndarray
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)   # ? 不懂这里为什么还要转换一下数据类型

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """随机裁剪样本中的图像。

    Args: 所需的输出尺寸。如果是int，则进行正方形裁剪。

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        # np.random.randint() 返回一个随机整形数，范围[0, new_h)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'landmarks': torch.from_numpy(landmarks)}


# Compose transforms
# 现在，我们在样本上应用变换。
# 假设我们想将图像的较短边重新缩放为 256，然后随机裁剪一个尺寸为 224 的正方形。
# 我们要合成 Rescale 和 RandomCrop 变换。
# torchvision.transform.Compose 是一个简单的可调用类，允许我们这样做。
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    tramsformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**tramsformed_sample)

plt.show()


# Iterating through the dataset
# 让我们把这一切放在一起，创建一个具有 transform 的 dataset
# 总结一下，每次对这个数据集进行采样：






