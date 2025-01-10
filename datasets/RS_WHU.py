import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

num_classes = 2
ST_COLORMAP = [[255, 255, 255], [0, 0, 0]]
ST_CLASSES = ['unchanged', 'changed']

# WHU_CD
MEAN_A = np.array([123.74064, 113.411436, 98.7985])
STD_A = np.array([51.422740, 47.749147, 43.159229])
MEAN_B = np.array([124.034032, 123.901724, 119.110628])
STD_B = np.array([54.117341, 53.457246, 52.934778])

# root = os.path.dirname(os.path.realpath(__file__))
root = r''

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    # IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap


def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def normalize_image(im, time='A'):
    assert time in ['A', 'B']
    if time == 'A':
        im = (im - MEAN_A) / STD_A
    else:
        im = (im - MEAN_B) / STD_B
    return im


def read_RSimages(mode, rescale=False):
    img_A_dir = os.path.join(root, mode, 'im1')
    img_B_dir = os.path.join(root, mode, 'im2')
    label_dir = os.path.join(root, mode, 'label')

    data_list = os.listdir(img_A_dir)
    imgs_list_A, imgs_list_B, labels = [], [], []
    count = 0
    for it in data_list:
        if (it[-4:] == '.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_path = os.path.join(label_dir, it)

            imgs_list_A.append(img_A_path)
            imgs_list_B.append(img_B_path)

            label = io.imread(label_path)
            label[label > 0] = 1
            labels.append(label)
        count += 1
        if not count % 500:
            print('%d/%d images loaded.' % (count, len(data_list)))

    print(labels[0].shape)
    print(str(len(imgs_list_A)) + ' ' + mode + ' images' + ' loaded.')

    return imgs_list_A, imgs_list_B, labels


class Data(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.random_flip = random_flip
        self.imgs_list_A, self.imgs_list_B, self.labels = read_RSimages(mode)

    def get_mask_name(self, idx):
        mask_name = os.path.split(self.imgs_list_A[idx])[-1]
        return mask_name

    def __getitem__(self, idx):
        img_A = io.imread(self.imgs_list_A[idx])
        img_A = normalize_image(img_A, 'A')
        img_B = io.imread(self.imgs_list_B[idx])
        img_B = normalize_image(img_B, 'B')

        label = self.labels[idx]

        if self.random_flip:
            img_A, img_B, label = transform.rand_rot90_flip_MCD(img_A, img_B, label)
        return F.to_tensor(img_A), F.to_tensor(img_B), torch.from_numpy(label)

    def __len__(self):
        return len(self.imgs_list_A)


# only png no label
class Data_test(data.Dataset):
    def __init__(self, test_dir):
        self.imgs_A = []
        self.imgs_B = []
        self.mask_name_list = []
        imgA_dir = os.path.join(test_dir, 'im1')
        imgB_dir = os.path.join(test_dir, 'im2')

        data_list = os.listdir(imgA_dir)
        for it in data_list:
            if (it[-4:] == '.png'):
                img_A_path = os.path.join(imgA_dir, it)
                img_B_path = os.path.join(imgB_dir, it)

                self.imgs_A.append(io.imread(img_A_path))
                self.imgs_B.append(io.imread(img_B_path))
                self.mask_name_list.append(it)

        self.len = len(self.imgs_A)

    def get_mask_name(self, idx):
        return self.mask_name_list[idx]

    def __getitem__(self, idx):
        img_A = self.imgs_A[idx]
        img_B = self.imgs_B[idx]

        img_A = normalize_image(img_A, 'A')
        img_B = normalize_image(img_B, 'B')
        return F.to_tensor(img_A), F.to_tensor(img_B)

    def __len__(self):
        return self.len


if __name__ == '__main__':
    train_set = Data("train", random_flip=True)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
