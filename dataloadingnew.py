import tensorflow as tf
import albumentations as A
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        #A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        #A.RandomCrop(height=320, width=320, always_apply=True),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        #A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #A.PadIfNeeded(384, 480)
        A.HorizontalFlip(p=0.5)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['paved-area', 'dirt', 'grass', 'gravel',
               'water', 'rocks', 'pool', 'vegetation', 'roof', 'wall',
               'window', 'door', 'fence', 'fence-pole', 'person', 'dog', 'car',
               'bicycle', 'tree', 'bald-tree', 'ar-marker', 'obstacle']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.imageids = os.listdir(images_dir)
        self.maskids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.imageids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.maskids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 1)
        masks = np.zeros_like(mask)
        # extract certain classes from mask (e.g. cars)
        class_colors = [(0, 0, 0), (128, 64, 128), (130, 76, 0), (0, 102, 0), (112, 103, 87),
                        (28, 42, 168), (48, 41, 30), (0, 50, 89), (107, 142, 35),
                        (70, 70, 70), (102, 102, 156), (254, 228, 12), (254, 148, 12),
                        (190, 153, 153), (153, 153, 153), (255, 22, 96),
                        (102, 51, 0), (9, 143, 150), (119, 11, 32), (51, 51, 0),
                        (190, 250, 190), (112, 150, 146), (2, 135, 115), (255, 0, 0)]
        for c in range(len(self.class_values)):
            masks[:, :, 0] += ((mask[:, :, 0] == c)
                                 * (class_colors[c][0])).astype('uint8')
            masks[:, :, 1] += ((mask[:, :, 0] == c)
                                 * (class_colors[c][1])).astype('uint8')
            masks[:, :, 2] += ((mask[:, :, 0] == c)
                                 * (class_colors[c][2])).astype('uint8')

        # add background if mask is not binary
        """
        if masks.shape[-1] != 1:
            background = 1 - masks.sum(axis=-1, keespdims=True)
            masks = np.concatenate((masks, background), axis=-1)"""

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=masks)
            image, masks = sample['image'], sample['mask']

        return image, masks

    def __len__(self):
        return len(self.imageids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)