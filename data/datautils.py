import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

import ipdb

ID_to_DIRNAME={
    'I': 'ImageNet',
    'A': 'imagenet-a',
    'K': 'ImageNet-Sketch',
    'R': 'imagenet-r',
    'V': 'imagenetv2-matched-frequency-format-val',
    'flower102': 'Flower102',
    'dtd': 'DTD',
    'pets': 'OxfordPets',
    'cars': 'StanfordCars',
    'ucf101': 'UCF101',
    'caltech101': 'Caltech101',
    'food101': 'Food101',
    'sun397': 'SUN397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

distortions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                'defocus_blur', 'glass_blur',
                'zoom_blur', 'frost',
                'brightness', 'contrast', 'elastic_transform',
                'pixelate','fog','speckle_noise','saturate', 'spatter', 'gaussian_blur']


FER2013_CLASSES = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
# datasets added for AGI-TPT (level tests on N = 7, 10; lambda selection on CIFAR-100)
EXTRA_SETS = ['cifar10', 'cifar100', 'stl10', 'fer2013', 'in21k']


class ToRGB(object):
    """Wrap a transform so grayscale images (FER-2013) enter as RGB."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x.convert('RGB'))


class ImageFolderSubset(datasets.ImageFolder):
    """ImageNet-21k class subset (appendix G.8): an ImageFolder of wnid folders and a
    `classnames_file` with lines 'wnid<TAB>class name'; only listed wnids are kept."""
    def __init__(self, root, classnames_file, transform=None):
        with open(classnames_file) as fp:
            pairs = [l.rstrip('\n').split('\t', 1) for l in fp if l.strip()]
        self.wnid_to_name = dict(pairs)
        super().__init__(root, transform=transform)

    def find_classes(self, directory):
        classes = sorted(w for w in self.wnid_to_name if os.path.isdir(os.path.join(directory, w)))
        return classes, {c: i for i, c in enumerate(classes)}


def build_extra_dataset(set_id, transform, data_root, mode='test', classnames_file=None):
    """Returns (dataset, classnames) for the sets in EXTRA_SETS."""
    train = (mode == 'train')
    if set_id == 'cifar10':
        ds = datasets.CIFAR10(os.path.join(data_root, 'cifar10'), train=train, transform=transform, download=False)
        return ds, ds.classes
    if set_id == 'cifar100':
        ds = datasets.CIFAR100(os.path.join(data_root, 'cifar100'), train=train, transform=transform, download=False)
        return ds, ds.classes
    if set_id == 'stl10':
        ds = datasets.STL10(os.path.join(data_root, 'stl10'), split='train' if train else 'test',
                            transform=transform, download=False)
        return ds, ds.classes
    if set_id == 'fer2013':
        ds = datasets.FER2013(data_root, split='train' if train else 'test', transform=ToRGB(transform))
        return ds, FER2013_CLASSES
    if set_id == 'in21k':
        assert classnames_file is not None, "--classnames_file is required for in21k"
        ds = ImageFolderSubset(os.path.join(data_root, 'imagenet21k'), classnames_file, transform)
        return ds, [ds.wnid_to_name[w] for w in ds.classes]
    raise NotImplementedError(set_id)


def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set (train split only for the equal-strength calibration)
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'train' if mode == 'train' else 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)

    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views



