import torch

import numpy as np
import numbers
import random
from torchvision.transforms import functional as F
from PIL import Image
import pdb
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, labels):
        for t in self.transforms:
            image, labels = t(image, labels)
        return image, labels

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Resize(object):
    def __init__(self, min_size=None, max_size=None, enforced_size=None):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.enforced_size = enforced_size
    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, images, labels):
        '''
        images: a list of PIL Image object 

        '''
        if self.enforced_size is None:
            size = self.get_size(images[0].size)
            for i, img in enumerate(images):
                images[i] = F.resize(img, size)
        else:
            for i, img in enumerate(images):
                images[i] = img.resize(self.enforced_size)

        return images, labels

    def __str__(self):
        return 'Resize(): Min {} | Max {}'.format(self.min_size, self.max_size)

class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, labels):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        if self.mean is not None and self.std is not None:
            image = F.normalize(image, mean=self.mean, std=self.std)
        else:
            image = image*2 - 1
        return image, labels

class ToTensor(object):
    def __call__(self, images, labels):
        '''
        images: list(PIL.Image)
        '''
        for i, img in enumerate(images):
            images[i] = F.to_tensor(img)
        if labels is not None:
            labels = torch.from_numpy(labels)
        return torch.stack(images, dim=1), labels

class Crop(object):
    def __call__(self, images, labels):
        '''
        images: list(PIL.Image)
        '''
        for i, img in enumerate(images):
            images[i] = F.to_tensor(img)
        if labels is not None:
            labels = torch.from_numpy(labels)
        return torch.stack(images, dim=1), labels

class RandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, labels):
        """
        Args:
            images (seq Images): list(PIL.Image)
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # # 8 - leave_to_right, 9 - leave_to_left
            # for t in range(labels.shape[1]):
            #     if np.argmax(labels[:, t]) == 8:
            #         labels[8, t] = 0
            #         labels[9, t] = 1
            #     elif np.argmax(labels[:, t]) == 9:
            #         labels[9, t] = 0
            #         labels[8, t] = 1
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images], labels
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
