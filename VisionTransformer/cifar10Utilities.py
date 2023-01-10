from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
import config

DATASET_PATH = config.cifar_path

def get_cifar_loaders(return_only_train=True):
    # the values in Normalize transform correspond to the values that scale and shift the data to a zero mean
    # and standard deviation of one
    test_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                              [0.24703223, 0.24348513, 0.26158784])
                                         ])
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                                                               [0.24703223, 0.24348513, 0.26158784])
                                          ])

    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    # Loading the test set
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)

    if return_only_train:
        return train_loader
    return train_loader, val_loader, test_loader


def plot_imgs_from_dataset(dataset):
    NUM_IMAGES = 4
    CIFAR_images = torch.stack([dataset[idx][0] for idx in range(NUM_IMAGES)], dim=0)
    img_grid = torchvision.utils.make_grid(CIFAR_images, nrow=4, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(8, 8))
    plt.title("Image examples of the CIFAR10 dataset")
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()


def image_to_patch(x, patch_size, flatten_images=True):
    '''
    :param x: torch.Tensor representing the image of shape [batch, channel, height, width]
    :param patch_size: number of pixels per dimension of patches (integer). px for patch_size =16 -> patches have 16*16 pixels
    :param flatten_images: if true, the patches will be returned in a flattened format as a feature vector instead of image grid
    :return:
    '''

    b, c, h, w = x.shape
    path_height = h//patch_size
    patch_width = w//patch_size
    # prepei: patch_size * patch_height * patch_width = img_height * img_width
    # px 9* (16*16) = 48*48 image size
    x = x.reshape(b, c, path_height , patch_size,patch_width, patch_size)

    x = x.permute(0,2,4,1,3,5) # [b, h', w', c, p_h, p_w]
    x = x.flatten(1,2) # [b, h'*w', c, p_h, p_w]

    if flatten_images:
        x = x.flatten(2,4) # [b, h'*w', c*p_h*p_w]
    return x