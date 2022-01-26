'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from resnet import ResnetClassifier

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

# Data
print('==> Preparing data..')
# data transformations
transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
# the following augmentation was used in
# Zagoruyko:2016 and cited by Zhang:2019 as standard
transform_augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# load random (but fixed for reproducibility) training set split
ind_train = np.load("training_ind.npy").tolist()
ind_val = np.load("validation_ind.npy").tolist()

# load dataset (from the internet if necessary)
dset_path = "../experiments/data/cifar"
cifar_train_full = CIFAR10(root=dset_path, train=True, transform=transform_augment, download=True)
cifar_test = CIFAR10(root=dset_path, train=False, transform=transform_basic)

# split full training set into training set (for prior) 
# and validation set (for certificates)
cifar_train = Subset(cifar_train_full, ind_train)
cifar_val = Subset(cifar_train_full, ind_val)

# data loaders
batch_size = 128
train_dataloader = DataLoader(cifar_train, batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(cifar_val, 100, shuffle=False, num_workers=8)
test_dataloader = DataLoader(cifar_test, 100, shuffle=False, num_workers=8)
# load with batchsize 100 in testing to account for torch.mean aggregation of
# accuracy percentages over an epoch (size of test set is divisible by 100)


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

cudnn.benchmark = True

classifier = ResnetClassifier(lr=0.1)
trainer = pl.Trainer(gpus=[0], default_root_dir="./split_train_features", max_epochs=200, callbacks=[ModelCheckpoint(monitor='train_accuracy')])
trainer.fit(classifier, train_dataloader, val_dataloader)
trainer.save_checkpoint("split_resnet_0.ckpt")
trainer.test(classifier, test_dataloader)