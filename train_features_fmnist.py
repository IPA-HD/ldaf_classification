'''Train CIFAR10 with PyTorch.'''
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from resnet import ResnetClassifier

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# data transformations
transform_basic = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])
transform_augment = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

# load random (but fixed for reproducibility) training set split
ind_train = np.load("training_ind_fmnist.npy").tolist()
ind_val = np.load("validation_ind_fmnist.npy").tolist()

# load dataset (from the internet if necessary)
dset_path = "../experiments/data/fashion-mnist"
fmnist_train_full = FashionMNIST(root=dset_path, train=True, transform=transform_augment, download=True)
fmnist_test = FashionMNIST(root=dset_path, train=False, transform=transform_basic)

# split full training set into training set (for prior) 
# and validation set (for certificates)
fmnist_train = Subset(fmnist_train_full, ind_train)
fmnist_val = Subset(fmnist_train_full, ind_val)

# data loaders
batch_size = 128
train_dataloader = DataLoader(fmnist_train, batch_size, shuffle=True, num_workers=8)
val_dataloader = DataLoader(fmnist_val, 100, shuffle=False, num_workers=8)
test_dataloader = DataLoader(fmnist_test, 100, shuffle=False, num_workers=8)
# load with batchsize 100 in testing to account for torch.mean aggregation of
# accuracy percentages over an epoch (size of test set is divisible by 100)

cudnn.benchmark = True

classifier = ResnetClassifier(lr=0.1)
trainer = pl.Trainer(gpus=[0], default_root_dir="./fmnist_features", max_epochs=200, callbacks=[ModelCheckpoint(monitor='train_accuracy')]) 
trainer.fit(classifier, train_dataloader, val_dataloader)
trainer.save_checkpoint("fmnist_resnet_0.ckpt")
trainer.test(classifier, test_dataloader)