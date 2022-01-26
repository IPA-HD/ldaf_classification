import torch
import numpy as np
from stochastic import LDAF_Stochastic
from distributions import UpdatedNormal

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# data transformations
transform_basic = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# load random (but fixed for reproducibility) training set split
ind_val = np.load("validation_ind.npy").tolist()

# load dataset (from the internet if necessary)
dset_path = "../experiments/data/cifar"
cifar_train_full = CIFAR10(root=dset_path, train=True, transform=transform_basic, download=True)
cifar_test = CIFAR10(root=dset_path, train=False, transform=transform_basic)

# split full training set into training set (for prior) 
# and validation set (for certificates)
cifar_val = Subset(cifar_train_full, ind_val)

# data loaders
batch_size = 128
val_dataloader = DataLoader(cifar_val, 100, shuffle=False, num_workers=8)
test_dataloader = DataLoader(cifar_test, 100, shuffle=False, num_workers=8)
eps = 0.01
m = float(len(ind_val))
n = 50
c = 10

# train deterministic classifier by empirical risk minimization
classifier = LDAF_Stochastic.load_from_checkpoint("split_erm_median.ckpt")
classifier.prior = UpdatedNormal(n, c)
classifier.posterior = UpdatedNormal(n, c)
classifier.posterior.init_from(classifier.prior)
classifier.train_posterior()
classifier.lr = 1e-1

for _ in range(5):
    trainer = pl.Trainer(gpus=[0], default_root_dir='alternating_posterior_optimization', max_epochs=5)
    trainer.fit(classifier, val_dataloader, val_dataloader)
    kl = trainer.logged_metrics["train_kl"]
    empirical_risk = trainer.logged_metrics["val_emp_risk_surrogate"]

    with torch.no_grad():
        lambda_opt, bound = classifier.optimal_lambda(empirical_risk, kl, m, eps)
        print("current optimal lambda", lambda_opt)
        print("current optimal bound", bound)
        classifier.temperature = lambda_opt
        classifier.set_temperature(lambda_opt)

trainer.save_checkpoint("optimized_posterior.ckpt")

print("test set")
trainer.test(classifier, test_dataloader)

print("validation set")
trainer.test(classifier, val_dataloader)
