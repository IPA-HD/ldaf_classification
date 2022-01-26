"""
Compare accuracy of QMC vs MC
"""
import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
from stochastic import LDAF_Stochastic
from tqdm import tqdm

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def load_single_batch(batch_size=1, shuffle=False):
    # data transformations
    transform_basic = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # load dataset (from the internet if necessary)
    dset_path = "../experiments/data/cifar"
    cifar_test = CIFAR10(root=dset_path, train=False, transform=transform_basic)
    test_dataloader = DataLoader(cifar_test, batch_size, shuffle=shuffle, num_workers=1)

    # load single batch for testing
    batch = next(iter(test_dataloader))
    return batch

def draw_mc_samples(n_samples, safety_offset=1e-6):
    samples = (1-2*safety_offset)*torch.rand(n_samples, 9)+safety_offset
    return samples

def sample_empirical_risk(classifier, n_samples, img, labels):
    # monte-carlo
    classifier.n_sobol_samples = n_samples
    classifier.sobol_samples = draw_mc_samples(n_samples)
    mc_loss, _ = classifier.empirical_risk(img, labels, classifier.posterior)

    # quasi monte-carlo
    soboleng = torch.quasirandom.SobolEngine(dimension=9, scramble=False)
    samples = soboleng.draw(n_samples)
    shift = (1-samples.max())/2
    classifier.sobol_samples = samples+shift
    qmc_loss, _ = classifier.empirical_risk(img, labels, classifier.posterior)

    return mc_loss.cpu().numpy(), qmc_loss.cpu().numpy()

def brute_force_mc(classifier, img, labels):
    mc_results = torch.zeros(img.shape[0], device=classifier.laf.features.linear.weight.device)
    n_samples = int(1e6)
    n_sample_batches = 100
    classifier.n_sobol_samples = n_samples
    
    print("Computing reference solution (n_samples = 100M)")
    for bi in tqdm(range(img.shape[0])):
        for _ in range(n_sample_batches):
            classifier.sobol_samples = draw_mc_samples(n_samples)
            loss, _ = classifier.empirical_risk(img[bi:bi+1,...], labels[bi:bi+1], classifier.posterior)
            mc_results[bi] += loss
    mc_results /= n_sample_batches
    return mc_results.detach().cpu().numpy()

if __name__ == '__main__':
    classifier = LDAF_Stochastic.load_from_checkpoint("split_erm_median.ckpt")
    classifier.eval()

    n_sobol_x = list(range(500, 1000, 100))+list(range(1000, 10000, 1000))+list(range(10000, 100000, 10000))
    n_trials = 100
    
    img, labels = load_single_batch(batch_size=n_trials)

    # select compute device
    gpu = torch.device('cuda:0')
    img = img.to(gpu)
    labels = labels.to(gpu)
    classifier.laf = classifier.laf.to(gpu)
    classifier.prior = classifier.prior.to(gpu)
    classifier.posterior = classifier.posterior.to(gpu)
    
    # load or compute reference solution
    reference_solution_path = "reference_solution_mc.npy"
    if os.path.isfile(reference_solution_path):
        reference_solution = np.load("reference_solution_mc.npy")
    else:
        with torch.no_grad():
            reference_solution = brute_force_mc(classifier, img, labels)
        np.save("reference_solution_mc.npy", reference_solution)
    assert not np.isnan(reference_solution.sum())

    # load or compute graph points
    if os.path.isfile("err_sobol_loss.npy"):
        err_sobol_loss = np.load("err_sobol_loss.npy")
        err_mc_loss = np.load("err_mc_loss.npy")
    else:
        print("sampling empirical_risk")
        err_sobol_loss = np.zeros((n_trials, len(n_sobol_x)))
        err_mc_loss = np.zeros((n_trials, len(n_sobol_x)))
        with torch.no_grad():
            for i in tqdm(range(n_trials)):
                for j, n_samples in enumerate(n_sobol_x):
                    mc_loss, qmc_loss = sample_empirical_risk(classifier, n_samples, img[i:i+1,...], labels[i:i+1])
                    err_sobol_loss[i,j] = np.abs(qmc_loss - reference_solution[i]) / reference_solution[0]
                    err_mc_loss[i,j] = np.abs(mc_loss - reference_solution[i]) / reference_solution[0]
        np.save("err_sobol_loss.npy", err_sobol_loss)
        np.save("err_mc_loss.npy", err_mc_loss)

    fig, ax = plt.subplots(1, 1)
    l1 = ax.errorbar(n_sobol_x, err_mc_loss.mean(axis=0), err_mc_loss.std(axis=0), label="MC")
    l2 = ax.errorbar(n_sobol_x, err_sobol_loss.mean(axis=0), err_sobol_loss.std(axis=0), label="QMC")
    ax.legend([l1[0],l2[0]], ["MC", "QMC"])
    ax.set_xlim(n_sobol_x[0], n_sobol_x[-1])
    ax.set_ylabel("Relative Error")
    ax.set_xlabel("# Samples")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()