"""
Stochastic LDAF Classifier on single scale with densely connected graph.
"""
import torch
import torch.nn as nn
import math
import numpy as np
import pytorch_lightning as pl
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
from assignment import LAF
from fitness import DenseOmega
from uq import *
from components import mean_free, tangent_basis
from torch.distributions.normal import Normal
from distributions import *
from resnet import ResNet18

class LDAF_Classifier(nn.Module):
    """
    Densely connected graph with n nodes (c constant) 
    and populate s0 from resnet feature extraction.
    """
    def __init__(self, n=50, c=10, t=1.0):
        super(LDAF_Classifier, self).__init__()
        self.n = n
        self.c = c
        self.t_end = t
        self.features = ResNet18()
        self.features.linear = nn.Linear(512, n*c)
        self.lift = nn.Softmax(dim=1)
        self.laf = LAF(DenseOmega(n, c), t)

    def flow_init(self, x):
        batch_size = x.shape[0]
        v0 = self.features(x).reshape((batch_size, self.c, self.n, 1))
        s0 = self.lift(v0)
        return s0

    def forward(self, x):
        batch_size = x.shape[0]
        v0 = self.features(x).reshape((batch_size, self.c, self.n, 1))
        s0 = self.lift(v0)
        vT = self.laf(s0)[:,:,-1,0]
        return v0[:,:,-1,0] + vT

    def perturbed_forward(self, x, v0):
        """
        Classify batch x by LDAF starting from v0 on the tangent space.
        """
        batch_size = x.shape[0]
        v = self.features(x).reshape((batch_size, self.c, self.n, 1))
        s0 = self.lift(v)
        vT = self.laf(s0)[:,:,-1,0]
        Omega = self.laf.fitness.dense_matrix()
        perturbation = dense_matrix_exp_action(Omega, s0, self.t_end, v0)
        return v[:,:,-1,0] + vT + perturbation[:,:,-1,0]

def mean_accuracy(logits, labels):
    """
    Mean 01-loss.
    """
    return (torch.argmax(logits, dim=1) == labels).float().sum()/labels.shape[0]

def set_bn_eval(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()
        if isinstance(module, torch.nn.modules.BatchNorm3d):
            module.eval()

class LDAF_Stochastic(pl.LightningModule):
    """
    Train stochastic LDAF classifier.
    """
    def __init__(self, lr=5e-2, schedule='MultiStepLR', n=50, c=10, t=1.0, error_prob=0.01, num_data=1e4, num_sobol=int(1e4), scramble_sobol=False, dense_cov=False):
        super(LDAF_Stochastic, self).__init__()
        self.save_hyperparameters()

        # optimization
        self.loss = nn.CrossEntropyLoss()
        self.lr = lr
        self.schedule = schedule
        self.training_mode = 'prior_training'

        # architecture
        self.laf = LDAF_Classifier(n, c, t)
        if dense_cov:
            self.prior = MultivariateNormal(n, c)
            self.posterior = MultivariateNormal(n, c)
        else:
            self.prior = UpdatedNormal(n, c)
            self.posterior = UpdatedNormal(n, c)

        # PAC risk bound
        self.temperature_logit = nn.Parameter(-1.0*torch.ones(1))
        self.temperature = 0.2
        self.error_prob = error_prob
        self.num_data = num_data

        # QMC integration
        self.n_sobol_samples = num_sobol
        soboleng = torch.quasirandom.SobolEngine(dimension=c-1, scramble=scramble_sobol)
        self.sobol_samples = soboleng.draw(self.n_sobol_samples)
        if not scramble_sobol:
            # this constant shift does not change anything about the low-discrepancy
            # nature of the sobol points, but it prevents evaluating the inverse
            # normal CDF at 0.0 with probability 1.0
            self.sobol_samples = torch.fmod(self.sobol_samples + torch.rand(1), 1.0)

    def set_temperature(self, l):
        with torch.no_grad():
            self.temperature_logit.fill_(0.0)
            self.temperature_logit += l

    def train_prior(self):
        for p in self.laf.parameters():
            #p.requires_grad = True
            p.requires_grad = False
        for p in self.prior.parameters():
            p.requires_grad = True
        for p in self.posterior.parameters():
            p.requires_grad = False
        self.temperature_logit.requires_grad = False
        self.training_mode = 'prior_training'

    def train_deterministic_erm(self):
        for p in self.laf.parameters():
            p.requires_grad = True
        #for p in self.laf.features.parameters():
        #    p.requires_grad = False
        #for p in self.laf.features.linear.parameters():
        #    p.requires_grad = True
        for p in self.prior.parameters():
            p.requires_grad = False
        for p in self.posterior.parameters():
            p.requires_grad = False
        self.temperature_logit.requires_grad = False
        self.training_mode = 'deterministic_erm_training'

    def train_posterior(self):
        for p in self.laf.parameters():
            p.requires_grad = False
        for p in self.prior.parameters():
            p.requires_grad = False
        for p in self.posterior.parameters():
            p.requires_grad = True
        self.temperature_logit.requires_grad = False
        self.training_mode = 'posterior_training'

    def train_lambda(self):
        for p in self.laf.parameters():
            p.requires_grad = False
        for p in self.prior.parameters():
            p.requires_grad = False
        for p in self.posterior.parameters():
            p.requires_grad = False
        self.temperature_logit.requires_grad = True
        self.training_mode = 'lambda_training'

    def configure_optimizers(self):
        if self.training_mode == 'posterior_training':
            return SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        optimizer = SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr, momentum=0.9, weight_decay=1e-3)
        if self.schedule == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=200)
        elif self.schedule == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, factor=0.1)
        elif self.schedule == 'MultiStepLR':
            # this schedule was taken from
            # Sergey Zagoruyko and Nikos Komodakis. Wide Residual Networks. 
            # In Richard C. Wilson, Edwin R. Hancock and William A. P. Smith, editors,
            # Proceedings of the British Machine Vision Conference (BMVC),
            # pages 87.1-87.12. BMVA Press, September 2016. 
            # which Zhang:2019, Lookahead Optimizer: k steps forward, 1 step back
            # cited as standard for resnet + cifar
            scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        else:
            print(f"Unknown schedule {self.schedule}")
            return optimizer
        loss_monitor = {
            'deterministic_erm_training': 'val_loss_erm',
            'prior_training': 'val_loss_prior'
        }[self.training_mode]
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": loss_monitor}}

    def temperature(self):
        """
        Temperature parameter used in Thiemann risk bound.
        The bound holds for lambda in (0, 2), but the optimal
        value is always below 1 so we constrain to (0,1)
        """
        p = torch.zeros(2, device=self.temperature_logit.device)
        p[:1] += self.temperature_logit
        p[-1:] -= self.temperature_logit
        return self.temperature_lift(p)[0]

    def optimal_lambda(self, empirical_risk, kl, m, eps):
        kl_ = kl.detach().cpu().numpy()
        empirical_risk_ = empirical_risk.detach().cpu().numpy()
        l = (np.arange(10000)+1) / 10000
        bound = empirical_risk_ / (1-l/2) + (kl_+np.log(2*np.sqrt(m))-np.log(eps)) / (m*l*(1-l/2))
        return l[np.argmin(bound)], bound.min()

    def risk_bound(self, empirical_risk, l=None, eps=None):
        """
        Evaluate Thiemann risk bound.
        """
        if l is None:
            temp = self.temperature
        else:
            temp = l
        if eps is None:
            error_prob = self.error_prob
        else:
            error_prob = eps
        relative_entropy = self.posterior.kl_to(self.prior)
        kl_offset = math.log(2) + 0.5*math.log(self.num_data) - math.log(error_prob)
        slack = (relative_entropy + kl_offset)/(self.num_data*temp*(1-temp/2))        
        risk_bound = empirical_risk/(1-temp/2) + slack
        return risk_bound, relative_entropy

    def pushforward_covariance(self, x, distr, dense=True):
        batch_size = x.shape[0]
        v = self.laf.features(x).reshape((batch_size, self.laf.c, self.laf.n, 1))
        s0 = self.laf.lift(v)
        if dense:
            cov = dense_pushforward_class_covariance(self.laf.laf.fitness.dense_matrix(), s0, self.laf.laf.t_end, distr)
        else:
            cov = pushforward_class_covariance(self.laf.laf.fitness, s0, self.laf.laf.t_end, distr)
        return cov

    def empirical_risk(self, x, labels, distr, dense=False):
        """
        Empirical risk of given sample and UnivariateNormal distribution of classifiers.
        """
        batch_size = x.shape[0]
        
        # forward pass 
        # mean of pushforward distribution has logits v+vT
        v = self.laf.features(x).reshape((batch_size, self.laf.c, self.laf.n, 1))
        s0 = self.laf.lift(v)
        vT = self.laf.laf(s0)[:,:,-1,0]

        # covariance of pushforward distribution
        if dense:
            cov = dense_pushforward_class_covariance(self.laf.laf.fitness.dense_matrix(), s0, self.laf.laf.t_end, distr)
        else:
            cov = pushforward_class_covariance(self.laf.laf.fitness, s0, self.laf.laf.t_end, distr)
        cov_regular = cov[:,:-1,:-1]
        #H = torch.linalg.cholesky(cov_regular)
        U, S, Vh = torch.linalg.svd(cov_regular)
        H = U @ torch.diag_embed(torch.sqrt(S))

        # sample shifts from mean in logits
        device = v.device
        if not self.sobol_samples.device == device:
            self.sobol_samples = self.sobol_samples.to(device)
        normal_distr = Normal(torch.tensor([0.0], device=device), torch.tensor([1.0], device=device))
        z = torch.kron(torch.ones(batch_size, 1, device=device), self.sobol_samples)
        H_extended = torch.kron(H, torch.ones(self.n_sobol_samples, 1, 1, device=device))
        x = torch.einsum("bij,bj->bi", H_extended, normal_distr.icdf(z))
        logit_shifts = tangent_basis(x)

        # vectorized logits (batch_size*n_sobol_points, c)
        v_extended = torch.kron(v[:,:,-1,0], torch.ones(self.n_sobol_samples, 1, device=device))
        vT_extended = torch.kron(vT, torch.ones(self.n_sobol_samples, 1, device=device))
        labels_extended = torch.kron(labels, torch.ones(self.n_sobol_samples, dtype=torch.long, device=labels.device))
        logits = v_extended + vT_extended + logit_shifts
        
        empirical_risk = self.loss(logits, labels_extended)
        empirical_accuracy = (torch.argmax(logits, dim=1) == labels_extended).float().mean()

        return empirical_risk, empirical_accuracy

    def training_step(self, batch, batch_idx):
        """
        Train either prior/weights by empirical risk minimization
        or posterior by bound minimization.
        """
        img, labels = batch
        if self.training_mode == 'deterministic_erm_training':
            # train feature extractor and Omega by empirical risk minimization
            # this ignores all stochasticity and trains a deterministic classifier
            logits = self.laf(img)
            loss = self.loss(logits, labels)
            accuracy = (torch.argmax(logits, dim=1) == labels).float().mean()
            self.log('train_accuracy_erm', accuracy)
            self.log('train_loss_erm', loss)
        elif self.training_mode == 'prior_training':
            # train feature extractor, Omega and prior distribution by empirical risk minimization
            # this trains a stochastic classifier
            loss, accuracy = self.empirical_risk(img, labels, self.prior)
            self.log('train_loss_prior', loss)
            self.log('train_accuracy_prior', accuracy)
        elif self.training_mode == 'posterior_training':
            set_bn_eval(self.laf.features)
            # train posterior distribution by risk bound minimization
            # this trains a stochastic classifier
            emp_risk, accuracy = self.empirical_risk(img, labels, self.posterior)
            loss, kl = self.risk_bound(emp_risk)
            self.log('train_emp_risk_surrogate', emp_risk)
            self.log('train_kl', kl, prog_bar=True)
            self.log('train_accuracy_posterior', accuracy)
            self.log('train_bound_surrogate', loss)
        else:
            print("Unknown training mode", self.training_mode)
            raise NotImplemented
        return loss

    def validation_step(self, batch, batch_idx):
        """
        """
        img, labels = batch
        if self.training_mode == 'deterministic_erm_training':
            # train feature extractor and Omega by empirical risk minimization
            # this ignores all stochasticity and trains a deterministic classifier
            logits = self.laf(img)
            loss = self.loss(logits, labels)
            accuracy = (torch.argmax(logits, dim=1) == labels).float().mean()
            self.log('val_accuracy_erm', accuracy, prog_bar=True)
            self.log('val_loss_erm', loss)
        elif self.training_mode == 'prior_training':
            # train feature extractor, Omega and prior distribution by empirical risk minimization
            # this trains a stochastic classifier
            loss, accuracy = self.empirical_risk(img, labels, self.prior)
            self.log('val_loss_prior', loss)
            self.log('val_accuracy_prior', accuracy, prog_bar=True)
        elif self.training_mode == 'posterior_training':
            # train posterior distribution by risk bound minimization
            # this trains a stochastic classifier
            emp_risk, accuracy = self.empirical_risk(img, labels, self.posterior)
            loss, _ = self.risk_bound(emp_risk)
            self.log('val_emp_risk_surrogate', emp_risk)
            self.log('val_accuracy_posterior', accuracy)
            self.log('val_bound_surrogate', loss, prog_bar=True)
        else:
            print("Unknown training mode", self.training_mode)
            raise NotImplemented

    def test_step(self, batch, batch_idx):
        # unpack batch
        img, labels = batch

        # simulate posterior
        v0 = self.posterior.draw(img.shape[0])
        logits_stochastic_posterior = self.laf.perturbed_forward(img, v0)
        accuracy_stochastic = (torch.argmax(logits_stochastic_posterior, dim=1) == labels).float().mean()
        loss_stochastic = self.loss(logits_stochastic_posterior, labels)
        self.log('test_accuracy_stochastic_posterior', accuracy_stochastic)
        self.log('test_loss_stochastic_posterior', loss_stochastic)

        # simulate prior
        v0 = self.prior.draw(img.shape[0])
        logits_stochastic_prior = self.laf.perturbed_forward(img, v0)
        accuracy_stochastic = (torch.argmax(logits_stochastic_prior, dim=1) == labels).float().mean()
        loss_stochastic = self.loss(logits_stochastic_prior, labels)
        self.log('test_accuracy_stochastic_prior', accuracy_stochastic)
        self.log('test_loss_stochastic_prior', loss_stochastic)

        # test mean classifier
        logits_mean = self.laf(img)
        accuracy_mean = (torch.argmax(logits_mean, dim=1) == labels).float().mean()
        loss_mean = self.loss(logits_mean, labels)
        self.log('test_accuracy_mean', accuracy_mean)
        self.log('test_loss_mean', loss_mean)

        # evaluate empirical risk and certificate
        prior_risk_01_surrogate, prior_accuracy = self.empirical_risk(img, labels, self.prior)
        self.log('test_loss_prior', prior_risk_01_surrogate)
        self.log('test_risk_prior', 1-prior_accuracy)
        posterior_risk_01_surrogate, posterior_accuracy = self.empirical_risk(img, labels, self.posterior)
        self.log('test_loss_posterior', posterior_risk_01_surrogate)
        self.log('test_risk_posterior', 1-posterior_accuracy)
        bound1, kl = self.risk_bound(1-posterior_accuracy, eps=0.01)
        bound5, _ = self.risk_bound(1-posterior_accuracy, eps=0.05)
        self.log('test_kl', kl)
        self.log('test_risk_bound_1', bound1)
        self.log('test_risk_bound_5', bound5)