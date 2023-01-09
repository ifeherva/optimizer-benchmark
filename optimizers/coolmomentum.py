import torch
from torch.optim.optimizer import Optimizer

"""Code copied from https://github.com/borbysh/coolmomentum/blob/master/optimizers/coolmom_pytorch.py"""


class Coolmomentum(Optimizer):
    """
        lr (float): learning rate
        momentum (float, optional): initial momentum constant (0 for SGD)
        weight_decay (float, optional): weight decay (L2 penalty)
        beta: cooling rate, close to 1, if beta=1 then no cooling
    """

    def __init__(self, params, lr=0.01, momentum=0,
                 weight_decay=0.0, beta=1.0, dropout=0.0):

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, beta=beta, dropout=dropout)
        super(Coolmomentum, self).__init__(params, defaults)
        self.T = 0.0
        self.number = 0
        self.iteration = 0
        self.filename = 'temperature.txt'

    def __setstate__(self, state):
        super(Coolmomentum, self).__setstate__(state)

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            beta = group['beta']
            beta_power = beta ** self.iteration

            rho_0 = momentum
            rho = 1 - (1 - rho_0) / beta_power
            rho = max(rho, 0)
            lrn = group['lr'] * (1 + rho) / 2  # lrn instead of lr
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported')
                state = self.state[p]
                # mask
                # m = torch.ones_like(p.data) * group['dropout']
                # mask = torch.bernoulli(m)
                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros_like(p.data)

                step = state['step']

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                # Update the step
                step.mul_(rho).add_(grad, alpha=-lrn)

                self.T += torch.sum(step * step)  # squared step
                self.number += torch.sum(step * 0 + 1)  # of trainable params
                p.data.add_(step)

        if self.iteration % 391 == 0:
            self.T = self.T / self.number  # compute temperature
            tfile = open(self.filename, "a+")
            print(self.iteration, self.T, rho, file=tfile)
            tfile.close()
            self.T = 0.0
            self.number = 0
        self.iteration += 1

        return None
