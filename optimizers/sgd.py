"""Code copied from https://github.com/HuangxingLin123/Learning-Rate-Dropout/blob/master/cifar10/sgd_lrd.py"""
import torch
from torch.optim.optimizer import Optimizer


class SGD_LRD(Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, dropout=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,dropout=dropout)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_LRD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_LRD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                ## mask
                m = torch.ones_like(p.data) * group['dropout']
                mask = torch.bernoulli(m)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)

                ##dropout learning rate
                lr_dropout = group['lr']*mask
                I_buf = lr_dropout * buf.clone()

                p.data.add_(-1, I_buf)

        return loss


# Code copied from https://github.com/AtheMathmo/AggMo/blob/master/src/aggmo.py
class SGDAggMo(Optimizer):
    r"""Implements Aggregated Momentum Gradient Descent
    """

    def __init__(self, params, lr, betas=[0.0, 0.9, 0.99], weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(SGDAggMo, self).__init__(params, defaults)

    @classmethod
    def from_exp_form(cls, params, lr, a=0.1, k=3, weight_decay=0):
        betas = [1 - a**i for i in range(k)]
        return cls(params, lr, betas, weight_decay)

    def __setstate__(self, state):
        super(SGDAggMo, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            betas = group['betas']
            total_mom = float(len(betas))

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = {}
                    for beta in betas:
                        param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)
                for beta in betas:
                    buf = param_state['momentum_buffer'][beta]
                    # import pdb; pdb.set_trace()
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group['lr'] / total_mom , buf)
        return loss

    def zero_momentum_buffers(self):
        for group in self.param_groups:
            betas = group['betas']
            for p in group['params']:
                param_state = self.state[p]
                param_state['momentum_buffer'] = {}
                for beta in betas:
                    param_state['momentum_buffer'][beta] = torch.zeros_like(p.data)

    def update_hparam(self, name, value):
        for param_group in self.param_groups:
            param_group[name] = value
