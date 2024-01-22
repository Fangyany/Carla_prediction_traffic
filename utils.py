# Copyright (c) 2020 Uber Technologies, Inc.
# Please check LICENSE for more detail

import numpy as np
import sys
import cv2
import os

import torch
from torch import optim


def index_dict(data, idcs):
    returns = dict()
    for key in data:
        returns[key] = data[key][idcs]
    return returns


def rotate(xy, theta):
    st, ct = torch.sin(theta), torch.cos(theta)
    rot_mat = xy.new().resize_(len(xy), 2, 2)
    rot_mat[:, 0, 0] = ct
    rot_mat[:, 0, 1] = -st
    rot_mat[:, 1, 0] = st
    rot_mat[:, 1, 1] = ct
    xy = torch.matmul(rot_mat, xy.unsqueeze(2)).view(len(xy), 2)
    return xy


def merge_dict(ds, dt):
    for key in ds:
        dt[key] = ds[key]
    return


class Logger(object):
    def __init__(self, log):
        self.terminal = sys.stdout
        self.log = open(log, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def load_pretrain(net, pretrain_dict):
    state_dict = net.state_dict()
    for key in pretrain_dict.keys():
        if key in state_dict and (pretrain_dict[key].size() == state_dict[key].size()):
            value = pretrain_dict[key]
            if not isinstance(value, torch.Tensor):
                value = value.data
            state_dict[key] = value
    net.load_state_dict(state_dict)


def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data



def to_long(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = to_long(data[key])
    if isinstance(data, list) or isinstance(data, tuple):
        data = [to_long(x) for x in data]
    if torch.is_tensor(data) and data.dtype == torch.int16:
        data = data.long()
    return data

class Optimizer(object):
    def __init__(self, params, coef=None):
        if not (isinstance(params, list) or isinstance(params, tuple)):
            params = [params]

        coef = [1.0] * len(params)
        self.coef = coef

        config = dict()
        config["opt"] = 'adam'
        config["lr"] = [1e-3, 1e-4]
        config["lr_epochs"] = [32]
        config["lr_func"] = StepLR(config["lr"], config["lr_epochs"])

        param_groups = []
        for param in params:
            param_groups.append({"params": param, "lr": 0})

        
        opt = config["opt"]
        assert opt == "sgd" or opt == "adam"
        if opt == "sgd":
            self.opt = optim.SGD(
                param_groups, momentum=config["momentum"], weight_decay=config["wd"]
            )
        elif opt == "adam":
            self.opt = optim.Adam(param_groups, weight_decay=0)

        self.lr_func = config["lr_func"]

        if "clip_grads" in config:
            self.clip_grads = config["clip_grads"]
            self.clip_low = config["clip_low"]
            self.clip_high = config["clip_high"]
        else:
            self.clip_grads = False

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self, epoch):
        if self.clip_grads:
            self.clip()

        lr = self.lr_func(epoch)
        for i, param_group in enumerate(self.opt.param_groups):
            param_group["lr"] = lr * self.coef[i]
        self.opt.step()
        return lr

    def clip(self):
        low, high = self.clip_low, self.clip_high
        params = []
        for param_group in self.opt.param_groups:
            params += list(filter(lambda p: p.grad is not None, param_group["params"]))
        for p in params:
            mask = p.grad.data < low
            p.grad.data[mask] = low
            mask = p.grad.data > high
            p.grad.data[mask] = high

    def load_state_dict(self, opt_state):
        self.opt.load_state_dict(opt_state)
    
    def state_dict(self):
        """
        Returns the state of the optimizer as a dictionary.
        It includes the learning rate, coefficients, and optimizer state.
        """
        state = {
            'lr_func': self.lr_func.state_dict(),
            'coef': self.coef,
            'opt_state': self.opt.state_dict()
        }
        return state

    # def load_state_dict(self, state_dict):
    #     """
    #     Loads the optimizer state.
    #     """
    #     self.coef = state_dict['coef']
    #     self.lr_func.load_state_dict(state_dict['lr_func'])
    #     self.opt.load_state_dict(state_dict['opt_state'])



# class StepLR:
#     def __init__(self, lr, lr_epochs):
#         assert len(lr) - len(lr_epochs) == 1
#         self.lr = lr
#         self.lr_epochs = lr_epochs

#     def __call__(self, epoch):
#         idx = 0
#         for lr_epoch in self.lr_epochs:
#             if epoch < lr_epoch:
#                 break
#             idx += 1
#         return self.lr[idx]
    
class StepLR:
    def __init__(self, lr, lr_epochs):
        assert len(lr) - len(lr_epochs) == 1
        self.lr = lr
        self.lr_epochs = lr_epochs

    def __call__(self, epoch):
        idx = 0
        for lr_epoch in self.lr_epochs:
            if epoch < lr_epoch:
                break
            idx += 1
        return self.lr[idx]

    def state_dict(self):
        """
        Returns the state of the learning rate scheduler as a dictionary.
        """
        return {'lr': self.lr, 'lr_epochs': self.lr_epochs}

    # def load_state_dict(self, state_dict):
    #     """
    #     Loads the learning rate scheduler state.
    #     """
    #     self.lr = state_dict['lr']
    #     self.lr_epochs = state_dict['lr_epochs']
