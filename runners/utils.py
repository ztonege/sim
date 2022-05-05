import random
from argparse import ArgumentParser
from collections import OrderedDict

import numpy as np
import torch
import torchmetrics
from fastargs import get_current_config


class MeanScalarMetric(torchmetrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, sample: torch.Tensor):
        self.sum += sample.sum()
        self.count += sample.numel()

    def compute(self):
        return self.sum.float() / self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.0 * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


def remove_additional_layer(model_dict):
    removed_model_dict = OrderedDict()
    for layer_name in model_dict:
        removed_layer_name = layer_name
        if layer_name.startswith("module"):
            removed_layer_name = ".".join(layer_name.split(".")[1:])
        removed_model_dict[removed_layer_name] = model_dict[layer_name]
    return removed_model_dict


def reset_seed():
    seed = 16824
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
    args = config.get()
    return args
