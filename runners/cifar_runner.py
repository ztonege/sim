"""
Fast training script for CIFAR-10 using FFCV.
For tutorial, see https://docs.ffcv.io/ffcv_examples/cifar10.html.

First, from the same directory, run:

    `python write_datasets.py --data.train_dataset [TRAIN_PATH] \
                              --data.val_dataset [VAL_PATH]`

to generate the FFCV-formatted versions of CIFAR.

Then, simply run this to train models with default hyperparameters:

    `python train_cifar.py --config-file default_config.yaml`

You can override arguments as follows:

    `python train_cifar.py --config-file default_config.yaml \
        --training.lr 0.2 --training.num_workers 4 ... [etc]`

or by using a different config file.
"""
from typing import List

import numpy as np
import torch
import torch as ch
import torchvision
from fastargs.decorators import param
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    Convert,
    Cutout,
    RandomHorizontalFlip,
    RandomTranslate,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from ffcv.transforms.common import Squeeze
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
from tqdm import tqdm

from models.cifar import (
    resnet,
    resnet_3D,
    resnet_anti_aliased,
    resnet_group,
    vgg_aa,
    vgg_orig,
    vgg_group,
    vgg_3D,
)


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


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]


class CifarRunner:
    def __init__(self):
        self.loaders = self.make_dataloaders()
        self.model = self.construct_model()

    @param("data.train_dataset")
    @param("data.val_dataset")
    @param("training.batch_size")
    @param("data.num_workers")
    def make_dataloaders(
        self,
        train_dataset=None,
        val_dataset=None,
        batch_size=None,
        num_workers=None,
    ):
        paths = {"train": train_dataset, "test": val_dataset}
        loaders = {}

        for name in ["train", "test"]:
            label_pipeline: List[Operation] = [
                IntDecoder(),
                ToTensor(),
                ToDevice("cuda:0"),
                Squeeze(),
            ]
            image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]
            if name == "train":
                image_pipeline.extend(
                    [
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                        Cutout(4, tuple(map(int, CIFAR_MEAN))),
                    ]
                )
            image_pipeline.extend(
                [
                    ToTensor(),
                    ToDevice("cuda:0", non_blocking=True),
                    ToTorchImage(),
                    Convert(ch.float16),
                    torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )

            ordering = OrderOption.RANDOM if name == "train" else OrderOption.SEQUENTIAL

            loaders[name] = Loader(
                paths[name],
                batch_size=batch_size,
                num_workers=num_workers,
                order=ordering,
                drop_last=(name == "train"),
                pipelines={"image": image_pipeline, "label": label_pipeline},
            )

        return loaders

    @param("experiment.model_type")
    def construct_model(self, model_type):
        if model_type == "original":
            print("original for cifar")
            model = resnet.ResNet18(num_classes=10)
            print(model)
        elif model_type == "aacnn":
            print("aacnn for cifar")
            model = resnet_anti_aliased.resnet18(num_classes=10)
            print(model)
        elif model_type == "group":
            print("group for cifar")
            model = resnet_group.ResNet18(num_classes=10)
            print(model)
        elif model_type == "3D":
            print("3D for cifar")
            model = resnet_3D.ResNet18(num_classes=10)
            print(model)
        elif model_type == "vgg_original":
            print("original vgg for cifar")
            model = vgg_orig.vgg11()
            print(model)
        elif model_type == "vgg_aa":
            print("aa vgg for cifar")
            model = vgg_aa.vgg11()
            print(model)
        elif model_type == "vgg_group":
            print("aa vgg for cifar")
            model = vgg_group.VGG(3)
            print(model)
        elif model_type == "vgg_3D":
            print("aa vgg for cifar")
            model = vgg_3D.VGG(3)
            print(model)
        else:
            raise NotImplementedError()

        model = model.cuda()
        return model

    @param("lr.lr")
    @param("training.epochs")
    @param("training.momentum")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    @param("lr.lr_peak_epoch")
    @param("experiment.model_type")
    def train(
        self,
        lr=None,
        epochs=None,
        label_smoothing=None,
        momentum=None,
        weight_decay=None,
        lr_peak_epoch=None,
        model_type="original",
    ):
        opt = SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        iters_per_epoch = len(self.loaders["train"])
        # Cyclic LR with single triangle
        lr_schedule = np.interp(
            np.arange((epochs + 1) * iters_per_epoch),
            [0, lr_peak_epoch * iters_per_epoch, epochs * iters_per_epoch],
            [0, 1, 0],
        )
        scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)
        scaler = GradScaler()
        loss_fn = CrossEntropyLoss(label_smoothing=label_smoothing)

        for _ in range(epochs):
            self.model.train()
            for ims, labs in tqdm(self.loaders["train"]):
                opt.zero_grad(set_to_none=True)
                with autocast():
                    out = self.model(ims)
                    loss = loss_fn(out, labs)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                scheduler.step()
            self.evaluate()

        ch.save(self.model.state_dict(), f"./cifar_{model_type}.pt")

    @param("validation.lr_tta")
    def evaluate(self, lr_tta=False):
        self.model.eval()
        with ch.no_grad():
            for name in ["test"]:
                total_correct, total_num = 0.0, 0.0
                for ims, labs in tqdm(self.loaders[name]):
                    with autocast():
                        out = self.model(ims)
                        if lr_tta:
                            out += self.model(ch.fliplr(ims))
                        total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                        total_num += ims.shape[0]
                print(f"{name} accuracy: {total_correct / total_num * 100:.1f}%")

    @param("validation.lr_tta")
    @param("experiment.model_type")
    def test(self, lr_tta=False, model_type="original"):
        device = "cuda" if ch.cuda.is_available() else "cpu"
        self.model.load_state_dict(
            ch.load(f"./cifar_{model_type}.pt", map_location=device)
        )
        self.model.eval()
        with ch.no_grad():
            for name in ["test"]:
                total_correct, total_num = 0.0, 0.0
                for ims, labs in tqdm(self.loaders[name]):
                    with autocast():
                        out = self.model(ims)
                        if lr_tta:
                            out += self.model(ch.fliplr(ims))
                        total_correct += out.argmax(1).eq(labs).sum().cpu().item()
                        total_num += ims.shape[0]
                print(f"{name} accuracy: {total_correct / total_num * 100:.1f}%")

        consist = AverageMeter()
        with ch.no_grad():
            for name in ["test"]:
                total_correct, total_num = 0.0, 0.0
                for ims, labs in tqdm(self.loaders[name]):
                    with autocast():
                        off0 = np.random.randint(7, size=2)
                        off1 = np.random.randint(7, size=2)
                        output0 = self.model(
                            ims[:, :, off0[0] : off0[0] + 32, off0[1] : off0[1] + 32]
                        )
                        output1 = self.model(
                            ims[:, :, off1[0] : off1[0] + 32, off1[1] : off1[1] + 32]
                        )
                        cur_agree = (
                            agreement(output0, output1)
                            .type(torch.FloatTensor)
                            .to(output0.device)
                        )
                        consist.update(cur_agree.item(), ims.size(0))
                        print("consist.avg:", consist.avg, "consist.val", consist.val)


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.0 * torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree
