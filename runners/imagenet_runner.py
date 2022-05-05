import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import List

import antialiased_cnns
import numpy as np
import torch as ch
import torch.distributed as dist
import torchmetrics
from fastargs import get_current_config
from fastargs.decorators import param
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    NormalizeImage,
    RandomHorizontalFlip,
    Squeeze,
    ToDevice,
    ToTensor,
    ToTorchImage,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm

from models.cifar import resnet_3D, resnet_group
from runners.utils import (
    AverageMeter,
    MeanScalarMetric,
    agreement,
    remove_additional_layer,
)

ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


@param("lr.lr")
@param("lr.step_ratio")
@param("lr.step_length")
@param("training.epochs")
def get_step_lr(epoch, lr, step_ratio, step_length, epochs):
    if epoch >= epochs:
        return 0

    num_steps = epoch // step_length
    return step_ratio**num_steps * lr


@param("lr.lr")
@param("training.epochs")
@param("lr.lr_peak_epoch")
def get_cyclic_lr(epoch, lr, epochs, lr_peak_epoch):
    xs = [0, lr_peak_epoch, epochs]
    ys = [1e-4 * lr, lr, 0]
    return np.interp([epoch], xs, ys)[0]


class ImageNetRunner:
    @param("training.distributed")
    @param("experiment.model_type")
    def __init__(self, gpu, distributed, model_type):
        self.all_params = get_current_config()
        self.gpu = gpu
        self.model_type = model_type
        self.step = 0

        if distributed:
            self.setup_distributed()

        self.train_loader = self.create_train_loader()
        self.val_loader = self.create_val_loader()
        self.model, self.scaler = self.create_model_and_scaler()
        self.create_optimizer()
        self.initialize_logger()

    @param("dist.address")
    @param("dist.port")
    @param("dist.world_size")
    def setup_distributed(self, address, port, world_size):
        os.environ["MASTER_ADDR"] = address
        os.environ["MASTER_PORT"] = port

        dist.init_process_group("nccl", rank=self.gpu, world_size=world_size)
        ch.cuda.set_device(self.gpu)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param("lr.lr_schedule_type")
    def get_lr(self, epoch, lr_schedule_type):
        lr_schedules = {"cyclic": get_cyclic_lr, "step": get_step_lr}

        return lr_schedules[lr_schedule_type](epoch)

    # resolution tools
    @param("resolution.min_res")
    @param("resolution.max_res")
    @param("resolution.end_ramp")
    @param("resolution.start_ramp")
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    @param("training.momentum")
    @param("training.optimizer")
    @param("training.weight_decay")
    @param("training.label_smoothing")
    def create_optimizer(self, momentum, optimizer, weight_decay, label_smoothing):
        assert optimizer == "sgd"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ("bn" in k)]
        other_params = [v for k, v in all_params if not ("bn" in k)]
        param_groups = [
            {"params": bn_params, "weight_decay": 0.0},
            {"params": other_params, "weight_decay": weight_decay},
        ]

        self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        self.loss = ch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    @param("data.train_dataset")
    @param("data.num_workers")
    @param("training.batch_size")
    @param("training.distributed")
    @param("data.in_memory")
    def create_train_loader(
        self, train_dataset, num_workers, batch_size, distributed, in_memory
    ):
        this_device = f"cuda:{self.gpu}"
        train_path = Path(train_dataset)
        assert train_path.is_file()

        res = self.get_resolution(epoch=0)
        self.decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        loader = Loader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=in_memory,
            drop_last=True,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        return loader

    @param("data.val_dataset")
    @param("data.num_workers")
    @param("validation.batch_size")
    @param("validation.resolution")
    @param("training.distributed")
    def create_val_loader(
        self, val_dataset, num_workers, batch_size, resolution, distributed
    ):
        this_device = f"cuda:{self.gpu}"
        val_path = Path(val_dataset)
        assert val_path.is_file()
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True),
        ]
        order = OrderOption.RANDOM
        loader = Loader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            drop_last=False,
            pipelines={"image": image_pipeline, "label": label_pipeline},
            distributed=distributed,
        )
        return loader

    @param("training.epochs")
    def train(self, epochs):
        for epoch in range(epochs):
            res = self.get_resolution(epoch)
            self.decoder.output_size = (res, res)
            train_stats = self.train_loop(epoch)

            # Validate
            val_stats = self.val_loop(epoch, True)
            content = {"train": train_stats, "val": val_stats}
            if self.gpu == 0:
                model_path = self.exp_dir / f"imagenet_{self.model_type}_{epoch}.pt"
                ch.save(self.model.state_dict(), model_path)
                self.log(epoch, "epoch", content)

        if self.gpu == 0:
            model_path = self.exp_dir / f"imagenet_{self.model_type}.pt"
            ch.save(self.model.state_dict(), model_path)
            ch.save(self.model.state_dict(), f"imagenet_{self.model_type}.pt")

    @param("training.distributed")
    def create_model_and_scaler(self, distributed):
        scaler = GradScaler()

        if self.model_type == "original":
            print("original for imagenet")
            model = getattr(models, "resnet18")(pretrained=False)

        elif self.model_type == "aacnn":
            print("aacnn for imagenet")
            model = antialiased_cnns.resnet18()

        elif self.model_type == "group":
            print("group for imagenet")
            model = resnet_group.ResNet18(num_classes=1000)

        elif self.model_type == "3D":
            print("3D for imagenet")
            model = resnet_3D.ResNet18(num_classes=1000)

        else:
            raise NotImplementedError()
        print(model)

        # model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        if distributed:
            model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])

        return model, scaler

    @param("validation.frequency")
    def train_loop(self, epoch, frequency):
        split = "train"
        self.model.train()

        # Reset meters
        [meter.reset() for meter in self.meters[split].values()]

        lr_start, lr_end = self.get_lr(epoch), self.get_lr(epoch + 1)
        iters = len(self.train_loader)
        lrs = np.interp(np.arange(iters), [0, iters], [lr_start, lr_end])

        start = time.time()
        with tqdm(self.train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"[Epoch {epoch}]")
            for ix, (images, target) in enumerate(tepoch):
                # Training start
                self.model.train()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lrs[ix]

                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    output = self.model(images)
                    loss_train = self.loss(output, target)

                self.scaler.scale(loss_train).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Training end

                # Logging start
                self.set_progress_bar_postfix(tepoch, split, output, target, loss_train)
                if self.step % frequency == 0:
                    train_stats = self.get_stats(split, time.time() - start)
                    val_stats = self.val_loop(self.step, False)
                    content = {"val": val_stats, "train": train_stats}
                    if self.gpu == 0:
                        self.log(self.step, "step", content)
                # Logging end
                self.step += 1

        stats = self.get_stats(split, time.time() - start)
        return stats

    @param("validation.count")
    @param("validation.lr_tta")
    def val_loop(self, epoch, full_set, count, lr_tta):
        count = -1 if full_set else count
        split = "val"
        self.model.eval()
        cur_count = 0

        # Reset meters
        [meter.reset() for meter in self.meters[split].values()]

        start = time.time()
        with ch.no_grad():
            with autocast():
                with tqdm(self.val_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"[Epoch {epoch}]")
                    for images, target in tepoch:
                        output = self.model(images)
                        if lr_tta:
                            output += self.model(ch.flip(images, dims=[3]))
                        loss_val = self.loss(output, target)
                        self.set_progress_bar_postfix(
                            tepoch, "val", output, target, loss_val
                        )
                        cur_count += len(target)
                        if count > 0 and cur_count >= count:
                            break

        run_time = time.time() - start
        stats = self.get_stats(split, run_time)
        return stats

    @param("validation.count")
    @param("validation.lr_tta")
    @param("validation.shift")
    def val_loop_shift(self, epoch, full_set, count, lr_tta, shift):
        count = -1 if full_set else count
        split = "val"
        self.model.eval()
        cur_count = 0

        # Reset meters
        [meter.reset() for meter in self.meters[split].values()]

        consistency_meter = AverageMeter()
        start = time.time()
        with ch.no_grad():
            with autocast():
                with tqdm(self.val_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"[Epoch {epoch}]")
                    for images, target in tepoch:
                        output = self.model(images)
                        if lr_tta:
                            output += self.model(ch.flip(images, dims=[3]))
                        loss_val = self.loss(output, target)

                        consistency = self.predict_consistency(images)
                        consistency_meter.update(consistency, images.shape[0])

                        self.set_progress_bar_postfix(
                            tepoch, "val", output, target, loss_val, consistency
                        )
                        cur_count += len(target)
                        if count > 0 and cur_count >= count:
                            break

        run_time = time.time() - start
        stats = self.get_stats(split, run_time)
        return stats, consistency_meter

    @param("validation.resolution")
    @param("validation.shift")
    def predict_consistency(self, images, shift, resolution):
        off0 = np.random.randint(shift, size=2)
        off1 = np.random.randint(shift, size=2)

        output0 = self.model(
            images[:, :, off0[0] : off0[0] + resolution, off0[1] : off0[1] + resolution]
        )
        output1 = self.model(
            images[:, :, off1[0] : off1[0] + resolution, off1[1] : off1[1] + resolution]
        )
        cur_agree = agreement(output0, output1).type(ch.FloatTensor).to(output0.device)
        return cur_agree.item()

    def set_progress_bar_postfix(
        self, tepoch, split, output, target, loss, consistency=0
    ):
        for k in ["top_1", "top_5"]:
            self.meters[split][k](output, target)
        self.meters[split]["loss"](loss)
        tepoch.set_postfix(
            {
                f"{split}_top_1": f"{self.meters[split]['top_1'].compute().cpu().numpy():.4f}",
                f"{split}_top_5": f"{self.meters[split]['top_5'].compute().cpu().numpy():.4f}",
                f"{split}_loss": f"{loss.detach().item():.4f}",
                f"{split}_consistency": f"{consistency:.4f}",
            }
        )

    def get_stats(self, split, run_time):
        stats = {"time": run_time}
        for k, m in self.meters[split].items():
            stats[k] = m.compute().item()
        return stats

    @param("experiment.root")
    @param("experiment.task")
    @param("experiment.dataset")
    @param("training.eval_only")
    def initialize_logger(self, root, task, dataset, eval_only):
        self.meters = {}
        for split in ["train", "val"]:
            self.meters[split] = {
                "top_1": torchmetrics.Accuracy(compute_on_step=False).to(self.gpu),
                "top_5": torchmetrics.Accuracy(compute_on_step=False, top_k=5).to(
                    self.gpu
                ),
                "loss": MeanScalarMetric(compute_on_step=False).to(self.gpu),
            }

        if self.gpu == 0 and not eval_only:
            date_str = datetime.now().__str__()
            self.exp_dir = (
                Path(root) / task / dataset / self.model_type / date_str
            ).absolute()

            self.exp_dir.mkdir(parents=True)
            self.writer = SummaryWriter(self.exp_dir)

            self.start_time = time.time()
            print(f"=> Logging in {self.exp_dir}")
            params = {
                ".".join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }

            with open(self.exp_dir / "params.json", "w+") as f:
                json.dump(params, f)

    def log(self, step, step_type, content):
        print(f"=> Log: {content}")
        if self.gpu != 0:
            return

        for split in content:
            for metric, value in content[split].items():
                self.writer.add_scalar(f"{split}/{step_type}-{metric}", value, step)

        cur_time = time.time()
        with open(self.exp_dir / "log", "a+") as fd:
            fd.write(
                json.dumps(
                    {
                        "timestamp": cur_time,
                        "relative_time": cur_time - self.start_time,
                        **content,
                    }
                )
                + "\n"
            )
            fd.flush()

    @param("validation.model_path")
    def test(self, model_path):
        epoch = 0
        model_dict = ch.load(model_path, map_location="cuda")
        model_dict = remove_additional_layer(model_dict)
        self.model.load_state_dict(model_dict)
        stats, consistency_meter = self.val_loop_shift(epoch, True)
        print(stats)
        print(f"Consistency Average: {consistency_meter.avg}")

    @classmethod
    @param("training.distributed")
    @param("dist.world_size")
    def launch_from_args(cls, distributed, world_size):
        if distributed:
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=world_size, join=True)
        else:
            make_config()
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param("training.distributed")
    @param("training.eval_only")
    def exec(cls, gpu, distributed, eval_only):
        trainer = cls(gpu=gpu)
        if eval_only:
            trainer.test()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="Fast imagenet training")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
