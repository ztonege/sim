from fastargs import Param, Section
from fastargs.validation import And, OneOf
from torchvision import models

from runners.cifar_runner import CifarRunner
from runners.imagenet_runner import ImageNetRunner
from runners.utils import make_config, reset_seed

Section("model", "model details").params(
    arch=Param(And(str, OneOf(models.__dir__())), default="resnet18"),
    pretrained=Param(int, "is pretrained? (1/0)", default=0),
)

Section("resolution", "resolution scheduling").params(
    min_res=Param(int, "the minimum (starting) resolution", default=160),
    max_res=Param(int, "the maximum (starting) resolution", default=160),
    end_ramp=Param(int, "when to stop interpolating resolution", default=0),
    start_ramp=Param(int, "when to start interpolating resolution", default=0),
)

Section("data", "data related stuff").params(
    train_dataset=Param(str, ".dat file to use for training", required=True),
    val_dataset=Param(str, ".dat file to use for validation", required=True),
    num_workers=Param(int, "The number of workers", required=True),
    in_memory=Param(int, "does the dataset fit in memory? (1/0)", required=True),
)

Section("lr", "lr scheduling").params(
    step_ratio=Param(float, "learning rate step ratio", default=0.1),
    step_length=Param(int, "learning rate step length", default=30),
    lr_schedule_type=Param(OneOf(["step", "cyclic"]), default="cyclic"),
    lr=Param(float, "learning rate", default=0.5),
    lr_peak_epoch=Param(int, "Epoch at which LR peaks", default=2),
)

Section("experiment", "experiment settings").params(
    root=Param(str, "root dir for experimetnts", required=True),
    task=Param(str, "task type", required=True),
    model_type=Param(str, "model type", required=True),
    dataset=Param(str, "dataset name", required=True),
    log_level=Param(int, "0 if only at end 1 otherwise", default=1),
)

Section("validation", "Validation parameters stuff").params(
    batch_size=Param(int, "The batch size for validation", default=512),
    resolution=Param(int, "final resized validation image size", default=224),
    lr_tta=Param(int, "should do lr flipping/avging at test time", default=1),
    count=Param(int, "number of sample to validate, -1 to validate all", default=-1),
    frequency=Param(int, "number of steps between validation", default=1000),
    shift=Param(int, "max number of pixels to shift when testing", default=1000),
    model_path=Param(str, "model path to test", default=""),
)

Section("training", "training hyper param stuff").params(
    eval_only=Param(int, "eval only?", default=False),
    batch_size=Param(int, "The batch size", default=512),
    optimizer=Param(And(str, OneOf(["sgd"])), "The optimizer", default="sgd"),
    momentum=Param(float, "SGD momentum", default=0.9),
    weight_decay=Param(float, "weight decay", default=4e-5),
    epochs=Param(int, "number of epochs", default=30),
    label_smoothing=Param(float, "label smoothing parameter", default=0.1),
    distributed=Param(int, "is distributed?", default=0),
    use_blurpool=Param(int, "use blurpool?", default=0),
)

Section("dist", "distributed training options").params(
    world_size=Param(int, "number gpus", default=1),
    address=Param(str, "address", default="localhost"),
    port=Param(str, "port", default="12355"),
)


def main():
    args = make_config()
    reset_seed()
    if args.experiment.task == "classification":
        if args.experiment.dataset == "imagenet":
            ImageNetRunner.launch_from_args()
        elif args.experiment.dataset == "cifar":
            runner = CifarRunner()
            runner.train()
    else:
        raise NotImplementedError()


if __name__ == "__main__":
    main()
