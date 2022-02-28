import argparse
from datetime import datetime
from pathlib import Path
from time import sleep

import DataLoad as DL
from AdversarialTrainer import AdversarialTrainer

LOADERS = {"cifar10": DL.get_loaders_cifar10, "cifar100": DL.get_loaders_cifar100}

MODELS = {
    "cifar10": "cifar10_resnet18_baseline_nat_acc_94.pt",
    "cifar100": "cifar100_resnet18_baseline_nat_acc_76p3.pt",
}


def now_datetime():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


def mk_topdir(topdir: str):
    dt_string = now_datetime()
    output_dir = Path(topdir, dt_string)
    if output_dir.is_dir():
        sleep(2)
        dt_string = now_datetime()
        output_dir = Path(topdir, dt_string)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir = str(output_dir)
    return output_dir


def setup_parser():
    _desc = """
    Code to train models on Cifar10/100 using various adversarial training
    methods. For parameter details see the train function in 
        AdversarialTrainer.py

    """
    parser = argparse.ArgumentParser(description=_desc)
    parser.add_argument("--data_path", default="./data", help="Path to data directory.")
    parser.add_argument(
        "--mode",
        default="fw",
        choices=[
            "fw",
            "pgd",
            "standard",
            "free",
            "fw_adapt",
            "fgsm_adapt",
            "grad_align",
        ],
        help="AT Mode.",
    )
    parser.add_argument(
        "--eval_mode", choices=["fw", "pgd"], default="pgd"
    )
    parser.add_argument(
        "--grad_align_lambda", default=1.0, type=float, help="Grad align reg strength"
    )
    parser.add_argument(
        "--K", default=5, help="Number of FW steps for training", type=int
    )
    parser.add_argument(
        "--ep_decay", default=15, help="Steps to decay the LR", type=int
    )
    parser.add_argument(
        "--topdir", default="experiments", help="Top directory to save results"
    )
    parser.add_argument("--eval_every", default=5, type=int)
    parser.add_argument("--lr", default=0.1, type=float)
    parser.add_argument("--min_dr", default=2.0, type=float)
    parser.add_argument("--epsilon", default=8 / 255.0, type=float)
    parser.add_argument("--log_every_n_batches", default=15, type=int)
    parser.add_argument("--num_epochs", default=30, type=int)
    parser.add_argument("--do_rob_plot", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--initialize", default="zero")
    parser.add_argument(
        "--multi_step",
        default=2,
        type=int,
        help="Number of steps to change by when using mode==fw_adapt",
    )
    parser.add_argument(
        "--alpha", default=2.5, type=float, help="PGD step will be alpha*eps/K"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    return parser


if __name__ == "__main__":

    parser = setup_parser()
    args = parser.parse_args()

    ds = args.dataset

    cpath = None
    if args.pretrained:
        cpath = MODELS[ds]

    eps_tag = str(args.epsilon)

    eps_tag = eps_tag.replace("0.03137254901960784", "8")
    eps_tag = eps_tag.replace("0.06274509803921569", "16")

    topdir = args.topdir + f"/{ds}/eps_{eps_tag}"
    topdir = mk_topdir(topdir)
    topdir = Path(topdir)

    train_args = {
        "mode": args.mode,
        "ep_decay": args.ep_decay,
        "learning_rate": args.lr,
        "K": args.K,
        "alpha": args.alpha,
        "min_distortion_ratio": args.min_dr,
        "epsilon": args.epsilon,
        "n_epochs": args.num_epochs,
        "checkpoint_path": cpath,
        "adv_init": args.initialize,
        "multi_step": args.multi_step,
        "grad_align_lambda": args.grad_align_lambda,
        "eval_mode": args.eval_mode
    }

    ignore_tags = [
        "checkpoint_path",
        "epsilon",
        "ep_decay",
        "learning_rate",
        "n_epochs",
    ]
    ttag = "_".join(
        [f"{k}_{train_args[k]}" for k in train_args if k not in ignore_tags]
    )
    ttag = ttag.replace(".", "p").replace("min_distortion_ratio", "min_dr")
    ttag = ttag.replace("learning_rate", "lr")
    ttag = ttag.replace("learning_rate", "lr")
    ttag = ttag.replace("0p03137254901960784", "8")
    ttag = ttag.replace("0p06274509803921569", "16")

    # load data
    loader = LOADERS[ds]
    train_loader, val_loader = loader(
        data_path=args.data_path,
        batch_size_train=512,
        batch_size_val=512,
        num_workers=14,
    )
    print(
        "# train batches = ", len(train_loader), ", # val batches = ", len(val_loader)
    )

    num_classes = 100 if ds == "cifar100" else 10
    at = AdversarialTrainer("resnet18", topdir=topdir, num_classes=num_classes)

    exp_path_check = at.topdir.joinpath(ttag)

    exp_path = at.train(
        train_loader,
        val_loader,
        eval_freq=args.eval_every,
        adv_norm="Linf",
        exp_tag=ttag,
        c=2,
        log_every_n_batches=args.log_every_n_batches,
        **train_args,
    )
