from pathlib import Path

import autoattack as aa
import argparse
import yaml
from AdversarialTrainer import AdversarialTrainer
import DataLoad as DL

LOADERS = {"cifar10": DL.get_loaders_cifar10, "cifar100": DL.get_loaders_cifar100}

MODELS = {
    "cifar10": "cifar10_resnet18_baseline_nat_acc_94.pt",
    "cifar100": "cifar100_resnet18_baseline_nat_acc_76p3.pt",
}


def get_best_checkpoint_simple(cd):
    bpath = list(Path(cd).rglob("checkpoint__best.pt"))
    return bpath[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--ds", default="cifar10")
    parser.add_argument("--data_path", default="./data", help="Path to data directory.")
    parser.add_argument("--break_at", default=-1)

    args = parser.parse_args()

    # load data
    loader_fn = LOADERS[args.ds]
    _, val_loader = loader_fn(
        data_path=args.data_path,
        batch_size_train=16,
        batch_size_val=2048,
        num_workers=20,
    )

    exp_dir = Path(args.exp_dir)
    logout_path = exp_dir.joinpath(f"autoattack_results_{args.tag}.txt")

    with exp_dir.joinpath("hparams.yaml").open("r") as f:
        eps = yaml.safe_load(f)["epsilon"]

    # Load model
    model_path = (
        get_best_checkpoint_simple(exp_dir)
        if args.model_path is None
        else args.model_path
    )

    num_classes = 100 if args.ds == "cifar100" else 10
    model = AdversarialTrainer("resnet18", topdir=exp_dir, num_classes=num_classes)
    model.load_model(model_path)

    model.model.eval()

    def forward_pass(x):
        return model.forward(x, inplace=False)

    adversary = aa.AutoAttack(forward_pass, norm="Linf", eps=eps, log_path=logout_path)

    # Run the evaluation
    num_cor = 0
    num_total = 0.0
    num_b = 0
    for batch in val_loader:
        num_b += 1
        if num_b == args.break_at:
            break
        x0, y0 = batch
        _, yadv = adversary.run_standard_evaluation(x0, y0, return_labels=True, bs=1024)

        cor = (yadv.cpu() == y0).sum().item()
        num_cor += cor
        num_total += y0.shape[0]

    res = {"aa_acc": 100 * (num_cor / num_total)}

    eval_args = {
        "mode": "pgd",
        "K": 50,
        "epsilon": eps,
        "adv_norm": "Linf",
        "adv": True,
    }

    eval_res = model.evaluate(val_loader, **eval_args)

    res["pgd50_acc"] = eval_res["acc"]
    with exp_dir.joinpath("autoattack_and_pgd50_value.yaml").open("w") as f:
        yaml.dump(res, f)
