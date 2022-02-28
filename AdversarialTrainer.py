#
# Frank-Wolfe Adaptive Adversarial Training (FW-Adapt-AT)
# Code used in paper
#   Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training (2021)
#

import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.autograd import Variable
from tqdm import tqdm

from attack_steps import FGSMAdaptAttacker
from densenet import DenseNet121
from inception import inceptionv3 as InceptionV3
from resnet import ResNet18, ResNet50

MODEL_FNS = {
    "resnet18": ResNet18,
    "resnet50": ResNet50,
    "inceptionv3": InceptionV3,
    "densenet121": DenseNet121,
}

class AdversarialTrainer:
    def __init__(self, arch:str="resnet18", num_classes:int=10, topdir:str="fw_experiments"):
        """
        Class to hold building, training, and evaluating models.

        Parameters
        ----------
            arch : 
                One of MODEL_FNS keys. Indicates which type of model to train.
            num_classes :
                The number of target classes.
            topdir :
                Where to save any results.
        """
        self.topdir = Path(topdir)
        self.topdir.mkdir(exist_ok=True, parents=True)

        self.model_args = {"arch": arch, "num_classes": num_classes}
        self.model = MODEL_FNS[arch](num_classes=num_classes)
        
        self.device = (
            torch.device("cuda") if torch.cuda.is_available else torch.device("gpu")
        )
        
        if torch.cuda.device_count() > 1:
            print("Using ", torch.cuda.device_count(), "GPUs")
            self.model = nn.DataParallel(self.model)
        
        print("device detected: ", self.device)
        self.model.to(device=self.device)
        self.loss_fn = nn.CrossEntropyLoss()

    def l2_distortion(self, d_adv:torch.Tensor, epsilon:float)->torch.Tensor:
        """
        Compute the ratio fo the L2 norm of d_adv and the max L2 norm
        of a point in the Linf ball of radius epsion.

        Parameters
        ----------
            d_adv:
                Tensor of shape (B, s1, s2, ..., sN)
            epsilon:
                Size of Linf ball to take ratio
        Returns
        -------
            distortion : 
                Tensor of shape (B, 1) whose ith element
                is the distortion ratio of d_adv[i]
        """
        B = d_adv.shape[0]
        RES = np.prod(d_adv.shape[1:])

        distortion = torch.norm( d_adv.view(B, -1), p=2, dim=1)
        distortion = distortion / (pow(RES, 0.5) * epsilon)
        return distortion

    def train(
        self,
        train_loader,
        val_loader,
        n_epochs:int,
        input_shape=(3,32,32),
        eval_freq:int=5,
        learning_rate:float=0.1,
        weight_decay:float=5e-4,
        momentum:float=0.9,
        ep_decay:int=50,
        gamma:float=0.1,
        adv_norm:str="Linf",
        adv_init:str="zero",
        epsilon:float=8 / 255,
        K:int=7,
        alpha:float=2.5 ,#* (8 / 255) / 7,
        c:float=2.0,
        mode:str="standard",
        exp_tag:str="experiments",
        log_every_n_batches:int=15,  
        checkpoint_path:str=None,
        min_distortion_ratio:float=0.9, # distortion ratio to increase back to K steps
        multi_step:int=1, # Number of steps to change by in fw-adapt-multi
        grad_align_lambda:float=1.0,
        eval_mode:str="pgd",
    ):
        """
        Trains the model via Xentr loss using SGD+Nesterov momentum optimizer.
        Results are saved in 
            |-topdir/
                |-exp_tag/
                    |-checkpoints/
                        |-<model checkpoints>.pt
                    |-hparams.yaml (params used in trainer)
                    |-train_results.csv
                    |-eval_results.csv
        MODES:
            standard : 
                Standard Xentr training
            pgd : 
                PGD-AT of strength `epsilon` using ``K steps and 
                step size `alpha` * 2.5 * epsilon / K
                    `Towards Deep Learning Models Resistant to Adversarial Attacks`
                    Madry, Makelov, Schmidt, et. al. (2018)
                    https://arxiv.org/abs/1706.06083
            fgsm_adapt : 
                FGSM-Adapt of strength `epsilon` with `K` checkpoints
                    `Understanding Catastrophic Overfitting in Single-step Adversarial Training`
                    Kim, Lee, Lee (2020)
                    https://arxiv.org/pdf/2010.01799.pdf
            grad_align : 
                Gradient alignment of strength `epsilon`
                 regularizer with strength `grad_align_lambda`
                    `Understanding and ImprovingFast Adversarial Training`
                    Andriushchenko, Flammarion (2020)
                    https://arxiv.org/pdf/2007.02617.pdf
            free : 
                Free AT of strength `epsilon` with `K` minibatch replays
                    `Adversarial Training for Free!`
                    Shafahi, Najibi, Ghiasi, Xu, et. al. (2019)
                    https://arxiv.org/abs/1904.12843
            fw : 
                Frank-Wolfe-AT of strength `epsilon` using `K` steps 
                and FW step size `c`
                    K = 1 + adv_init=random : 
                        `Fast is better than free: Revisiting adversarial training`
                        Wong, Rice, Kolter (2020)
                        https://arxiv.org/abs/2001.03994
                    K > 1:
                        (Ours 2021)
            fw_adapt : 
                Adaptive Frank-Wolfe AT of 
                strength `epsilo`n and FW step size `c`
                    - Start training with 1 attack step
                    - Start epochs with FW-2 and log the distortion
                    - At 1000 images check the average distortion d
                    - If d > `min_distortion_ratio` swithc to attack steps / 2
                    - Else increase by + `multi_step`
                (Ours 2021)

        Parameters Not Mentioned Above
        ------------------------------
            train_loader : torch DataLoader
                Inputs to train on.
            val_loader : torch DataLoader
                Inputs to validate on.
            n_epochs : 
                Number of epochs to train for.
            input_shape :
                Shape of input, only needed if mode==free
            eval_freq : 
                run validation every `eval_freq` epochs
            learning_rate :
                SGD learning rate
            weight_decay : 
                SGD weight decay
            gamma :
                SGD + Nesterov momentum strength
            adv_norm :
                Either L2 or Linf (should only do Linf for distortion work)
            exp_tag :
                Additional directory to save work under
            log_every_n_batches :
                How often to update train_results.csv
            checkpoint_path : 
                Path of model weights to load
            eval_mode : 
                Either `pgd` or `fw` dictating which method to generate a
                10 step attack for validation.
        """
        assert mode in ["fw_adapt", "fw", "pgd", "standard", "fgsm_adapt", "grad_align", "free"]
        if checkpoint_path is not None:
            self.load_model(checkpoint_path)
        
        alpha = alpha * epsilon / float(K)

        exp_path = self.topdir.joinpath(exp_tag)
        self.checkpoint_dir = exp_path.joinpath("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        train_params = {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "ep_decay": ep_decay,
            "adv_norm": adv_norm,
            "epsilon": epsilon,
            "K": K,
            "alpha": alpha,
            "c": c,
            "mode": mode,
            "model_args": self.model_args,
            "checkpoint_path": checkpoint_path,
            'min_distortion_ratio': min_distortion_ratio,
            "adv_init": adv_init,
            "multi_step": multi_step,
            "grad_align_lambda": grad_align_lambda,
            "eval_mode": eval_mode
        }

        with exp_path.joinpath("hparams.yaml").open("w") as f:
            yaml.dump(train_params, f)

        print("Training mode:", mode)
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=True,
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=ep_decay, gamma=gamma
        )

        acc_eval_best = 0.0
        self.model.train()
        t0 = time.time()

        batch_size = train_loader.batch_size
        free_delta = torch.zeros((batch_size, *input_shape), device=self.device)

        attack_steps=K
        if mode == "fw_adapt":
            attack_steps = 1
            multi_train_steps = 1
        
        for epoch in range(n_epochs):
            self.model.train()
            loss_train = 0.0
            # Training Loop
            batch = 0
            num_images = 0
            epoch_distortion_check = 0.0
            epoch_checked = False
            
            if mode == "fw_adapt":
                attack_steps=2
                
            train_iterator = tqdm(train_loader)
            for tensor_batch in train_iterator:
                # Reset attack steps when monitoring\
                x:torch.Tensor = tensor_batch[0]
                y:torch.Tensor = tensor_batch[1]

                num_images += x.shape[0]
                x = x.to(device=self.device)  # Bx3x32x32
                y = y.to(device=self.device)
                if mode != "free":                  
                    d_adv = torch.zeros_like(x).detach()
                    if mode in ["fw", "fw_adapt"]:
                        d_adv = find_adv_input_fw(
                            self.model,
                            x,
                            y,
                            epsilon,
                            attack_steps,
                            c,
                            self.device,
                            adv_norm,
                            self.loss_fn,
                            initialize=adv_init
                        )
                        self.model.train()
                        outputs = model_n(self.model, x + d_adv)
                    elif mode == "pgd":
                        d_adv = find_adv_input_pgd(
                            self.model,
                            x,
                            y,
                            epsilon,
                            attack_steps,
                            alpha,
                            self.device,
                            adv_norm,
                            self.loss_fn,
                            initialize=adv_init
                        )
                        self.model.train()
                        outputs = model_n(self.model, x + d_adv)
                    elif mode == "standard":
                        self.model.train()
                        outputs = model_n(self.model, x)
                    elif mode == "fgsm_adapt":
                        fgsm_a_attacker = FGSMAdaptAttacker(epsilon, adapt_steps=K)
                        d_adv = fgsm_a_attacker.get_adv_perturbation(self.model, x, y)
                        
                        self.model.train()
                        outputs = model_n(self.model, x + d_adv)
                    elif mode=="grad_align":
                        self.model.eval()
                        # Get the FGSM Attack
                        d_adv = find_adv_input_fw(
                            self.model,
                            x,
                            y,
                            epsilon,
                            1,
                            c,
                            self.device,
                            adv_norm,
                            self.loss_fn,
                            initialize="zero"
                        )
                        ga_reg = grad_align(self.model, x, y, epsilon, grad_align_lambda)                        
                        
                        self.model.train()
                        outputs = model_n(self.model, x + d_adv)
                    
                    else:
                        raise NameError("Invalid mode")

                    loss = self.loss_fn(outputs, y)
                    if mode == "grad_align":
                        loss = loss + ga_reg
                    xentr_loss = loss.detach().item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                # Free AT Loop
                else:
                    # Mini batch replay loop
                     for _ in range(K):
                        # Warm start perturbation
                        noise_batch = Variable(free_delta[0:x.shape[0]], requires_grad=True).cuda()
                        in1 = x + noise_batch
                        in1 = in1.clamp(0.0, 1.0)

                        # Forward pass
                        outputs = model_n(self.model, in1)
                        loss = self.loss_fn(outputs, y)
                        xentr_loss = loss.detach().item()

                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # FGSM step
                        pert = epsilon * noise_batch.grad.sign()
                        #pert = pert.clamp(-epsilon, epsilon)

                        free_delta[0:x.shape[0]] = free_delta[0:x.shape[0]] + pert.data
                        free_delta = free_delta.clamp(-epsilon, epsilon)
                        
                        # Update weights
                        optimizer.step()

                        d_adv = pert.clone().detach()

                loss_train += loss.item()

                # Compute distortion                
                distortion = self.l2_distortion(d_adv, epsilon)
                distortion_log = distortion.mean().item()           

                descr = f"<EPOCH {epoch}> Steps: {attack_steps}  |  XLoss: {round(xentr_loss, 4)}  | Dist: {round(distortion_log, 4)}"
                train_iterator.set_description(descr)
                
                if not epoch_checked and mode=="fw_adapt":
                    epoch_distortion_check += distortion.sum().item()

                    # First few batches you check the distortion to see
                    # if you can drop the step size
                    if num_images >= 1e3:
                        epoch_checked = True

                        epoch_distortion_check = epoch_distortion_check / num_images

                        #if mode =="fw_adapt":
                        if epoch_distortion_check >= min_distortion_ratio:
                            multi_train_steps = max(1, multi_train_steps // 2)
                            attack_steps = multi_train_steps
                        else:
                            multi_train_steps = min(15, multi_train_steps + multi_step)
                            attack_steps = multi_train_steps
                        
                if batch % log_every_n_batches == 0:
                    train_res = {
                        'batch': batch,
                        'epoch': epoch,
                        'time': time.time() - t0,
                        'loss': xentr_loss,
                        'distortion': distortion_log,
                        'attack_steps': attack_steps
                    }

                    save_mode = "a"
                    if epoch == 0 and batch == 0:
                        save_mode = "w"
                    self.save_log(train_res, exp_path, mode=save_mode, stage='train')

                batch +=1

            print(
                "{} Epoch {}, Train loss {}".format(
                    datetime.datetime.now(), epoch + 1, loss_train / len(train_loader)
                )
            )
            scheduler.step()

            # Eval Loop
            self.did_eval=False
            if (epoch == 0) or ((epoch + 1) % eval_freq == 0):
                clean_acc_eval = self.evaluate(
                                        val_loader,
                                        adv=False
                                    )["acc"]
                eval_res = self.evaluate(
                                    val_loader,
                                    adv=True,
                                    epsilon=epsilon,
                                    mode=eval_mode
                                    )

                acc_eval = eval_res["acc"]
                dist_eval = eval_res["distortion"]
                
                # Set eval distortion for FW-Adapt-Multi
                self.eval_dist = dist_eval

                tag = f"epoch_{epoch}_adv_acc_{100*acc_eval:0.2f}_nat_acc_{100*clean_acc_eval:0.2f}"
                cpath = self.save_model(self.checkpoint_dir, tag)

                eval_results = {
                    "epoch": epoch,
                    "nat_acc": clean_acc_eval,
                    "adv_acc": acc_eval,
                    "dist": dist_eval
                }

                # Let multi know an eval occured
                self.did_eval = True

                if acc_eval_best < acc_eval:
                    print("Robust accuracy improved to: %.2f" % (acc_eval * 100.0))
                    acc_eval_best = acc_eval

                    tag = "_best"
                    cpath = self.save_model(self.checkpoint_dir, tag)

                save_mode = "a" if epoch > 0 else "w"
                self.save_log(eval_results, exp_path, mode=save_mode)

        print("Training complete, model saved:", cpath)
        return exp_path

    def evaluate(
        self,
        val_loader,
        adv=False,
        epsilon=8 / 255.0,
        adv_norm="Linf",
        K=10,
        verbose=True,
        mode="fw",
    ):
        """
        Evaluate the model
        """
        self.model.eval()
        correct = 0
        total = 0
        loss_eval = 0.0

        epoch_dist = 0.0
        num_examples = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                # construct adversarial attacks
                if adv:
                    with torch.enable_grad():
                        if mode == "fw":
                            d_att = find_adv_input_fw(
                                self.model,
                                x,
                                y,
                                epsilon=epsilon,
                                K=K,
                                c=2.0,
                                device=self.device,
                                adv_norm=adv_norm,
                                loss_fn=self.loss_fn,
                            )
                        else:
                            d_att = find_adv_input_pgd(
                                self.model,
                                x,
                                y,
                                epsilon=epsilon,
                                K=K,
                                alpha=2.5 * epsilon / float(K),
                                device=self.device,
                                adv_norm=adv_norm,
                                loss_fn=self.loss_fn,
                            )
                    xadv = x + d_att
                else:
                    xadv = x
                    d_att = xadv - x
                # Compute outputs
                outputs = model_n(self.model, xadv)
                loss = self.loss_fn(outputs, y)
                _, yp = torch.max(outputs, dim=1)
                total += y.shape[0]
                correct += int((yp == y).sum())
                loss_eval += loss.item()

            distortion = self.l2_distortion(d_att, epsilon)
            epoch_dist += distortion.sum().item()
            num_examples += x.shape[0]
        
        epoch_dist = epoch_dist / num_examples
        acc = correct / total
        if verbose:
            eval_type = "Adversarial" if adv else "Natural"
            _msg = f"{eval_type} Eval: \
            Val acc = {acc * 100.0:.3f}  \
            Val loss = {loss_eval / len(val_loader):.5f}  \
            Val dist = {epoch_dist:.4f}\
            "
            print(_msg)
        
        eval_res = {
            "acc": acc,
            "loss": loss_eval / len(val_loader),
            "distortion": epoch_dist
        }
        return eval_res

    def save_model(self, exp_path: Path, save_tag: str):
        fname = f"checkpoint_{save_tag}.pt"
        chkpt_path = exp_path.joinpath(fname)
        torch.save(self.model.state_dict(), chkpt_path)
        return chkpt_path

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def save_log(self, results: dict, exp_path: Path, mode="w", stage="eval"):
        csv_path = exp_path.joinpath(f"{stage}_results.csv")

        save_args = {"mode": mode}
        save_args["header"] = mode == "w"
        save_args["index"] = False

        pd.DataFrame(results, index=[0]).to_csv(csv_path, **save_args)


    def forward(self, x, inplace=True):
        return model_n(self.model, x, inplace=inplace)

# ===========================================================
# Helper functions

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

def find_adv_input_fw(
    model, x, y, epsilon, K, c, device, adv_norm="Linf", loss_fn=nn.CrossEntropyLoss(),initialize="zero", stage="train"
):
    """Compute FW input adversarial perturbation
    x = input tensor
    y = label tensor
    epsilon = perturbation size
    K = number of steps
    c = step size decay parameter
    adv_norm = L2 / Linf
    """
    model.eval()
    assert initialize in ["zero", "random"]
    assert stage in ["train", "eval"]

    if initialize=="zero":
        delta = torch.zeros_like(x, requires_grad=True).to(device=device)
    else:
        delta = torch.rand_like(x, requires_grad=True).to(device=device)
        delta = 2 * epsilon * (delta - 0.5)

    for k in range(K):
        # forward pass
        output = model_n(model, x + delta)
        cost = loss_fn(output, y)

        # backward pass
        cost.backward()

        # step size
        gamma = c / (c + k)

        if adv_norm == "L2":
            delta.data = (
                1 - gamma
            ) * delta.data + gamma * epsilon * delta.grad.detach() / norms(
                delta.grad.detach()
            )
        else:  # adv_norm='Linf'
            delta.data = (
                1 - gamma
            ) * delta.data + gamma * epsilon * delta.grad.detach().sign()
        delta.grad.zero_()
    return delta.detach()

def find_adv_input_pgd(
    model,
    x,
    y,
    epsilon,
    K,
    alpha,
    device,
    adv_norm="Linf",
    loss_fn=nn.CrossEntropyLoss(),
    initialize="zero"
):
    """Compute PGD input adversarial perturbation
    x = input tensor
    y = label tensor
    epsilon = perturbation size
    K = number of steps
    alpha = step size
    adv_norm = L2 / Linf
    """
    model.eval()

    assert initialize in ["zero", "random"]

    if initialize=="zero":
        delta = torch.zeros_like(x, requires_grad=True).to(device=device)
    else:
        delta = torch.rand_like(x, requires_grad=True).to(device=device)
        delta.data = 2 * epsilon * (delta - 0.5)

    for _ in range(K):
        # forward pass
        output = model_n(model, x + delta)
        cost = loss_fn(output, y)

        # backward pass
        cost.backward()

        if adv_norm == "L2":
            delta.data += alpha * delta.grad.detach() / norms(delta.grad.detach())
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        else:  # adv_norm='Linf'
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(
                -epsilon, epsilon
            )
        delta.grad.zero_()
    return delta.detach()


def model_n(model, x, inplace=True)->torch.Tensor:
    """ Input normalization and model evaluation """
    xn = normalize(x, mean, std, inplace=inplace)
    outputs = model(xn)
    return outputs


def normalize(tensor, mean, std, inplace=True):
    if inplace:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    else:
        tensor2 = tensor.clone()
        for t, m, s in zip(tensor2, mean, std):
            t.sub_(m).div_(s)
        return tensor2


def inv_normalize(tensor, mean, std, inplace=True):
    #for t, m, s in zip(tensor, mean, std):
    for i in enumerate(zip(tensor, mean, std)):
        t, m, s = tensor[i], mean[i], std[i]
        if inplace:
            t.mul_(s).add_(m)
        else:
            t = torch.mul(t, s)
            t = torch.add(m)
            tensor[i] = t
    return tensor


def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


import torch.nn.functional as F

# GradAlign Code modified from:
# https://github.com/tml-epfl/understanding-fast-adv-training

def get_uniform_delta(shape, eps, requires_grad=True)->torch.Tensor:
    delta = torch.zeros(shape).cuda()
    delta.uniform_(-eps, eps)
    delta.requires_grad = requires_grad
    return delta

def get_input_grad(model:torch.nn.Module, X:torch.Tensor, y:torch.Tensor,  eps:float, delta_init='none', backprop=False)->torch.Tensor:
    if delta_init == 'none':
        delta = torch.zeros_like(X, requires_grad=True)
    elif delta_init == 'random_uniform':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
    elif delta_init == 'random_corner':
        delta = get_uniform_delta(X.shape, eps, requires_grad=True)
        delta = eps * torch.sign(delta)
    else:
        raise ValueError('wrong delta init')

    output = model_n(model, X + delta)
    loss = F.cross_entropy(output, y)
   
    grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
    if not backprop:
        grad, delta = grad.detach(), delta.detach()
    return grad

def grad_align(model, X, y, eps, grad_align_lambda)->torch.Tensor:
    grad1 = get_input_grad(model, X, y, eps, delta_init='none', backprop=False)
    grad2 = get_input_grad(model, X, y, eps, delta_init='random_uniform', backprop=True)
    grad1, grad2 = grad1.reshape(len(grad1), -1), grad2.reshape(len(grad2), -1)
    cos = F.cosine_similarity(grad1, grad2, 1)
    reg = grad_align_lambda * (1.0 - cos.mean())
    return reg

