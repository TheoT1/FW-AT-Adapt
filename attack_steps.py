import torch
from torch.nn import Module, CrossEntropyLoss

from torch.autograd import grad


class Attacker:
    """
    Base class for Linf adversarial attacks
    """

    def __init__(self, epsilon: float, num_steps: int, rand_init:bool=False):
        super().__init__()
        self.eps = epsilon
        self.N = num_steps
        self.loss = CrossEntropyLoss()
        self.rand_init =rand_init

    def get_adv_perturbation(
        self, model: Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        pass

    def update_parameters(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def random_delta(self, x):
        """
        """
        delta = 2 * (torch.rand_like(x, requires_grad=True) - 0.5) * self.eps
        return delta

class FWLinfAttacker(Attacker):
    """
    FW approximation of an Linf adversarial attack.
        g[k] = grad_d [l(x + d[k])]
        s[k] = argmax < g[k] | s >
             = epsilon sgn( g[k] )

        d[k+1] = a[k] s[k] + (1-a[k])d[k]
    where a[k] = c/(c + k) and 1<=c
    """

    def __init__(self, num_steps: int, epsilon: float = 8 / 255.0, c: float = 2.0, **kwargs):
        super().__init__(epsilon, num_steps, **kwargs)
        self.c = c

    def lmo(self, g):
        """
        The linear maximization oracle for the Linf
        problem.
        """
        return self.eps * g.sign()

    def get_adv_perturbation(
        self, model: Module, x: torch.Tensor, y: torch.Tensor, return_grads:bool=False
    ) -> torch.Tensor:

        delta = torch.zeros_like(x, requires_grad=True)
        if self.rand_init:
            delta = self.random_delta(x)
        grads = []
        for k in range(self.N):
            gamma = self.c / (self.c + k)

            output = model(x + delta)
            loss = self.loss(output, y)

            g = grad(loss, delta)[0]
            if return_grads:
                grads.append(g.detach())
            s = self.lmo(g)

            delta = gamma * s + (1 - gamma) * delta
        if return_grads:
            return delta, grads
        return delta.detach()
    
    def update_parameters(self, num_steps:int=1)->Attacker:
        self.N = num_steps
        return self


class PGDLinfAttacker(Attacker):
    """
    PGD approximation of an Linf adversarial attack.
        g[k] = grad_d [l(x + d[k])]

        d[k+1] = Clamp(d[k] + step_size * sgn(g[k]) ; eps)
    Where Clamp(x; eps) = sgn(x) min(|x| , eps) and
        step_size = alpha * eps / N
    for some alpha.
    """

    def __init__(self, num_steps: int, epsilon: float = 8 / 255.0, alpha: float = 2.5, **kwargs):
        super().__init__(epsilon, num_steps, **kwargs)
        self.alpha = alpha

    def proj(self, v):
        """
        Project onto the Linf ball of radius epsilon.
        """
        return v.clamp(-self.eps, self.eps)

    def get_adv_perturbation(
        self, model: Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        step_size = self.eps * self.alpha / self.N

        delta = torch.zeros_like(x, requires_grad=True)
        if self.rand_init:
            delta = self.random_delta(x)

        for _ in range(self.N):

            output = model(x + delta)
            loss = self.loss(output, y)

            g = grad(loss, delta)[0].sign()

            delta = delta + step_size * g
            delta = self.proj(delta)

        return delta.detach()
    
    def update_parameters(self, num_steps:int=1)->Attacker:
        self.N = num_steps
        return self


def unsqueeze_if(v, num_dims=4):
    N = len(v.shape)
    if N == num_dims:
        return v
    elif N == num_dims - 1:
        return v.unsqueeze(0)
    else:
        print("Shape of v weird")
        return v


class FGSMAdaptAttacker(Attacker):
    def __init__(self, epsilon: float, adapt_steps: int, alpha:float=None):
        super().__init__(epsilon, 1)
        self.adapt_steps = adapt_steps
        if alpha is not None:
            print("Currently alpha is not used")
        self.alpha = alpha
        
        # Not quite what they did since there is no alpha step.
        # However the step size was 1.25 eps whereas we use
        # eps. Lets see if we get anything at least then... TODO: Add step size in
        self.fast_attacker = FWLinfAttacker(
                                num_steps=1,
                                epsilon=epsilon,
                                rand_init=True
                            )
    
    def get_adv_perturbation(self, model: Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = len(x)
        Nc = self.adapt_steps
        C = torch.Tensor([(i+1)/Nc for i in range(Nc)]).to(device)
        
        
        delta = self.fast_attacker.get_adv_perturbation(model, x, y)
        # Get correctly classified indexes.
        logit_clean = model(x)
        preds_clean = torch.argmax(logit_clean, dim=1)
        correct = (preds_clean == y)

        correct_idx = torch.masked_select(torch.arange(batch_size).to(device), correct)
        
        # Mistakes are passed through so only worry about correct images
        delta_chk = delta[correct_idx]
        y_chk = y[correct_idx]
        X_chk = x[correct_idx]

        idxs = correct_idx
        for j in range(Nc):
            delta_j = delta_chk * C[j]
            X_adv = X_chk + delta_j.detach()
            yj = model(X_adv)
            
            yj = torch.argmax(yj, dim=1)

            cor_idx = (yj == y_chk)
            # Update deltas which were flipped
            update_idxs = idxs[~cor_idx]
            delta[update_idxs] = delta_j[~cor_idx]

            # Reduce the tensors we're checking
            delta_chk = delta_chk[cor_idx]
            X_chk = unsqueeze_if(X_chk[cor_idx], num_dims=4)
            y_chk = unsqueeze_if(y_chk[cor_idx], num_dims=1)

            idxs = idxs[cor_idx]
            if len(idxs) == 0:
                break
        
        return delta
