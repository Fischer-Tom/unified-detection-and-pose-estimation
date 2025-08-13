import torch
from torch import nn
from itertools import chain
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR




def construct_optimizer(model: torch.nn.Module, cfg: DictConfig):
    def _use_grad_clip(optim, cfg):
        class ClippedOptimizer(optim):
            def step(self, closure=None):
                all_params = chain(
                    *[x["params"] for x in self.param_groups]
                )
                torch.nn.utils.clip_grad_norm_(all_params, cfg.clip_grad_norm)
                super().step(closure=closure)

        return ClippedOptimizer if cfg.clip_grad_norm > 0 else optim

    params = split_parameters(model, cfg.weight_decay)


    if cfg.name == "AdamW":
        optimizer = _use_grad_clip(torch.optim.AdamW, cfg)(
           params, lr=cfg.lr,
        )
    elif cfg.name == "Adam":
        optimizer = _use_grad_clip(torch.optim.Adam, cfg)(
            params, lr=cfg.lr
        )
    elif cfg.name == "SGD":
        optimizer = _use_grad_clip(torch.optim.SGD, cfg)(
            params, lr=cfg.lr
        )
    else:
        raise NotImplementedError("Optimizer not implemented")

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: step / cfg.warmup_iters)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.train_iters - cfg.warmup_iters, eta_min=cfg.eta_min
    )

    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_iters]
    )

    return optimizer, lr_scheduler

def split_parameters(model: torch.nn.Module, wd: float):
    decay = set()
    no_decay = set()
    mesh_type = type(model.module.neural_mesh) if hasattr(model, 'module') else type(model.neural_mesh)
    whitelist_weight_modules = (nn.Conv2d, nn.Linear, nn.MultiheadAttention, nn.Parameter, mesh_type)
    blacklist_weight_modules = (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn
            if not p.requires_grad:
                # skip frozen parameters
                continue
            elif pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
    assert len(
        param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params),)
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": wd},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups