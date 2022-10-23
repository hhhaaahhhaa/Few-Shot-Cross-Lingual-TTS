import torch
import numpy as np


def get_optimizer(model, model_config, train_config):
    init_lr = np.power(256, -0.5)  # TODO: do not depent on model config

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=init_lr,
        betas=train_config["optimizer"]["betas"],
        eps=train_config["optimizer"]["eps"],
        weight_decay=train_config["optimizer"]["weight_decay"],
    )
    return optimizer
