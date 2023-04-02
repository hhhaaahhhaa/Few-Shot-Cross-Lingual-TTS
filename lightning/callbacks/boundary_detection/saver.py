import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only
from typing import List

import Define
from ..base_saver import BaseSaver
from ..t2u.plot import AttentionVisualizer


CSV_COLUMNS = ["Total Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


def set_format(keys: List[str]):
    global CSV_COLUMNS, COL_SPACE
    CSV_COLUMNS = keys
    COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]


class Saver(BaseSaver):
    def __init__(self, data_configs, model_config, log_dir, result_dir):
        super().__init__(log_dir, result_dir)
        self.visualizer = AttentionVisualizer()
        self.data_configs = data_configs
        
        self.model_config = model_config
        
        self.val_loss_dicts = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']

        step = pl_module.global_step + 1
        if isinstance(pl_module.logger, list):
            assert len(list(pl_module.logger)) == 1
            logger = pl_module.logger[0]
        else:
            logger = pl_module.logger

        # Log message to log.txt and print to stdout
        if step % trainer.log_every_n_steps == 0 and pl_module.local_rank == 0:
            loss_dict = {k: v.item() for k, v in loss.items()}
            set_format(list(loss_dict.keys()))
            
            loss_dict.update({"Step": step, "Stage": "Training"})
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if len(self.log_loss_dicts)==0:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            else:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
            self.log_loss_dicts.append(loss_dict)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 1:  # unlabeled data, do nothing
            return
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']
        
        step = pl_module.global_step + 1
        if isinstance(pl_module.logger, list):
            assert len(list(pl_module.logger)) == 1
            logger = pl_module.logger[0]
        else:
            logger = pl_module.logger

        loss_dict = {k: v.item() for k, v in loss.items()}
        set_format(list(loss_dict.keys()))
        self.val_loss_dicts.append(loss_dict)

        # Log loss for each sample to csv files
        self.save_csv("Validation", step, 0, loss_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_dict = merge_dicts(self.val_loss_dicts)
        step = pl_module.global_step + 1

        # Log total loss to log.txt and print to stdout
        loss_dict.update({"Step": step, "Stage": "Validation"})
        # To stdout
        df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
        if len(self.log_loss_dicts)==0:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
        else:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
        # To file
        self.log_loss_dicts.append(loss_dict)
        log_file_path = os.path.join(self.log_dir, 'log.txt')
        df = pd.DataFrame(self.log_loss_dicts, columns=["Step", "Stage"]+CSV_COLUMNS).set_index("Step")
        df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=True)
        
        # Reset
        self.log_loss_dicts = []

    # New logging functions
    def plot_tensor(self, x, title, x_labels: List[str], y_labels: List[str]):
        "Wrapper method for calling visualizer to plot."
        info = {
            "title": title,
            "x_labels": x_labels,
            "y_labels": y_labels,
            "attn": x.detach().cpu().numpy()
        }
        fig = self.visualizer.plot(info)
        return fig
    
    def log_1D_tensor(self, logger, x, step, title, stage="val"):
        """ Visualize any 1D tensors """
        fig = self.plot_tensor(x.view(1, -1), title, [str(i) for i in range(len(x))], ["Weight"])
        figure_name = f"{stage}/{title}/step_{step}"
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=step,
            )
        plt.close(fig)

    def log_2D_tensor(self, logger, x, step, title, x_labels: List[str], y_labels: List[str], stage="val"):
        """ Visualize any 2D tensors """
        fig = self.plot_tensor(x, title, x_labels, y_labels)
        figure_name = f"{stage}/{title}/step_{step}"
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=step,
            )
        plt.close(fig)
        
    def log_attn(self, logger, attn, batch_idx, step, title, x_labels: List[str], y_labels: List[str], stage="val"):
        """
        Visualize 2D attention for all heads for a sample.
        attn: Tensor with size nH, len(y_labels), len(x_labels) (Be careful need to be opposite)
        """
        nH, ly, lx = attn.shape
        assert lx == len(x_labels)
        assert ly == len(y_labels)

        for hid in range(nH):
            fig = self.plot_tensor(attn[hid], f"Head-{hid}", x_labels, y_labels)
            figure_name = f"{stage}/{title}/step_{step}_{batch_idx:03d}_h{hid}"
            if isinstance(logger, pl.loggers.CometLogger):
                logger.experiment.log_figure(
                    figure_name=figure_name,
                    figure=fig,
                    step=step,
                )
            plt.close(fig)
    
    def log_layer_weights(self, logger, layer_weights, step, stage="val"):
        self.log_1D_tensor(logger, layer_weights, step, "Layer weights", stage=stage)
