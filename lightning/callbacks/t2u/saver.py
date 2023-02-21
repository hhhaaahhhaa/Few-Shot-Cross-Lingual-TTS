from typing import Dict, List, Tuple
import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib

from lightning.build import build_id2symbols
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.base import merge_dicts
from pytorch_lightning.utilities import rank_zero_only

from text.define import LANG_ID2NAME
from .plot import AttentionVisualizer


CSV_COLUMNS = ["Total Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


def set_format(keys: List[str]):
    global CSV_COLUMNS, COL_SPACE
    CSV_COLUMNS = keys
    COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]


class Saver(Callback):

    def __init__(self, data_configs, log_dir, result_dir, re_id=True):
        super().__init__()
        self.visualizer = AttentionVisualizer()
        self.data_configs = data_configs
        self.id2symbols = build_id2symbols(self.data_configs)
        self.re_id = re_id
        increment = 0
        self.re_id_increment = {}
        for k, v in self.id2symbols.items():
            self.re_id_increment[k] = increment
            increment += len(v)
        # print(self.re_id_increment)

        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        print("Result directory:", self.result_dir)

        self.val_loss_dicts = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        symbol_id = outputs['symbol_id']
        alignment = outputs['alignment']

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

            # log asr results
            sentence = _batch[6][0]
            gt_sentence, pred_sentence = self.recover_sentences(sentence, output[0].argmax(dim=1), symbol_id=symbol_id)
            
            self.log_text(logger, "Train/GT: " + ", ".join(gt_sentence), step)
            self.log_text(logger, "Train/Pred: " + ", ".join(pred_sentence), step)
            self.log_alignment(logger, alignment, _batch, step, "train")

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']  # batch or qry_batch
        symbol_id = outputs['symbol_id']
        alignment = outputs['alignment']
        
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
        self.log_csv("Validation", step, 0, loss_dict)

        # Log asr results to logger + calculate acc
        if batch_idx == 0 and pl_module.local_rank == 0:

            # log asr results
            sentence = _batch[6][0]
            gt_sentence, pred_sentence = self.recover_sentences(sentence, output[0].argmax(dim=1), symbol_id=symbol_id)
            
            self.log_text(logger, "Val/GT: " + ", ".join(gt_sentence), step)
            self.log_text(logger, "Val/Pred: " + ", ".join(pred_sentence), step)
            self.log_alignment(logger, alignment, _batch, step, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_dict = merge_dicts(self.val_loss_dicts)
        step = pl_module.global_step+1

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
    
    def log_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_text(self, logger, text, step):
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_text(
                text=text,
                step=step,
            )
    
    def log_alignment(self, logger, alignment, _batch, step, stage="val"):
        for idx, data in enumerate(alignment[:4]):
            lang_id, symbol_id = LANG_ID2NAME[_batch[9][idx].item()], _batch[10][idx]
            # print(lang_id, symbol_id)
            # print(_batch[6][idx])
            # print(_batch[3][idx])
            x_labels, y_labels = [], []
            for id in _batch[6][idx]:
                x_labels.append(self.id2symbols[symbol_id][id])
            for id in _batch[3][idx]:
                if id == 0:  # <PAD>
                    y_labels.append(0)
                else:
                    if self.re_id:
                        y_labels.append(self.id2symbols[lang_id][id - self.re_id_increment[lang_id]])
                    else:
                        y_labels.append(self.id2symbols[lang_id][id])
            info = {
                "title": "Alignment",
                "x_labels": x_labels,
                "y_labels": y_labels,
                "attn": data.transpose(0, 1).detach().cpu().numpy()
            }
            
            fig = self.visualizer.plot(info)
            figure_name = f"{stage}/step_{step}_{idx:03d}"
            if isinstance(logger, pl.loggers.CometLogger):
                logger.experiment.log_figure(
                    figure_name=figure_name,
                    figure=fig,
                    step=step,
                )
            plt.close(fig)

    # For TransEmbCSystem
    def log_codebook_attention(self, logger, attn, symbol_id, batch_idx, step, stage="val"):
        """
        TODO: In fact we are getting framewise results, but not symbols-code attention since we now pass codebook before averaging
        attn: Tensor with size B, nH, n_symbols, codebook_size
        """
        B, nH, n_symbols, codebook_size = attn.shape
        symbols = self.id2symbols[symbol_id]
        for hid in range(nH):
            info = {
                "title": f"Head-{hid}",
                "x_labels": symbols,
                "y_labels": [str(i) for i in range(codebook_size)],
                "attn": attn[0][hid].transpose(0, 1).detach().cpu().numpy()
            }
            
            fig = self.visualizer.plot(info)
            figure_name = f"{stage}/codebook/step_{step}_{batch_idx:03d}"
            if isinstance(logger, pl.loggers.CometLogger):
                logger.experiment.log_figure(
                    figure_name=figure_name,
                    figure=fig,
                    step=step,
                )
            plt.close(fig)
    
    # For TransEmb/TransEmbCSystem
    def log_layer_weights(self, logger, layer_weights, step, stage="val"):
        info = {
            "title": "Layer weights",
            "x_labels": [str(i) for i in range(len(layer_weights))],
            "y_labels": ["Weight"],
            "attn": layer_weights.view(1, -1).detach().cpu().numpy()
        }
        
        fig = self.visualizer.plot(info)
        figure_name = f"{stage}/weights/step_{step}"
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_figure(
                figure_name=figure_name,
                figure=fig,
                step=step,
            )
        plt.close(fig)

    def recover_sentences(self, gt_ids, pred_ids, symbol_id: str) -> Tuple[List[str], List[str]]:
        gt_sentence, pred_sentence = [], []
        for (gt_id, pred_id) in zip(gt_ids, pred_ids):
            if gt_id == 0:
                break
            # t2u only predict one kind of symbol, no re_id is required (unlike SSLBaseline recognized multilingual symbols)
            gt_sentence.append(self.id2symbols[symbol_id][gt_id])
            pred_sentence.append(self.id2symbols[symbol_id][pred_id])

        return gt_sentence, pred_sentence
