import torch
from torch import nn

from dlhlp_lib.utils.tool import get_mask_from_lengths

from .hparams import hparams as hps


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction = 'none')

    def forward(self, model_outputs, targets):
        mel_out, mel_out_postnet, gate_out, _ = model_outputs
        gate_out = gate_out.view(-1, 1)

        mel_target, gate_target, output_lengths = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        output_lengths.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)
        mel_mask = get_mask_from_lengths(output_lengths.data).to(model_outputs.device)

        mel_loss = self.loss(mel_out, mel_target) + \
            self.loss(mel_out_postnet, mel_target)
        mel_loss = mel_loss.sum(1).masked_fill_(mel_mask, 0.)/mel_loss.size(1)
        mel_loss = mel_loss.sum()/output_lengths.sum()

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss+gate_loss, (mel_loss.item(), gate_loss.item())