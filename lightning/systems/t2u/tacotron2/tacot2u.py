import torch
from torch import nn

from dlhlp_lib.utils.tool import get_mask_from_lengths

from .hparams import model_config2hparams
from .hparams import hparams as hps
from .tacot2u_model import Encoder, Decoder


class TacoT2U(nn.Module):
	def __init__(self, model_config):
		super(TacoT2U, self).__init__()
		model_config2hparams(model_config)
		self.d_out = hps.n_units
		self.encoder = Encoder()
		self.decoder = Decoder()

	# def parse_batch(self, batch):
    #     text_padded, input_lengths, max_text_len, unit_padded, output_lengths, spks, lang_ids = batch

	# 	return (
	# 		(text_padded, input_lengths, unit_padded, max_len, output_lengths, spks, lang_ids),
	# 		(unit_padded, output_lengths))

	def parse_output(self, outputs, output_lengths=None):
		if output_lengths is not None:
			mask = get_mask_from_lengths(output_lengths).to(outputs[0].device) # (B, T)
			mask = mask.expand(self.d_out, mask.size(0), mask.size(1)) # (n_units, B, T)
			mask = mask.permute(1, 2, 0) # (B, T, n_units)
			
			outputs[0].data.masked_fill_(mask, 0.0) # (B, T, n_units)
		return outputs

	def forward(self, inputs):
		emb_text_inputs, text_lengths, units, max_len, output_lengths, spks, lang_ids = inputs
		text_lengths, output_lengths = text_lengths.data, output_lengths.data

		embedded_inputs = emb_text_inputs.transpose(1, 2)  # B, dim, L

		encoder_outputs = self.encoder(embedded_inputs, text_lengths)

		logits, alignments = self.decoder(
			encoder_outputs, units, memory_lengths=text_lengths)  # logits shape is [B, T, n_units]

		return self.parse_output([logits, alignments], output_lengths)

	def inference(self, embedded_inputs, spks, lang_ids):
		encoder_outputs = self.encoder.inference(embedded_inputs.transpose(1, 2))

		hidden_outputs, alignments = self.decoder.inference(encoder_outputs)
		outputs = self.parse_output([hidden_outputs, alignments])

		return outputs

	# def teacher_infer(self, inputs, mels, lang_ids):
	# 	il, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]),
	# 						dim = 0, descending = True)
	# 	text_lengths = il.to(Define.DEVICE)

	# 	embedded_inputs = self.embedding(inputs, lang_ids[0]).transpose(1, 2)

	# 	encoder_outputs = self.encoder(embedded_inputs, text_lengths)

	# 	mel_outputs, gate_outputs, alignments = self.decoder(
	# 		encoder_outputs, mels, memory_lengths=text_lengths)
		
	# 	mel_outputs_postnet = self.postnet(mel_outputs)
	# 	mel_outputs_postnet = mel_outputs + mel_outputs_postnet

	# 	return self.parse_output(
	# 		[mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
