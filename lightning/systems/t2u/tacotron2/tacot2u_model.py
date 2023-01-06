import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import random

from dlhlp_lib.utils.tool import get_mask_from_lengths

from .hparams import hparams as hps
from .layers import ConvNorm, LinearNorm


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        '''
        PARAMS
        ------
        query: decoder output (batch, num_mels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)
        RETURNS
        -------
        alignment (batch, max_time)
        '''

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        '''
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        '''
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Encoder(nn.Module):
    '''Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    '''
    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hps.symbols_embedding_dim if i == 0 \
                            else hps.encoder_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size, stride=1,
                         padding=int((hps.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hps.encoder_embedding_dim,
                            int(hps.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.teacher_forcing_ratio = 1.0
        self.prenet = Prenet(
            hps.d_unit,
            [hps.prenet_dim, hps.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hps.prenet_dim + hps.encoder_embedding_dim,
            hps.attention_rnn_dim)

        self.attention_layer = Attention(
            hps.attention_rnn_dim, hps.encoder_embedding_dim,
            hps.attention_dim, hps.attention_location_n_filters,
            hps.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hps.attention_rnn_dim + hps.encoder_embedding_dim,
            hps.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            hps.encoder_embedding_dim)
        
        self.unit_embedding = nn.Embedding(hps.n_units, hps.d_unit)
        self.final_proj = nn.Linear(hps.encoder_embedding_dim, hps.n_units)

    def get_go_frame(self, memory):
        ''' Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs
        RETURNS
        -------
        decoder_input: all zeros frames
        '''
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, hps.d_unit).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        ''' Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        '''
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, hps.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, hps.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, hps.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, hps.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, hps.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        ''' Prepares decoder inputs, i.e. unit embeddings
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. unit embeddings
        RETURNS
        -------
        inputs: processed decoder inputs
        '''
        # (B, T_out, d_unit) -> (T_out, B, d_unit)
        return decoder_inputs.transpose(0, 1)

    def parse_decoder_outputs(self, hidden_outputs, alignments):
        ''' Prepares decoder outputs for output '''
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
       
        # (T_out, B, n_units) -> (B, T_out, n_units)
        hidden_outputs = torch.stack(hidden_outputs).transpose(0, 1).contiguous()
        return hidden_outputs, alignments

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: prenet(previous unit embedding)
        RETURNS
        -------
        decoder_output: logits outputs
        attention_weights:
        '''
        cell_input = torch.cat((decoder_input, self.attention_context), -1)  # B, prenet_dim + encoder_dim
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, hps.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, hps.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)  # B, encoder_dim

        decoder_output = self.final_proj(decoder_output)  # B, n_unit

        return decoder_output, self.attention_weights

    def forward(self, memory, units, memory_lengths):
        ''' Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        units: Decoder inputs for teacher forcing. i.e. units.
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        hidden_outputs: logits outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.unit_embedding(units)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=get_mask_from_lengths(memory_lengths).to(memory.device))

        hidden_outputs, alignments = [], []
        while len(hidden_outputs) < decoder_inputs.size(0) - 1:
            if random.uniform(0, 1) < self.teacher_forcing_ratio or len(hidden_outputs) == 0:
                decoder_input = decoder_inputs[len(hidden_outputs)]
                # print(decoder_input.shape)
            else:
                prediction = torch.argmax(hidden_outputs[-1], dim=1).detach()
                decoder_input = self.prenet(self.unit_embedding(prediction))
                # print(decoder_input.shape)
            hidden_output, attention_weights = self.decode(decoder_input)
            hidden_outputs += [hidden_output.squeeze(1)]
            alignments += [attention_weights]
        hidden_outputs, alignments = self.parse_decoder_outputs(
            hidden_outputs, alignments)
        return hidden_outputs, alignments

    def inference(self, memory):  # currently does not support batch inference
        ''' Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        hidden_outputs: logits outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        hidden_outputs, alignments = [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            hidden_output, alignment = self.decode(decoder_input)
            prediction = torch.argmax(hidden_output.squeeze(1), dim=1)

            if prediction[0].item() == 8:
                # print('Terminated by <eos>.')
                break
            elif len(hidden_outputs) / alignment.shape[1] >= hps.max_decoder_ratio:
                print('Warning: Reached max decoder steps.')
                break
            
            hidden_outputs += [hidden_output.squeeze(1)]
            alignments += [alignment]
            
            decoder_input = self.unit_embedding(prediction)

        hidden_outputs, alignments = self.parse_decoder_outputs(hidden_outputs, alignments)
        return hidden_outputs, alignments