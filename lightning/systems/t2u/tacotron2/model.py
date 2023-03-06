import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

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


class Postnet(nn.Module):
    '''Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    '''

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hps.postnet_embedding_dim))
        )

        for i in range(1, hps.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hps.postnet_embedding_dim,
                             hps.postnet_embedding_dim,
                             kernel_size=hps.postnet_kernel_size, stride=1,
                             padding=int((hps.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hps.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.postnet_embedding_dim, hps.num_mels,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hps.num_mels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
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
        self.prenet = Prenet(
            hps.num_mels * hps.n_frames_per_step,
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
            hps.num_mels * hps.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

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
        decoder_input = Variable(memory.data.new(
            B, hps.num_mels * hps.n_frames_per_step).zero_())
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
        ''' Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        RETURNS
        -------
        inputs: processed decoder inputs
        '''
        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/hps.n_frames_per_step), -1)
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        ''' Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:
        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        '''
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, hps.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        ''' Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output
        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        '''
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
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
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        ''' Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=get_mask_from_lengths(memory_lengths).to(memory.device))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        ''' Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        '''
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > hps.gate_threshold:
                print('Terminated by gate.')
                break
            elif hps.n_frames_per_step*len(mel_outputs)/alignment.shape[1] \
                    >= hps.max_decoder_ratio:
                print('Warning: Reached max decoder steps.')
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments