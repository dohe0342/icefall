#!/usr/bin/env python3
# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
#                2022  Xiaomi Corp.                              (author: Quandong Wang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from combiner import RandomCombine
from scaling import (
    ActivationBalancer,
    BasicNorm,
    DoubleSwish,
    ScaledConv1d,
    ScaledLinear,
)
from subsampling import Conv2dSubsampling
from transformer import Supervisions, Transformer, encoder_padding_mask
from transformers import (
    GPT2Tokenizer, 
    GPT2Model, 
    BertTokenizer, 
    BertModel, 
    AutoTokenizer, 
    MistralModel,
)
from fairseq.modules import (
    TransposeLast,
    GumbelVectorQuantizer,
)


class Conformer(Transformer):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        subsampling_factor: int = 4,
        d_model: int = 256,
        nhead: int = 4,
        dim_feedforward: int = 2048,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 15,
        aux_layer_period: int = 3,
        group_num: int = 0,
        interctc: bool = False,
        interctc_condition: bool = False,
        learnable_alpha: bool = True,
        distill:bool = False,
        lm_name:str = 'None', 
        quant: bool = False,
    ) -> None:
        """
        Args:
          num_features (int):
            number of input features.
          num_classes (int):
            number of output classes.
          subsampling_factor (int):
            subsampling factor of encoder;
            currently, subsampling_factor MUST be 4.
          d_model (int):
            attention dimension, also the output dimension.
          nhead (int):
            number of heads in multi-head attention;
            must satisfy d_model // nhead == 0.
          dim_feedforward (int):
            feedforward dimention.
          num_encoder_layers (int):
            number of encoder layers.
          num_decoder_layers (int):
            number of decoder layers.
          dropout (float):
            dropout rate.
          layer_dropout (float):
            layer-dropout rate.
          cnn_module_kernel (int):
            kernel size of convolution module.
          aux_layer_period (int):
            determines the auxiliary encoder layers.
        """

        super().__init__(
            num_features=num_features,
            num_classes=num_classes,
            subsampling_factor=subsampling_factor,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            layer_dropout=layer_dropout,
        )

        self.num_features = num_features
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        # self.encoder_embed converts the input of shape (N, T, num_features)
        # to the shape (N, T//subsampling_factor, d_model).
        # That is, it does two things simultaneously:
        #   (1) subsampling: T -> T//subsampling_factor
        #   (2) embedding: num_features -> d_model
        self.encoder_embed = Conv2dSubsampling(num_features, d_model)

        self.encoder_pos = RelPositionalEncoding(d_model, dropout)

        if quant:
            self.quant = GumbelVectorQuantizer(dim=d_model, 
                                               num_vars=200, 
                                               temp=(2, 0.5, 0.999995), 
                                               groups=2, 
                                               combine_groups=False, 
                                               vq_dim=256, 
                                               time_first=True,)
        else:
            self.quant = None

        encoder_layer = ConformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_dropout=layer_dropout,
            cnn_module_kernel=cnn_module_kernel,
        )

        # aux_layers from 1/3
        self.encoder = ConformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            aux_layers=list(
                range(
                    num_encoder_layers // 3,
                    num_encoder_layers - 1,
                    aux_layer_period,
                )
            ),
        )

        self.group_num = group_num
        if self.group_num != 0:
            self.learnable_alpha = learnable_alpha
            self.group_layer_num = int(num_encoder_layers // self.group_num)
            if self.learnable_alpha:
                self.alpha = nn.Parameter(torch.rand(self.group_num))
                self.sigmoid = nn.Sigmoid()
            self.layer_norm = nn.LayerNorm(d_model)

        self.interctc = interctc
        self.interctc_condition = interctc_condition
        if self.interctc_condition:
            self.condition_layer = ScaledLinear(500, d_model)
        else:
            self.condition_layer = None

        self.distill = distill
        if self.distill:
            ########### for gpt2
            if 'bert' in lm_name:
                self.tokenizer = BertTokenizer.from_pretrained(lm_name)
                #self.tokenizer.pad_token = self.tokenizer.eos_token
                self.lm = BertModel.from_pretrained(lm_name)

            if 'gpt2' in lm_name:
                self.tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.lm = GPT2Model.from_pretrained(lm_name)
            
            if 'mistral' in lm_name:
                self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.lm = MistralModel.from_pretrained(lm_name, torch_dtype=torch.float16)

            if 'phi-2' in lm_name:
                from transformers import PhiModel
                self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.lm = PhiModel.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)

            self.lm_decoder = nn.ModuleList()
            conv_layers = [(d_model, 5, 2)] * 2
            for conv in conv_layers:
                d, k, s = conv
                self.lm_decoder.append(ScaledConv1d(d, d, k, s, bias=False))
                self.lm_decoder.append(nn.Sequential(
                              TransposeLast(),
                              nn.LayerNorm(d, elementwise_affine=True),
                              TransposeLast(),
                              )) 
                self.lm_decoder.append(nn.GELU())
            #self.lm_decoder.append(ScaledLinear(d, 768, bias=False))
            #self.lm_decoder.append(nn.Linear(d_model, self.lm.embed_dim, bias=False))
            if 'gpt2' in lm_name:
                #self.lm_decoder.append(nn.Linear(self.lm.embed_dim, 256, bias=False))
                self.lm_decoder.append(nn.Linear(self.lm.embed_dim, d_model, bias=False))
                #self.lm_decoder.append(nn.Linear(d_model, self.lm.embed_dim, bias=False))
            else:
                self.lm_decoder.append(nn.Linear(self.lm.config.hidden_size, d_model, bias=False))

            #if quant:
            #    del self.lm_decoder[-1]

            #self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
            #self.lm = BertModel.from_pretrained("bert-large-uncased-whole-word-masking")
            #self.lm = GPT2Model.from_pretrained('/home/work/workspace/models/checkpoint-420500')
            #self.distill_linear = ScaledLinear(d_model, 768)
            #self.ins_norm = torch.nn.InstanceNorm1d(768)
            ##############################################################

    def run_encoder(
        self,
        x: torch.Tensor,
        supervisions: Optional[Supervisions] = None,
        warmup: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            the input tensor. Its shape is (batch_size, seq_len, feature_dim).
          supervisions:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            CAUTION: It contains length information, i.e., start and number of
            frames, before subsampling
            It is read directly from the batch, without any sorting. It is used
            to compute encoder padding mask, which is used as memory key padding
            mask for the decoder.
          warmup:
            a floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up".  It is used
            to turn modules on sequentially.

        Returns:
          torch.Tensor: Predictor tensor of dimension (S, N, C).
          torch.Tensor: Mask tensor of dimension (N, S)
        """
        x = self.encoder_embed(x)
        x, pos_emb = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (N, S, C) -> (S, N, C)
        mask = encoder_padding_mask(x.size(0), supervisions)
        mask = mask.to(x.device) if mask is not None else None

        x, layer_outputs = self.encoder(
            x, 
            pos_emb, 
            src_key_padding_mask=mask, 
            warmup=warmup, 
            condition_layer=self.condition_layer, 
            ctc_output=self.ctc_output,
        )  # (S, N, C)
        
        if self.group_num > 0:
            x = 0
            if self.learnable_alpha:
                for enum, alpha in enumerate(self.alpha):
                    x += self.sigmoid(alpha) * layer_outputs[(enum+1)*self.group_layer_num-1]
            else:
                for enum in range(self.group_num):
                    x += (1/self.group_num) * layer_outputs[(enum+1)*self.group_layer_num-1]
            x = self.layer_norm(x)

        if self.interctc or self.interctc_condition or self.group_num > 0:
            return (x, layer_outputs), mask
        else:
            return x, mask
        
    def forward(
        self,
        x: torch.Tensor,
        supervision: Optional[Supervisions] = None,
        warmup: float = 1.0,
        texts: list = None,
        vis: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (N, S, C).
          supervision:
            Supervision in lhotse format.
            See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32  # noqa
            (CAUTION: It contains length information, i.e., start and number of
             frames, before subsampling)
          warmup:
            a floating point value that gradually increases from 0 throughout
            training; when it is >= 1.0 we are "fully warmed up". It is used
            to turn modules on sequentially.

        Returns:
          Return a tuple containing 3 tensors:
            - CTC output for ctc decoding. Its shape is (N, S, C)
            - Encoder output with shape (S, N, C). It can be used as key and
              value for the decoder.
            - Encoder output padding mask. It can be used as
              memory_key_padding_mask for the decoder. Its shape is (N, S).
              It is None if `supervision` is None.
        """
        encoder_memory, memory_key_padding_mask = self.run_encoder(
            x, supervision, warmup
        )
        
        if type(encoder_memory) == tuple:
            (encoder_memory, layer_outputs) = encoder_memory
            x = self.ctc_output(encoder_memory)
            layer_outputs = [self.ctc_output(x) for x in layer_outputs]
            return (x, layer_outputs), encoder_memory, memory_key_padding_mask
        
        elif self.distill:
            if texts is None: texts = supervision["text"]
            x = self.ctc_output(encoder_memory)
            ############for distillation###########
            device = encoder_memory.device
            tgt_list = [text.lower() for text in texts]
            lm_input = self.tokenizer(tgt_list, return_tensors='pt', padding=True, return_attention_mask=True).to(device)
            with torch.no_grad():
                lm_output = self.lm(**lm_input)
                lm_output = lm_output['last_hidden_state']
                lm_output = F.normalize(lm_output, dim=2)
            
            am_output = encoder_memory.transpose(0, 1).transpose(1, 2)
            
            '''
            if self.quant is None:
                for layer in self.lm_decoder[:-1]:
                    am_output = layer(am_output)
                am_output = am_output.transpose(1, 2)
                #am_output = self.lm_decoder[-1](am_output)
                lm_output = self.lm_decoder[-1](lm_output)
                am_output = F.normalize(am_output, dim=2)

            else:
                for layer in self.lm_decoder:
                    am_output = layer(am_output)
                am_output = am_output.transpose(1, 2)
            '''
            for layer in self.lm_decoder[:-1]:
                am_output = layer(am_output)
            am_output = am_output.transpose(1, 2)
            #am_output = self.lm_decoder[-1](am_output)
            lm_output = self.lm_decoder[-1](lm_output)
            #am_output = F.normalize(am_output, dim=2)

            if self.quant is not None:
                am_output = self.quant(am_output)
                am_output = am_output['x']
            
            lm_am_sim = torch.bmm(am_output, lm_output.transpose(1, 2))
            lm_am_sim = 200*lm_am_sim
            lm_am_sim_cp = lm_am_sim.clone()

            lm_am_sim = F.log_softmax(lm_am_sim, dim=-1)
            lm_am_sim = F.pad(lm_am_sim, (1, 0, 0, 0, 0, 0), value=np.log(np.e**-1))
            lm_am_sim = lm_am_sim.contiguous()
            
            if vis:
                lm_am_sim_cp = F.softmax(lm_am_sim_cp, dim=-1)
                #lm_am_sim_prob, lm_am_sim_idx = lm_am_sim_cp.max(-1)
                #lm_am_sim_bool = lm_am_sim_prob > 0.6
                file_name = str(torch.randint(1, 10000, (1,)).item())
                
                _, aligned_idx = lm_am_sim_cp.max(-1)
                print(alinged_idx)
                '''
                for batch in range(lm_am_sim_cp.size(0)):
                    audio_len = lm_am_sim_cp.size(1)
                    target_len = lm_am_sim_cp.size(2)
                    
                    aligned_idx = []
                    alignment = 0
                    
                    sorted_prob, sorted_idx = torch.sort(lm_am_sim_cp[batch], descending=True)

                    for time, (prob, idx) in enumerate(zip(sorted_prob, sorted_idx)):
                        aligned_idx.append(idx[0].item())
                        """
                        i = 0
                        while True:
                            now_alignment = idx[i].item() == alignment
                            should_plus1 = idx[i].item() == (alignment + 1)
                            should_plus2 = idx[i].item() == (alignment + 2)
                            should_plus3 = idx[i].item() == (alignment + 3)
                            
                            if should_plus1:
                                alignment += 1
                                now_alignment = idx[i] == alignment
                            if should_plus2:
                                alignment += 2
                                now_alignment = idx[i] == alignment
                            if should_plus3:
                                alignment += 3
                                now_alignment = idx[i] == alignment

                            if prob[0] < 0.3:
                                print(f'warning: alignment prob is too low, prob: {100*prob[i]} %')

                            if now_alignment: 
                                aligned_idx.append(alignment)
                                print(aligned_idx)
                                break
                            else:
                                i += 1

                            #if i > 3:
                        """
                    #plt.matshow(lm_am_sim_cp[batch][:20,:13].T.cpu().numpy())
                '''
                '''
                plt.matshow(lm_am_sim_cp[batch].T.cpu().numpy())
                plt.colorbar()
                if not os.path.exists(f'./png/{file_name}'):
                    try: os.makedirs(f'./png/{file_name}')
                    except: pass
                plt.savefig(f'./png/{file_name}/alingment{batch}.png')
                plt.close()
                '''
            #print(lm_am_sim.size())
            #print('0'*20)

            ##############################

            #############for alignment target ###############################
            alignment_lengths = torch.sum(lm_input["attention_mask"], 1)
            alignment_target = [[int(j+1) for j in range(alignment_lengths[i])] for i in range(len(alignment_lengths))]
            
            alignment_flat = torch.linspace(
                                                1,
                                                alignment_lengths[0],
                                                steps=alignment_lengths[0]
                                        ).to(device)
            
            for i in alignment_lengths[1:]:
                temp_target = torch.linspace(1, i, steps=i).to(device)
                alignment_flat = torch.cat([alignment_flat, temp_target])
                alignment_flat = alignment_flat.to(torch.cuda.IntTensor())
            #############for alignment target ###############################
            return (x, lm_am_sim, alignment_target), encoder_memory, memory_key_padding_mask
        
        else:
            x = self.ctc_output(encoder_memory)
            return x, encoder_memory, memory_key_padding_mask


class ConformerEncoderLayer(nn.Module):
    """
    ConformerEncoderLayer is made up of self-attn, feedforward and convolution networks.
    See: "Conformer: Convolution-augmented Transformer for Speech Recognition"

    Examples:
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        bypass_scale: float = 0.1,
        layer_dropout: float = 0.075,
        cnn_module_kernel: int = 31,
    ) -> None:
        """
        Args:
          d_model:
            the number of expected features in the input (required).
          nhead:
            the number of heads in the multiheadattention models (required).
          dim_feedforward:
            the dimension of the feedforward network model (default=2048).
          dropout:
            the dropout value (default=0.1).
          bypass_scale:
            a scale on the layer's output, used in bypass (resnet-type) skip-connection;
            when the layer is bypassed the final output will be a
            weighted sum of the layer's input and layer's output with weights
            (1.0-bypass_scale) and bypass_scale correspondingly (default=0.1).
          layer_dropout:
            the probability to bypass the layer (default=0.075).
          cnn_module_kernel (int):
            kernel size of convolution module (default=31).
        """
        super().__init__()

        if bypass_scale < 0.0 or bypass_scale > 1.0:
            raise ValueError("bypass_scale should be between 0.0 and 1.0")

        if layer_dropout < 0.0 or layer_dropout > 1.0:
            raise ValueError("layer_dropout should be between 0.0 and 1.0")

        self.bypass_scale = bypass_scale
        self.layer_dropout = layer_dropout

        self.self_attn = RelPositionMultiheadAttention(d_model, nhead, dropout=0.0)

        self.feed_forward = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.feed_forward_macaron = nn.Sequential(
            ScaledLinear(d_model, dim_feedforward),
            ActivationBalancer(channel_dim=-1),
            DoubleSwish(),
            nn.Dropout(dropout),
            ScaledLinear(dim_feedforward, d_model, initial_scale=0.25),
        )

        self.conv_module = ConvolutionModule(d_model, cnn_module_kernel)

        self.norm_final = BasicNorm(d_model)

        # try to ensure the output is close to zero-mean (or at least, zero-median).
        self.balancer = ActivationBalancer(
            channel_dim=-1, min_positive=0.45, max_positive=0.55, max_abs=6.0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layer.

        Args:
          src:
            the sequence to the encoder layer of shape (S, N, C) (required).
          pos_emb:
            positional embedding tensor of shape (N, 2*S-1, C) (required).
          src_mask:
            the mask for the src sequence of shape (S, S) (optional).
          src_key_padding_mask:
            the mask for the src keys per batch of shape (N, S) (optional).
          warmup:
            controls selective bypass of of layers; if < 1.0, we will
            bypass layers more frequently.

        Returns:
            Output tensor of the shape (S, N, C), where
            S is the source sequence length,
            N is the batch size,
            C is the feature number
        """
        src_orig = src

        warmup_scale = min(self.bypass_scale + warmup, 1.0)
        # alpha = 1.0 means fully use this encoder layer, 0.0 would mean
        # completely bypass it.
        if self.training:
            alpha = (
                warmup_scale
                if torch.rand(()).item() <= (1.0 - self.layer_dropout)
                else self.bypass_scale
            )
        else:
            alpha = 1.0

        # macaron style feed forward module
        src = src + self.dropout(self.feed_forward_macaron(src))

        # multi-headed self-attention module
        src_att = self.self_attn(
            src,
            src,
            src,
            pos_emb=pos_emb,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]

        src = src + self.dropout(src_att)

        # convolution module
        src = src + self.dropout(self.conv_module(src))

        # feed forward module
        src = src + self.dropout(self.feed_forward(src))

        src = self.norm_final(self.balancer(src))

        if alpha != 1.0:
            src = alpha * src + (1 - alpha) * src_orig

        return src


class ConformerEncoder(nn.Module):
    """
    ConformerEncoder is a stack of N encoder layers

    Examples:
        >>> encoder_layer = ConformerEncoderLayer(d_model=512, nhead=8)
        >>> conformer_encoder = ConformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = conformer_encoder(src, pos_emb)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        aux_layers: List[int],
    ) -> None:

        """
        Args:
          encoder_layer:
            an instance of the ConformerEncoderLayer() class (required).
          num_layers:
            the number of sub-encoder-layers in the encoder (required).
          aux_layers:
            list of indexes of sub-encoder-layers outputs to be combined (required).
        """

        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

        assert len(set(aux_layers)) == len(aux_layers)

        assert num_layers - 1 not in aux_layers
        self.aux_layers = aux_layers + [num_layers - 1]

        self.combiner = RandomCombine(
            num_inputs=len(self.aux_layers),
            final_weight=0.5,
            pure_prob=0.333,
            stddev=2.0,
        )

    def forward(
        self,
        src: torch.Tensor,
        pos_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        warmup: float = 1.0,
        condition_layer = None,
        ctc_output = None,
    ) -> torch.Tensor:
        """
        Pass the input through the encoder layers in turn.

        Args:
          src:
            the sequence to the encoder of shape (S, N, C) (required).
          pos_emb:
            positional embedding tensor of shape (N, 2*S-1, C) (required).
          mask:
            the mask for the src sequence of shape (S, S) (optional).
          src_key_padding_mask:
            the mask for the src keys per batch of shape (N, S) (optional).
          warmup:
            controls selective bypass of layer; if < 1.0, we will
            bypass the layer more frequently (default=1.0).

        Returns:
          Output tensor of the shape (S, N, C), where
          S is the source sequence length,
          N is the batch size,
          C is the feature number.

        """
        output = src

        outputs = []
        layer_outputs = []
        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                warmup=warmup,
            )
            
            layer_outputs.append(output)
            if i in self.aux_layers:
                outputs.append(output)

            if i+1 in [3,6,9,12,15] and condition_layer is not None:
                ctc_out = ctc_output(output, log_prob=False)
                output = output + condition_layer(ctc_out).transpose(0,1)

        output = self.combiner(outputs)
        return output, layer_outputs


class RelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.

    See: Appendix B in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/embedding.py

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """
        Construct an PositionalEncoding object.

        Args:
          d_model: Embedding dimension.
          dropout_rate: Dropout rate.
          max_len: Maximum input length.

        """
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor) -> None:
        """
        Reset the positional encodings.

        Args:
          x:
            input tensor (N, T, C), where
            T is the source sequence length,
            N is the batch size.
            C is the feature number.

        """
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                # Note: TorchScript doesn't implement operator== for torch.Device
                if self.pe.dtype != x.dtype or str(self.pe.device) != str(x.device):
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add positional encoding.

        Args:
          x:
            input tensor (N, T, C).

        Returns:
          torch.Tensor: Encoded tensor (N, T, C).
          torch.Tensor: Encoded tensor (N, 2*T-1, C), where
          T is the source sequence length,
          N is the batch size.
          C is the feature number.

        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2
            - x.size(1)
            + 1 : self.pe.size(1) // 2  # noqa E203
            + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class RelPositionMultiheadAttention(nn.Module):
    """
    Multi-Head Attention layer with relative position encoding
    See reference: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context".

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        """
        Args:
          embed_dim:
            total dimension of the model.
          num_heads:
            parallel attention heads.
          dropout:
            a Dropout layer on attn_output_weights. Default: 0.0.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj = ScaledLinear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = ScaledLinear(
            embed_dim, embed_dim, bias=True, initial_scale=0.25
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(embed_dim, embed_dim, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
        self.pos_bias_u_scale = nn.Parameter(torch.zeros(()).detach())
        self.pos_bias_v_scale = nn.Parameter(torch.zeros(()).detach())
        self._reset_parameters()

    def _pos_bias_u(self):
        return self.pos_bias_u * self.pos_bias_u_scale.exp()

    def _pos_bias_v(self):
        return self.pos_bias_v * self.pos_bias_v_scale.exp()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_bias_u, std=0.01)
        nn.init.normal_(self.pos_bias_v, std=0.01)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          query, key, value: map a query and a set of key-value pairs to an output.
          pos_emb: Positional embedding tensor
          key_padding_mask: if provided, specified padding elements in the key will
                            be ignored by the attention. When given a binary mask
                            and a value is True, the corresponding value on the attention
                            layer will be ignored. When given a byte mask and a value is
                            non-zero, the corresponding value on the attention layer will be ignored.
          need_weights: output attn_output_weights.
          attn_mask: 2D or 3D mask that prevents attention to certain positions.
                     A 2D mask will be broadcasted for all the batches while a 3D
                     mask allows to specify a different mask for the entries of each batch.

        Shape:
          - Inputs:
          - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
          - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
          - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
          - pos_emb: :math:`(N, 2*L-1, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
          - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the position
            with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
          - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

          - Outputs:
          - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
          - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """
        return self.multi_head_attention_forward(
            query,
            key,
            value,
            pos_emb,
            self.embed_dim,
            self.num_heads,
            self.in_proj.get_weight(),
            self.in_proj.get_bias(),
            self.dropout,
            self.out_proj.get_weight(),
            self.out_proj.get_bias(),
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positional encoding.

        Args:
          x:
            input tensor (batch, head, time1, 2*time1-1).
            time1 means the length of query vector.

        Returns:
          torch.Tensor: tensor of shape (batch, head, time1, time2)
          (note: time2 has the same value as time1, but it is for
          the key, while time1 is for the query).
        """
        (batch_size, num_heads, time1, n) = x.shape
        assert n == 2 * time1 - 1
        # Note: TorchScript requires explicit arg for stride()
        batch_stride = x.stride(0)
        head_stride = x.stride(1)
        time1_stride = x.stride(2)
        n_stride = x.stride(3)
        return x.as_strided(
            (batch_size, num_heads, time1, time1),
            (batch_stride, head_stride, time1_stride - n_stride, n_stride),
            storage_offset=n_stride * (time1 - 1),
        )

    def multi_head_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_emb: torch.Tensor,
        embed_dim_to_check: int,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        dropout_p: float,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
        training: bool = True,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
          query, key, value: map a query and a set of key-value pairs to an output.
          pos_emb: Positional embedding tensor
          embed_dim_to_check: total dimension of the model.
          num_heads: parallel attention heads.
          in_proj_weight, in_proj_bias: input projection weight and bias.
          dropout_p: probability of an element to be zeroed.
          out_proj_weight, out_proj_bias: the output projection weight and bias.
          training: apply dropout if is ``True``.
          key_padding_mask: if provided, specified padding elements in the key will
                            be ignored by the attention. This is an binary mask.
                            When the value is True, the corresponding value on the
                            attention layer will be filled with -inf.
          need_weights: output attn_output_weights.
          attn_mask: 2D or 3D mask that prevents attention to certain positions.
                     A 2D mask will be broadcasted for all the batches while a 3D
                     mask allows to specify a different mask for the entries of each batch.

        Shape:
          Inputs:
          - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
          - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
          - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
          - pos_emb: :math:`(N, 2*L-1, E)` or :math:`(1, 2*L-1, E)` where L is the target sequence
            length, N is the batch size, E is the embedding dimension.
          - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
            will be unchanged. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
          - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.

          Outputs:
          - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
          - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == embed_dim_to_check
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        head_dim = embed_dim // num_heads
        assert (
            head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        scaling = float(head_dim) ** -0.5

        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = nn.functional.linear(query, in_proj_weight, in_proj_bias).chunk(
                3, dim=-1
            )

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            k, v = nn.functional.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = nn.functional.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = nn.functional.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = nn.functional.linear(value, _w, _b)

        if attn_mask is not None:
            assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
            ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(
                attn_mask.dtype
            )
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask is deprecated. Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * num_heads,
                    query.size(0),
                    key.size(0),
                ]:
                    raise RuntimeError("The size of the 3D attn_mask is not correct.")
            else:
                raise RuntimeError(
                    f"attn_mask's dimension {attn_mask.dim()} is not supported"
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        q = (q * scaling).contiguous().view(tgt_len, bsz, num_heads, head_dim)
        k = k.contiguous().view(-1, bsz, num_heads, head_dim)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

        src_len = k.size(0)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz, "{} == {}".format(
                key_padding_mask.size(0), bsz
            )
            assert key_padding_mask.size(1) == src_len, "{} == {}".format(
                key_padding_mask.size(1), src_len
            )

        q = q.transpose(0, 1)  # (batch, time1, head, d_k)

        pos_emb_bsz = pos_emb.size(0)
        assert pos_emb_bsz in (1, bsz)  # actually it is 1
        p = self.linear_pos(pos_emb).view(pos_emb_bsz, -1, num_heads, head_dim)
        p = p.transpose(1, 2)  # (batch, head, 2*time1-1, d_k)

        q_with_bias_u = (q + self._pos_bias_u()).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        q_with_bias_v = (q + self._pos_bias_v()).transpose(
            1, 2
        )  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" Section 3.3
        k = k.permute(1, 2, 3, 0)  # (batch, head, d_k, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k)  # (batch, head, time1, time2)

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(
            q_with_bias_v, p.transpose(-2, -1)
        )  # (batch, head, time1, 2*time1-1)
        matrix_bd = self.rel_shift(matrix_bd)

        attn_output_weights = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, -1)

        assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(
                bsz * num_heads, tgt_len, src_len
            )

        attn_output_weights = nn.functional.softmax(attn_output_weights, dim=-1)
        attn_output_weights = nn.functional.dropout(
            attn_output_weights, p=dropout_p, training=training
        )

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        )
        attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / num_heads
        else:
            return attn_output, None


class ConvolutionModule(nn.Module):
    def __init__(self, channels: int, kernel_size: int, bias: bool = True) -> None:
        """
        ConvolutionModule in Conformer model.
        Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/conformer/convolution.py
        Construct a ConvolutionModule object.

        Args:
          channels (int):
            the number of channels of conv layers.
          kernel_size (int):
            kernerl size of conv layers.
          bias (bool):
            whether to use bias in conv layers (default=True).
        """
        super().__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = ScaledConv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

        # after pointwise_conv1 we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.deriv_balancer1 = ActivationBalancer(
            channel_dim=1, max_abs=10.0, min_positive=0.05, max_positive=1.0
        )

        self.depthwise_conv = ScaledConv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )

        self.deriv_balancer2 = ActivationBalancer(
            channel_dim=1, min_positive=0.05, max_positive=1.0
        )

        self.activation = DoubleSwish()

        self.pointwise_conv2 = ScaledConv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            initial_scale=0.25,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute convolution module.

        Args:
          x:
            input tensor of shape (T, N, C).

        Returns:
          torch.Tensor: Output tensor (T, N, C), where
          T is the source sequence length,
          N is the batch size,
          C is the feature number.

        """
        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channels, time)

        x = self.deriv_balancer1(x)
        x = nn.functional.glu(x, dim=1)  # (batch, channels, time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)

        x = self.deriv_balancer2(x)
        x = self.activation(x)

        x = self.pointwise_conv2(x)  # (batch, channel, time)

        return x.permute(2, 0, 1)
