# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from scaling import ScaledConv1d, ScaledEmbedding, ScaledLinear
from transformer import (
        TransformerDecoderLayer, 
        TransformerDecoder, 
        PositionalEncoding,
        decoder_padding_mask,
        generate_square_subsequent_mask,
)


from icefall.utils import is_jit_tracing, add_sos


class Decoder(nn.Module):
    """This class modifies the stateless decoder from the following paper:

        RNN-transducer with stateless prediction network
        https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9054419

    It removes the recurrent connection from the decoder, i.e., the prediction
    network. Different from the above paper, it adds an extra Conv1d
    right after the embedding layer.

    TODO: Implement https://arxiv.org/pdf/2109.07513.pdf
    """

    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        context_size: int,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id

        assert context_size >= 1, context_size
        self.context_size = context_size
        self.vocab_size = vocab_size
        if context_size > 1:
            self.conv = ScaledConv1d(
                in_channels=decoder_dim,
                out_channels=decoder_dim,
                kernel_size=context_size,
                padding=0,
                groups=decoder_dim,
                bias=False,
            )
        else:
            # It is to support torch script
            self.conv = nn.Identity()

    def forward(
        self,
        y: torch.Tensor,
        need_pad: bool = True  # Annotation should be Union[bool, torch.Tensor]
        # but, torch.jit.script does not support Union.
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        if isinstance(need_pad, torch.Tensor):
            # This is for torch.jit.trace(), which cannot handle the case
            # when the input argument is not a tensor.
            need_pad = bool(need_pad)

        y = y.to(torch.int64)
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y)
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)
        if self.context_size > 1:
            embedding_out = embedding_out.permute(0, 2, 1)
            if need_pad:
                embedding_out = F.pad(embedding_out, pad=(self.context_size - 1, 0))
            else:
                # During inference time, there is no need to do extra padding
                # as we only need one output
                if not is_jit_tracing():
                    assert embedding_out.size(-1) == self.context_size
            embedding_out = self.conv(embedding_out)
            embedding_out = embedding_out.permute(0, 2, 1)
        embedding_out = F.relu(embedding_out)
        return embedding_out


class TransformerPrediction(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        decoder_dim: int,
        blank_id: int,
        nhead: int,
        num_decoder_layers: int,
        dropout: float = 0.1,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          decoder_dim:
            Dimension of the input embedding, and of the decoder output.
          blank_id:
            The ID of the blank symbol.
          context_size:
            Number of previous words to use to predict the next word.
            1 means bigram; 2 means trigram. n means (n+1)-gram.
        """
        super().__init__()

        self.embedding = ScaledEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=decoder_dim,
            padding_idx=blank_id,
        )
        self.blank_id = blank_id
        self.vocab_size = vocab_size
        
        self.pos = PositionalEncoding(decoder_dim, dropout)
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=decoder_dim*4,
            dropout=dropout,
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            aux_layers=[],
        )
    
    def forward(
        self,
        memory,
        memory_mask,
        y,
        warmup=1.0,
        need_pad: bool = True  # Annotation should be Union[bool, torch.Tensor]
        # but, torch.jit.script does not support Union.
    ) -> torch.Tensor:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U).
          need_pad:
            True to left pad the input. Should be True during training.
            False to not pad the input. Should be False during inference.
        Returns:
          Return a tensor of shape (N, U, decoder_dim).
        """
        device = memory.device
        try: 
            ys_in = add_sos(y, sos_id=self.blank_id)
        except:
            ys_in = y
        ys_in = ys_in.tolist()
        ys_in = [torch.tensor(y) for y in ys_in]
        print(ys_in)
        ys_in_pad = pad_sequence(ys_in, batch_first=True, padding_value=float(self.blank_id))
        print(ys_in_pad)
        tgt_mask = generate_square_subsequent_mask(ys_in_pad.shape[-1]).to(device)
        print(tgt_mask)

        tgt_key_padding_mask = decoder_padding_mask(ys_in_pad, ignore_id=self.blank_id).to(device)
        tgt_key_padding_mask[:, 0] = False

        #y = y.to(torch.int64)
        y = ys_in_pad.to(torch.int64).to(device)
        #print('1111', y)
        #print('222', tgt_mask)
        #print('333', tgt_key_padding_mask)
        #exit()
        # this stuff about clamp() is a temporary fix for a mismatch
        # at utterance start, we use negative ids in beam_search.py
        if torch.jit.is_tracing():
            # This is for exporting to PNNX via ONNX
            embedding_out = self.embedding(y) 
        else:
            embedding_out = self.embedding(y.clamp(min=0)) * (y >= 0).unsqueeze(-1)
        
        ## embedding_out : (N, U, decoder_dim)
        embedding_out = self.pos(embedding_out)
        embedding_out = embedding_out.permute(1, 0, 2) # (N, T, C) -> (T, N, C)
        memory = memory.permute(1, 0, 2)
        
        #print(embedding_out.size())
        #print(memory.size())
        #print(tgt_mask.size())
        #print(tgt_key_padding_mask.size())
        #print(memory_mask.size())
        
        embedding_out = self.decoder(
            tgt=embedding_out,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_mask,
            warmup=warmup,
        )  # (T, N, C)

        embedding_out = embedding_out.permute(1, 0, 2) # (N, T, C) -> (T, N, C)

        return embedding_out
