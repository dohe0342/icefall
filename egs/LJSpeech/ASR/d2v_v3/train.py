#!/usr/bin/env python3
# Copyright    2021-2022  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Mingshuang Luo,)
#                                                       Zengwei Yao)
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
"""
Usage:

export CUDA_VISIBLE_DEVICES="0,1,2,3"

./pruned_transducer_stateless7_ctc/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --exp-dir pruned_transducer_stateless7_ctc/exp \
  --full-libri 1 \
  --max-duration 300

# For mix precision training:

./pruned_transducer_stateless7_ctc/train.py \
  --world-size 4 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --exp-dir pruned_transducer_stateless7_ctc/exp \
  --full-libri 1 \
  --max-duration 550

# For d2v-T training:
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

./pruned_transducer_stateless_d2v_v2/train.py \
    --wandb true \
    --input-strategy AudioSamples \
    --enable-spec-aug False \
    --multi-optim True \
    --world-size 8 \ 
    --num-epochs 30 \
    --start-epoch 1 \ 
    --full-libri 0 \ 
    --exp-dir ./pruned_transducer_stateless_d2v_v2/$1 \
    --max-duration 250 \
    --freeze-finetune-updates 2000 \
    --use-fp16 1 \ 
    --peak-enc-lr 0.001 \
    --peak-dec-lr 0.05 \
    --accum-grads 1 \ 
    --encoder-type d2v \
    --additional-block True \
    --encoder-dim 768 \
    --decoder-dim 768 \
    --joiner-dim 768 \
    --prune-range 20 \
    --context-size 2 \ 
    --ctc-loss-scale 0.2

"""


import random
import argparse
import copy
import logging
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional, Tuple, Union

import k2
import optim
import sentencepiece as spm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from decoder import Decoder
from joiner import Joiner
from lhotse.cut import Cut
from lhotse.dataset.sampling.base import CutSampler
from lhotse.utils import fix_random_seed
from model import Transducer
from optim import Eden, ScaledAdam
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from zipformer import Zipformer
from data2vec_encoder import FairSeqData2VecEncoder

from icefall import diagnostics
from icefall.checkpoint import remove_checkpoints
from icefall.checkpoint import update_averaged_model
from checkpoint import (
    save_checkpoint as save_checkpoint_impl,
    save_checkpoint_with_global_batch_idx,
    load_checkpoint
)
from icefall.dist import cleanup_dist, setup_dist
from icefall.env import get_env_info
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
    save_args,
)

import wandb

#from icefall.checkpoint import save_checkpoint as save_checkpoint_impl
LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, optim.LRScheduler]


def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    if isinstance(model, DDP):
        # get underlying nn.Module
        model = model.module
    for module in model.modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
    model.encoder.num_updates = int(batch_count)


def add_adapter_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--add-adapter",
        type=str2bool,
        default=False,
        help="add adapter to rep model's encoder"
    )
    
    parser.add_argument(
        "--adapter-lr",
        type=float,
        default=0.0001,
        help="adapter learning rate"
    )


def add_rep_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=True,
        help="Use wandb for MLOps",
    )
    parser.add_argument(
        "--accum-grads",
        type=int,
        default=1,
        help="accum-grad num.",
    )

    parser.add_argument(
        "--multi-optim",
        type=str2bool,
        default=True,
        help="use sperate optimizer (enc / dec)",
    )
    
    parser.add_argument(
        "--peak-enc-lr",
        type=float,
        default=0.0001,
        help="The initial learning rate.  This value should not need to be changed.",
    )

    parser.add_argument(
        "--peak-dec-lr",
        type=float,
        default=0.001,
        help="The initial learning rate.  This value should not need to be changed.",
    )
    
    parser.add_argument(
        "--encoder-type",
        type=str,
        default='d2v',
        help="Type of encoder (e.g. conformer, w2v, d2v...",
    )
    
    parser.add_argument(
        "--encoder-dim",
        type=int,
        default=768,
        help="encoder embedding dimension",
    )
    
    parser.add_argument(
        "--freeze-finetune-updates",
        type=int,
        default=0
    )

    parser.add_argument(
        "--additional-block",
        type=str2bool,
        default=True,
    )

    parser.add_argument(
        "--decode-interval",
        type=int,
        default=200,
        help="decode interval",
    )
        

def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,4,3,2,4",
        help="Number of zipformer encoder layers, comma separated.",
    )

    parser.add_argument(
        "--feedforward-dims",
        type=str,
        default="1024,1024,2048,2048,1024",
        help="Feedforward dimension of the zipformer encoder layers, comma separated.",
    )

    parser.add_argument(
        "--nhead",
        type=str,
        default="8,8,8,8,8",
        help="Number of attention heads in the zipformer encoder layers.",
    )

    parser.add_argument(
        "--encoder-dims",
        type=str,
        default="384,384,384,384,384",
        help="Embedding dimension in the 2 blocks of zipformer encoder layers, comma separated",
    )

    parser.add_argument(
        "--attention-dims",
        type=str,
        default="192,192,192,192,192",
        help="""Attention dimension in the 2 blocks of zipformer encoder layers, comma separated;
        not the same as embedding dimension.""",
    )

    parser.add_argument(
        "--encoder-unmasked-dims",
        type=str,
        default="256,256,256,256,256",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "Must be <= each of encoder_dims.  Empirically, less than 256 seems to make performance "
        " worse.",
    )

    parser.add_argument(
        "--zipformer-downsampling-factors",
        type=str,
        default="1,2,4,8,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--cnn-module-kernels",
        type=str,
        default="31,31,31,31,31",
        help="Sizes of kernels in convolution modules",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="""If positive, --start-epoch is ignored and
        it loads the checkpoint from exp-dir/checkpoint-{start_batch}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_ctc/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.05, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=5000,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--lr-epochs",
        type=float,
        default=3.5,
        help="""Number of epochs that affects how rapidly the learning rate decreases.
        """,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )

    parser.add_argument(
        "--prune-range",
        type=int,
        default=5,
        help="The prune range for rnnt loss, it means how many symbols(context)"
        "we are using to compute the loss",
    )

    parser.add_argument(
        "--lm-scale",
        type=float,
        default=0.25,
        help="The scale to smooth the loss with lm "
        "(output of prediction network) part.",
    )

    parser.add_argument(
        "--am-scale",
        type=float,
        default=0.0,
        help="The scale to smooth the loss with am (output of encoder network) part.",
    )

    parser.add_argument(
        "--simple-loss-scale",
        type=float,
        default=0.5,
        help="To get pruning ranges, we will calculate a simple version"
        "loss(joiner is just addition), this simple loss also uses for"
        "training (as a regularization item). We will scale the simple loss"
        "with this parameter before adding to the final loss.",
    )

    parser.add_argument(
        "--ctc-loss-scale",
        type=float,
        default=0.2,
        help="Scale for CTC loss.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=2000,
        help="""Save checkpoint after processing this number of batches"
        periodically. We save checkpoint to exp-dir/ whenever
        params.batch_idx_train % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/checkpoint-{params.batch_idx_train}.pt'
        Note: It also saves checkpoint to `exp-dir/epoch-xxx.pt` at the
        end of each epoch where `xxx` is the epoch number counting from 0.
        """,
    )

    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=30,
        help="""Only keep this number of checkpoints on disk.
        For instance, if it is 3, there are only 3 checkpoints
        in the exp-dir with filenames `checkpoint-xxx.pt`.
        It does not affect checkpoints with name `epoch-xxx.pt`.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=True,
        help="Whether to use half precision training.",
    )

    add_model_arguments(parser)
    add_rep_arguments(parser)
    add_adapter_arguments(parser)

    return parser


def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    Commandline options are merged into `params` after they are parsed, so
    you can also access them via `params`.

    Explanation of options saved in `params`:

        - best_train_loss: Best training loss so far. It is used to select
                           the model that has the lowest training loss. It is
                           updated during the training.

        - best_valid_loss: Best validation loss so far. It is used to select
                           the model that has the lowest validation loss. It is
                           updated during the training.

        - best_train_epoch: It is the epoch that has the best training loss.

        - best_valid_epoch: It is the epoch that has the best validation loss.

        - batch_idx_train: Used to writing statistics to tensorboard. It
                           contains number of batches trained so far across
                           epochs.

        - log_interval:  Print training loss if batch_idx % log_interval` is 0

        - reset_interval: Reset statistics if batch_idx % reset_interval is 0

        - valid_interval:  Run validation if batch_idx % valid_interval is 0

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

        - encoder_dim: Hidden dim for multi-head attention model.

        - num_decoder_layers: Number of decoder layer of transformer decoder.

        - warm_step: The warmup period that dictates the decay of the
              scale on "simple" (un-pruned) loss.
    """
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 50,
            "reset_interval": 200,
            "valid_interval": 3000,  # For the 100h subset, use 800
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 320,  # not passed in, this is fixed.
            # parameters for ctc loss
            "beam_size": 10,
            "use_double_scores": True,
            "warm_step": 4000,
            #"warm_step": 3000,
            "env_info": get_env_info(),
        }
    )

    return params


def get_encoder_model(params: AttributeDict) -> nn.Module:
    # TODO: We can add an option to switch between Zipformer and Transformer
    def to_int_tuple(s: str):
        return tuple(map(int, s.split(",")))
    
    if params.encoder_type == 'd2v':
        encoder = FairSeqData2VecEncoder(
                    input_size=params.encoder_dim,
                    w2v_url='None',
                    output_size=params.encoder_dim,
                    freeze_finetune_updates=params.freeze_finetune_updates,
                    additional_block=params.additional_block,
                ) 
    else:
        encoder = Zipformer(
            num_features=params.feature_dim,
            output_downsampling_factor=2,
            zipformer_downsampling_factors=to_int_tuple(
                params.zipformer_downsampling_factors
            ),
            encoder_dims=to_int_tuple(params.encoder_dims),
            attention_dim=to_int_tuple(params.attention_dims),
            encoder_unmasked_dims=to_int_tuple(params.encoder_unmasked_dims),
            nhead=to_int_tuple(params.nhead),
            feedforward_dim=to_int_tuple(params.feedforward_dims),
            cnn_module_kernels=to_int_tuple(params.cnn_module_kernels),
            num_encoder_layers=to_int_tuple(params.num_encoder_layers),
        )

    return encoder


def get_decoder_model(params: AttributeDict) -> nn.Module:
    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )
    return decoder


def get_joiner_model(params: AttributeDict) -> nn.Module:
    joiner = Joiner(
        encoder_dim=params.encoder_dim if params.encoder_type == 'd2v' else int(params.encoder_dims.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return joiner


def get_transducer_model(params: AttributeDict) -> nn.Module:
    encoder = get_encoder_model(params)
    decoder = get_decoder_model(params)
    joiner = get_joiner_model(params)

    model = Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=params.encoder_dim if params.encoder_type == 'd2v' else int(params.encoder_dims.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )
    return model


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_batch is positive, it will load the checkpoint from
    `params.exp_dir/checkpoint-{params.start_batch}.pt`. Otherwise, if
    params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer` it also updates
    `best_train_epoch`, `best_train_loss`, `best_valid_epoch`,
    and `best_valid_loss` in `params`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer that we are using.
      scheduler:
        The scheduler that we are using.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_batch > 0:
        filename = params.exp_dir / f"checkpoint-{params.start_batch}.pt"
    elif params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch-1}.pt"
    elif params.add_adapter:
        filename = params.exp_dir / f"d2v-base-T.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
        optimizer=optimizer,
        scheduler=scheduler,
        strict=True if not params.add_adapter else False,
    )

    keys = [
        "best_train_epoch",
        "best_valid_epoch",
        "batch_idx_train",
        "best_train_loss",
        "best_valid_loss",
    ]
    for k in keys:
        params[k] = saved_params[k]

    if params.start_batch > 0:
        if "cur_epoch" in saved_params:
            params["start_epoch"] = saved_params["cur_epoch"]

        if "cur_batch_idx" in saved_params:
            params["cur_batch_idx"] = saved_params["cur_batch_idx"]

    return saved_params


def save_checkpoint(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    sampler: Optional[CutSampler] = None,
    scaler: Optional[GradScaler] = None,
    rank: int = 0,
) -> None:
    """Save model, optimizer, scheduler and training stats to file.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The training model.
      model_avg:
        The stored model averaged from the start of training.
      optimizer:
        The optimizer used in the training.
      sampler:
       The sampler for the training dataset.
      scaler:
        The scaler used for mix precision training.
    """
    if rank != 0:
        return
    filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
    save_checkpoint_impl(
        filename=filename,
        model=model,
        model_avg=model_avg,
        params=params,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        scaler=scaler,
        rank=rank,
    )

    if params.best_train_epoch == params.cur_epoch:
        best_train_filename = params.exp_dir / "best-train-loss.pt"
        copyfile(src=filename, dst=best_train_filename)

    if params.best_valid_epoch == params.cur_epoch:
        best_valid_filename = params.exp_dir / "best-valid-loss.pt"
        copyfile(src=filename, dst=best_valid_filename)


def compute_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    decode: bool = False,
) -> Tuple[Tensor, MetricsTracker]:
    """
    Compute transducer loss given the model and its inputs.

    Args:
      params:
        Parameters for training. See :func:`get_params`.
      model:
        The model for training. It is an instance of Zipformer in our case.
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      is_training:
        True for training. False for validation. When it is True, this
        function enables autograd during computation; when it is False, it
        disables autograd.
     warmup: a floating point value which increases throughout training;
        values >= 1.0 are fully warmed up and have all modules present.
    """
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    feature = batch["inputs"]
    # at entry, feature is (N, T, C)
    assert feature.ndim == 2 or feature.ndim == 3
    feature = feature.to(device)

    supervisions = batch["supervisions"]

    if feature.ndim == 2:
        feature_lens = []
        for supervision in supervisions['cut']:
            try: feature_lens.append(supervision.tracks[0].cut.recording.num_samples)
            except: feature_lens.append(supervision.recording.num_samples)
        feature_lens = torch.tensor(feature_lens)

    elif feature.ndim == 3:
        feature_lens = supervisions["num_frames"].to(device)

    batch_idx_train = params.batch_idx_train
    warm_step = params.warm_step

    texts = batch["supervisions"]["text"]
    token_ids = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(token_ids).to(device)

    with torch.set_grad_enabled(is_training):
        simple_loss, pruned_loss, ctc_output = model(
            x=feature,
            x_lens=feature_lens,
            y=y,
            prune_range=params.prune_range,
            am_scale=params.am_scale,
            lm_scale=params.lm_scale,
        )

        s = params.simple_loss_scale
        # take down the scale on the simple loss from 1.0 at the start
        # to params.simple_loss scale by warm_step.
        simple_loss_scale = (
            s
            if batch_idx_train >= warm_step
            else 1.0 - (batch_idx_train / warm_step) * (1.0 - s)
        )
        pruned_loss_scale = (
            1.0
            if batch_idx_train >= warm_step
            else 0.1 + 0.9 * (batch_idx_train / warm_step)
        )

        loss = simple_loss_scale * simple_loss + pruned_loss_scale * pruned_loss
    
    info = MetricsTracker()
    
    if params.ctc_loss_scale > 0:
        # Compute ctc loss

        # NOTE: We need `encode_supervisions` to sort sequences with
        # different duration in decreasing order, required by
        # `k2.intersect_dense` called in `k2.ctc_loss`
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            supervision_segments, token_ids = encode_supervisions(
                supervisions,
                subsampling_factor=params.subsampling_factor,
                token_ids=token_ids,
            )
        
        # Works with a BPE model
        decoding_graph = k2.ctc_graph(token_ids, modified=False, device=device)
        dense_fsa_vec = k2.DenseFsaVec(
            ctc_output,
            supervision_segments,
            allow_truncate=params.subsampling_factor - 1,
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=params.beam_size,
            reduction="sum",
            use_double_scores=params.use_double_scores,
        )
        assert ctc_loss.requires_grad == is_training
        loss += params.ctc_loss_scale * ctc_loss
    
        info["ctc_loss"] = ctc_loss.detach().cpu().item()
    
    assert loss.requires_grad == is_training

    if decode:
        model.eval()
        with torch.no_grad():
            hypos = model.module.decode(
                x=feature,
                x_lens=feature_lens,
                y=y,
                sp=sp
            )
            logging.info(f'ref: {batch["supervisions"]["text"][0]}')
            logging.info(f'hyp: {" ".join(hypos[0])}')
        model.train()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info["frames"] = (feature_lens // params.subsampling_factor).sum().item()

    # Note: We use reduction=sum while computing the loss.
    info["utterances"] = feature.size(0)
    info["loss"] = loss.detach().cpu().item()
    info["simple_loss"] = simple_loss.detach().cpu().item()
    info["pruned_loss"] = pruned_loss.detach().cpu().item()

    return loss, info


def compute_validation_loss(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    sp: spm.SentencePieceProcessor,
    valid_dl: torch.utils.data.DataLoader,
    world_size: int = 1,
) -> MetricsTracker:
    """Run the validation process."""
    model.eval()

    tot_loss = MetricsTracker()

    for batch_idx, batch in enumerate(valid_dl):
        loss, loss_info = compute_loss(
            params=params,
            model=model,
            sp=sp,
            batch=batch,
            is_training=False,
        )
        assert loss.requires_grad is False
        tot_loss = tot_loss + loss_info

    if world_size > 1:
        tot_loss.reduce(loss.device)

    loss_value = tot_loss["loss"] / tot_loss["utterances"]
    if loss_value < params.best_valid_loss:
        params.best_valid_epoch = params.cur_epoch
        params.best_valid_loss = loss_value

    return tot_loss


def train_one_epoch(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer or [torch.optim.Optimizer, torch.optim.Optimizer],
    scheduler: LRSchedulerType or [LRSchedulerType, LRSchedulerType],
    sp: spm.SentencePieceProcessor,
    train_dl: torch.utils.data.DataLoader,
    valid_dl: torch.utils.data.DataLoader,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
    wb = None,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      optimizer:
        The optimizer we are using.
      scheduler:
        The learning rate scheduler, we call step() every step.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      model_avg:
        The stored model averaged from the start of training.
      tb_writer:
        Writer to write log messages to tensorboard.
      world_size:
        Number of nodes in DDP training. If it is 1, DDP is disabled.
      rank:
        The rank of the node in DDP training. If no DDP is used, it should
        be set to 0.
    """
    model.train()

    tot_loss = MetricsTracker()

    cur_batch_idx = params.get("cur_batch_idx", 0)

    if params.multi_optim:
        optimizer_enc, optimizer_dec = optimizer[0], optimizer[1]
        scheduler_enc, scheduler_dec = scheduler[0], scheduler[1]

    for batch_idx, batch in enumerate(train_dl):
        if batch_idx < cur_batch_idx:
            continue
        cur_batch_idx = batch_idx
    
        if batch_idx % params.accum_grads == 0: params.batch_idx_train += 1
        batch_size = len(batch["supervisions"]["text"])

        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, loss_info = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                    decode = True if batch_idx % params.decode_interval == 0 else False,
                )
            loss_info.reduce(loss.device)

            numel = params.world_size / (params.accum_grads * loss_info["utterances"])
            loss *= numel ## normalize loss over utts(batch size)

            # summary stats
            tot_loss = (tot_loss * (1 - 1 / params.reset_interval)) + loss_info

            # NOTE: We use reduction==sum and loss is computed over utterances
            # in the batch and there is no normalization to it so far.
            scaler.scale(loss).backward()

            if params.multi_optim and (batch_idx+1) % params.accum_grads == 0:
                set_batch_count(model, params.batch_idx_train)
                scheduler_enc.step_batch(params.batch_idx_train)
                scheduler_dec.step_batch(params.batch_idx_train)
                scaler.step(optimizer_enc)
                scaler.step(optimizer_dec)
                scaler.update()
                optimizer_enc.zero_grad()
                optimizer_dec.zero_grad()
            elif not params.multi_optim and (batch_idx+1) % params.accum_grads == 0:
                set_batch_count(model, params.batch_idx_train)
                scheduler.step_batch(params.batch_idx_train)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        except:  # noqa
            display_and_save_batch(batch, params=params, sp=sp)
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if (
            params.batch_idx_train > 0
            and params.batch_idx_train % params.save_every_n == 0
        ):
            params.cur_batch_idx = batch_idx
            save_checkpoint_with_global_batch_idx(
                out_dir=params.exp_dir,
                global_batch_idx=params.batch_idx_train,
                model=model,
                model_avg=model_avg,
                params=params,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=train_dl.sampler,
                scaler=scaler,
                rank=rank,
            )
            del params.cur_batch_idx
            remove_checkpoints(
                out_dir=params.exp_dir,
                topk=params.keep_last_k,
                rank=rank,
            )

        if batch_idx % 100 == 0 and params.use_fp16:
            # If the grad scale was less than 1, try increasing it.    The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            cur_grad_scale = scaler._scale.item()
            '''
            if cur_grad_scale < 1.0 or (cur_grad_scale < 8.0 and batch_idx % 400 == 0):
                scaler.update(cur_grad_scale * 2.0)
            '''
            if cur_grad_scale < 0.01:
                logging.warning(f"Grad scale is small: {cur_grad_scale}")
            if cur_grad_scale < 1.0e-05:
                wb.log({"valid/loss": 10000})
                raise RuntimeError(
                    f"grad_scale is too small, exiting: {cur_grad_scale}"
                )

        if params.batch_idx_train > 4000 and loss > 300 and params.wandb:
            wb.log({"valid/loss": 10000})
            raise RunteimError(
                    f"divergence... exiting: loss={loss}"
                )

        if batch_idx % (params.log_interval*params.accum_grads) == 0:
            if params.multi_optim:
                cur_enc_lr = scheduler_enc.get_last_lr()[0]
                cur_dec_lr = scheduler_dec.get_last_lr()[0]
                cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

                logging.info(
                    f"Epoch {params.cur_epoch}, "
                    f"batch {batch_idx}, loss[{loss_info}], "
                    f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                    f"enc_lr: {cur_enc_lr:.2e}, "
                    f"dec_lr: {cur_dec_lr:.2e}, "
                    + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
                )

            else:
                cur_lr = scheduler.get_last_lr()[0]
                cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

                logging.info(
                    f"Epoch {params.cur_epoch}, "
                    f"batch {batch_idx}, loss[{loss_info}], "
                    f"tot_loss[{tot_loss}], batch size: {batch_size}, "
                    f"lr: {cur_lr:.2e}, "
                    + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
                )

            if tb_writer is not None:
                if params.multi_optim:
                    tb_writer.add_scalar(
                        "train/enc_learning_rate", cur_enc_lr, params.batch_idx_train
                    )
                    tb_writer.add_scalar(
                        "train/dec_learning_rate", cur_dec_lr, params.batch_idx_train
                    )

                else:
                    tb_writer.add_scalar(
                        "train/learning_rate", cur_lr, params.batch_idx_train
                    )

                loss_info.write_summary(
                    tb_writer, "train/current_", params.batch_idx_train
                )
                tot_loss.write_summary(tb_writer, "train/tot_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar(
                        "train/grad_scale",
                        cur_grad_scale,
                        params.batch_idx_train,
                    )
            
            if wb is not None and rank == 0:
                wb.log({"train/loss": loss_info["loss"]*numel})
                wb.log({"train/simple_loss": loss_info["simple_loss"]*numel})
                wb.log({"train/pruned_loss": loss_info["pruned_loss"]*numel})
                wb.log({"train/ctc_loss": loss_info["ctc_loss"]*numel})

    logging.info("Computing validation loss")
    valid_info = compute_validation_loss(
        params=params,
        model=model,
        sp=sp,
        valid_dl=valid_dl,
        world_size=world_size,
    )
    model.train()
    logging.info(f"Epoch {params.cur_epoch}, validation: {valid_info}")
    logging.info(
        f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
    )
    if tb_writer is not None:
        valid_info.write_summary(
            tb_writer, "train/valid_", params.batch_idx_train
        )
    
    if wb is not None and rank == 0:
        numel = 1 / (params.accum_grads * valid_info["utterances"])
        wb.log({"valid/loss": valid_info["loss"]*numel})
        wb.log({"valid/simple_loss": valid_info["simple_loss"]*numel})
        wb.log({"valid/pruned_loss": valid_info["pruned_loss"]*numel})
        wb.log({"valid/ctc_loss": valid_info["ctc_loss"]*numel})

    loss_value = tot_loss["loss"] / tot_loss["utterances"]
    params.train_loss = loss_value
    if params.train_loss < params.best_train_loss:
        params.best_train_epoch = params.cur_epoch
        params.best_train_loss = params.train_loss


def run(rank, world_size, args, wb=None):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))
    #params.warm_step *= params.accum_grads

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)
    logging.info(model)
    exit()

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    if params.multi_optim:
        logging.info("Using seperate optimizers over encoder, decoder ...")

        enc_param = []
        enc_names = []

        dec_names = []
        dec_param = []
        
        for n, p in model.named_parameters():
            name = n.split('.')[1]
            if name == 'encoder' and 'feature_extractor' not in n:
                enc_names.append(n)
                enc_param.append(p)
            elif 'ctc_output' in n:
                enc_names.append(n)
                enc_param.append(p)
            elif 'feature_extractor' not in n:
                dec_names.append(n)
                dec_param.append(p)

        optimizer_enc = ScaledAdam(
            enc_param,
            lr=params.peak_enc_lr,
            clipping_scale=None,
            parameters_names=[enc_names],
        )
        optimizer_dec = ScaledAdam(
            dec_param,
            lr=params.peak_dec_lr,
            clipping_scale=5.0,
            parameters_names=[dec_names],
        )

        scheduler_enc = Eden(optimizer_enc, params.lr_batches, params.lr_epochs)
        scheduler_dec = Eden(optimizer_dec, params.lr_batches, params.lr_epochs)
        optimizer = [optimizer_enc, optimizer_dec]
        scheduler = [scheduler_enc, scheduler_dec]

    else:
        parameters_names = []
        parameters_names.append(
            [name_param_pair[0] for name_param_pair in model.named_parameters()]
        )

        logging.info(f"len name = {len(parameters_names)}")
        logging.info(f"len param = {len(list(model.parameters()))}")
        
        optimizer = ScaledAdam(
            model.parameters(),
            lr=params.base_lr,
            clipping_scale=2.0,
            parameters_names=parameters_names,
        )

        scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and ("optimizer" in checkpoints or "optimizer_enc" in checkpoints):
        if params.multi_optim:
            logging.info("Loading optimizer state dict")
            optimizer_enc.load_state_dict(checkpoints["optimizer_enc"])
            optimizer_dec.load_state_dict(checkpoints["optimizer_dec"])

        else:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])

    if checkpoints:
        if (
            params.multi_optim 
            and "scheduler_enc" in checkpoints
            and checkpoints["scheduler_enc"] is not None
        ):
            logging.info("Loading enc/dec scheduler state dict")
            scheduler_enc.load_state_dict(checkpoints["scheduler_enc"])
            scheduler_dec.load_state_dict(checkpoints["scheduler_dec"])        
        else:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            2**22
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = librispeech.train_clean_100_cuts()
    if params.full_libri:
        train_cuts += librispeech.train_clean_360_cuts()
        train_cuts += librispeech.train_other_500_cuts()

    def remove_short_and_long_utt(c: Cut):
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ../local/display_manifest_statistics.py
        #
        # You should use ../local/display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = librispeech.dev_clean_cuts()
    valid_cuts += librispeech.dev_other_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)
    
    '''
    if not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
        )
    '''

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        if params.multi_optim:
            scheduler_enc.step_epoch(epoch - 1)
            scheduler_dec.step_epoch(epoch - 1)
        else:
            scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            wb=wb,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def run_adapter(rank, world_size, args, wb=None):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    num_param = sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params, model=model, model_avg=model_avg
    )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    adapter_names = []
    adapter_param = []
    for n, p  in model.named_parameters():
        if 'adapters' in n:
            adapter_names.append(n)
            adapter_param.append(p)
        #else:
        #    p.requires_grad = False
    optimizer_adapter = ScaledAdam(
            adapter_param,
            lr=params.adapter_lr,
            clipping_scale=5.0,
            parameters_names=[adapter_names],
        )
    scheduler_adapter = Eden(optimizer_adapter, 5000, 3.5) #params.lr_batche, params.lr_epochs)

    optimizer, scheduler = optimizer_adapter, scheduler_adapter
    
    librispeech = LibriSpeechAsrDataModule(args)

    train_cuts = librispeech.train_cuts()

    def remove_short_and_long_utt(c: Cut):
        return 1.0 <= c.duration <= 20.0

    train_cuts = train_cuts.filter(remove_short_and_long_utt)

    sampler_state_dict = None

    #train_dl = librispeech.train_dataloaders(
    #    train_cuts, sampler_state_dict=sampler_state_dict
    #)

    #valid_cuts = librispeech.dev_clean_cuts()
    #valid_cuts += librispeech.dev_other_cuts()
    #valid_dl = librispeech.valid_dataloaders(valid_cuts)
    train_dl = librispeech.train_dataloaders(
        train_cuts, sampler_state_dict=sampler_state_dict
    )

    valid_cuts = librispeech.dev_cuts()
    valid_dl = librispeech.valid_dataloaders(valid_cuts)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dl=valid_dl,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
            wb=wb,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def display_and_save_batch(
    batch: dict,
    params: AttributeDict,
    sp: spm.SentencePieceProcessor,
) -> None:
    """Display the batch statistics and save the batch into disk.

    Args:
      batch:
        A batch of data. See `lhotse.dataset.K2SpeechRecognitionDataset()`
        for the content in it.
      params:
        Parameters for training. See :func:`get_params`.
      sp:
        The BPE model.
    """
    from lhotse.utils import uuid4

    filename = f"{params.exp_dir}/batch-{uuid4()}.pt"
    logging.info(f"Saving batch to {filename}")
    torch.save(batch, filename)

    supervisions = batch["supervisions"]
    features = batch["inputs"]

    logging.info(f"features shape: {features.shape}")

    y = sp.encode(supervisions["text"], out_type=int)
    num_tokens = sum(len(i) for i in y)
    logging.info(f"num tokens: {num_tokens}")


def scan_pessimistic_batches_for_oom(
    model: Union[nn.Module, DDP],
    train_dl: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    params: AttributeDict,
):
    from lhotse.dataset import find_pessimistic_batches

    logging.info(
        "Sanity check -- see if any of the batches in epoch 1 would cause OOM."
    )
    batches, crit_values = find_pessimistic_batches(train_dl.sampler)
    for criterion, cuts in batches.items():
        batch = train_dl.dataset[cuts]
        try:
            with torch.cuda.amp.autocast(enabled=params.use_fp16):
                loss, _ = compute_loss(
                    params=params,
                    model=model,
                    sp=sp,
                    batch=batch,
                    is_training=True,
                )
            loss.backward()
            optimizer.zero_grad()
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logging.error(
                    "Your GPU ran out of memory with the current "
                    "max_duration setting. We recommend decreasing "
                    "max_duration and trying again.\n"
                    f"Failing criterion: {criterion} "
                    f"(={crit_values[criterion]}) ..."
                )
            display_and_save_batch(batch, params=params, sp=sp)
            raise
        logging.info(
            f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated()//1000000}MB"
        )


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    #args.exp_dir = args.exp_dir + str(random.randint(0,400))
    args.exp_dir = Path(args.exp_dir)

    logging.info("save arguments to config.yaml...")
    save_args(args)

    if args.wandb: wb = wandb.init(project="d2v-T", entity="dohe0342", config=vars(args))
    else: wb = None

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run if not args.add_adapter else run_adapter, 
                 args=(world_size, args, wb), 
                 nprocs=world_size, 
                 join=True
            )
    else:
        if not args.add_adapter: run(rank=0, world_size=1, args=args, wb=wb)
        else: run(rank=0, world_size=1, args=args, wb=wb)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()
