#!/usr/bin/env python3
#
# Copyright 2021-2022 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
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


import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import k2
import sentencepiece as spm
import torch
import torch.nn as nn
from asr_datamodule import LibriSpeechAsrDataModule
from beam_search import (
    beam_search,
    fast_beam_search_nbest,
    fast_beam_search_nbest_LG,
    fast_beam_search_nbest_oracle,
    fast_beam_search_one_best,
    greedy_search,
    greedy_search_batch,
    modified_beam_search,
)

#from train_tta import add_model_arguments, add_rep_arguments, get_params, get_transducer_model
from prompt_tuning import add_model_arguments, add_rep_arguments, get_params, get_transducer_model

from icefall.checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    find_checkpoints,
    load_checkpoint,
)
from icefall.lexicon import Lexicon
from icefall.utils import (
    AttributeDict,
    setup_logger,
    store_transcripts,
    str2bool,
    write_error_stats,
)

import fairseq

from optim import Eden, ScaledAdam
from copy import deepcopy

LOG_EPS = math.log(1e-10)

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="""It specifies the model file name to use for decoding.""",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--iter",
        type=int,
        default=0,
        help="""If positive, --epoch is ignored and it
        will use the checkpoint exp_dir/checkpoint-iter.pt.
        You can specify --avg to use more checkpoints for model averaging.
        """,
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=9,
        help="Number of checkpoints to average. Automatically select "
        "consecutive checkpoints before the checkpoint specified by "
        "'--epoch' and '--iter'",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="pruned_transducer_stateless7_ctc/exp",
        help="The experiment dir",
    )

    parser.add_argument(
        "--bpe-model",
        type=str,
        default="data/lang_bpe_500/bpe.model",
        help="Path to the BPE model",
    )

    parser.add_argument(
        "--lang-dir",
        type=Path,
        default="data/lang_bpe_500",
        help="The lang dir containing word table and LG graph",
    )

    parser.add_argument(
        "--decoding-method",
        type=str,
        default="greedy_search",
        help="""Possible values are:
          - greedy_search
          - beam_search
          - modified_beam_search
          - fast_beam_search
          - fast_beam_search_nbest
          - fast_beam_search_nbest_oracle
          - fast_beam_search_nbest_LG
        If you use fast_beam_search_nbest_LG, you have to specify
        `--lang-dir`, which should contain `LG.pt`.
        """,
    )

    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="""An integer indicating how many candidates we will keep for each
        frame. Used only when --decoding-method is beam_search or
        modified_beam_search.""",
    )

    parser.add_argument(
        "--beam",
        type=float,
        default=20.0,
        help="""A floating point value to calculate the cutoff score during beam
        search (i.e., `cutoff = max-score - beam`), which is the same as the
        `beam` in Kaldi.
        Used only when --decoding-method is fast_beam_search,
        fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle
        """,
    )

    parser.add_argument(
        "--ngram-lm-scale",
        type=float,
        default=0.01,
        help="""
        Used only when --decoding_method is fast_beam_search_nbest_LG.
        It specifies the scale for n-gram LM scores.
        """,
    )

    parser.add_argument(
        "--max-contexts",
        type=int,
        default=8,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--max-states",
        type=int,
        default=64,
        help="""Used only when --decoding-method is
        fast_beam_search, fast_beam_search_nbest, fast_beam_search_nbest_LG,
        and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=2,
        help="The context size in the decoder. 1 means bigram; 2 means tri-gram",
    )
    parser.add_argument(
        "--max-sym-per-frame",
        type=int,
        default=1,
        help="""Maximum number of symbols per frame.
        Used only when --decoding_method is greedy_search""",
    )

    parser.add_argument(
        "--num-paths",
        type=int,
        default=200,
        help="""Number of paths for nbest decoding.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--nbest-scale",
        type=float,
        default=0.5,
        help="""Scale applied to lattice scores when computing nbest paths.
        Used only when the decoding method is fast_beam_search_nbest,
        fast_beam_search_nbest_LG, and fast_beam_search_nbest_oracle""",
    )

    parser.add_argument(
        "--simulate-streaming",
        type=str2bool,
        default=False,
        help="""Whether to simulate streaming in decoding, this is a good way to
        test a streaming model.
        """,
    )

    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=16,
        help="The chunk size for decoding (in frames after subsampling)",
    )

    parser.add_argument(
        "--left-context",
        type=int,
        default=64,
        help="left context can be seen during decoding (in frames after subsampling)",
    )
    
    parser.add_argument(
        "--res-name",
        type=str,
    )

    # optimizer related
    parser.add_argument(
        "--base-lr", type=float, default=6e-5, help="The base learning rate."
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
        "--subsampling-factor",
        type=int,
        default=320,
        help="shit0",
    )

    parser.add_argument(
        "--use-double-scores",
        type=bool,
        default=True,
        help="shit0",
    )

    parser.add_argument(
        "--warm-step",
        type=int,
        default=0,
        help="shit0",
    )

    # tta related
    parser.add_argument(
        "--num-augment",
        type=int,
        default=4,
        help="shit1",
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=10,
        help="shit1",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="shit1",
    )

    add_model_arguments(parser)
    add_rep_arguments(parser)

    return parser

from typing import Any, Dict, Optional, Tuple, Union

from torch import Tensor

from torch.nn.parallel import DistributedDataParallel as DDP

from icefall.utils import (
    AttributeDict,
    MetricsTracker,
    encode_supervisions,
    setup_logger,
    str2bool,
    save_args,
)
import warnings

def decode_one_batch(
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
) -> Dict[str, List[List[str]]]:
    """Decode one batch and return the result in a dict. The dict has the
    following format:

        - key: It indicates the setting used for decoding. For example,
               if greedy_search is used, it would be "greedy_search"
               If beam search with a beam size of 7 is used, it would be
               "beam_7"
        - value: It contains the decoding result. `len(value)` equals to
                 batch size. `value[i]` is the decoding result for the i-th
                 utterance in the given batch.
    Args:
      params:
        It's the return value of :func:`get_params`.
      model:
        The neural model.
      sp:
        The BPE model.
      batch:
        It is the return value from iterating
        `lhotse.dataset.K2SpeechRecognitionDataset`. See its documentation
        for the format of the `batch`.
      word_table:
        The word symbol table.
      decoding_graph:
        The decoding graph. Can be either a `k2.trivial_graph` or HLG, Used
        only when --decoding_method is fast_beam_search, fast_beam_search_nbest,
        fast_beam_search_nbest_oracle, and fast_beam_search_nbest_LG.
    Returns:
      Return the decoding result. See above description for the format of
      the returned dict.
    """
    device = next(model.parameters()).device
    feature = batch["inputs"]
    assert feature.ndim == 2 or feature.ndim == 3

    feature = feature.to(device)
    # at entry, feature is (N, T, C)

    supervisions = batch["supervisions"]
    #feature_lens = supervisions["num_frames"].to(device)
    if feature.ndim == 2:
        feature_lens = [] 
        for supervision in supervisions['cut']:
            try: feature_lens.append(supervision.tracks[0].cut.recording.num_samples)
            except: feature_lens.append(supervision.recording.num_samples)
        feature_lens = torch.tensor(feature_lens)

    elif feature.ndim == 3:
        feature_lens = supervisions["num_frames"].to(device)

    if params.simulate_streaming:
        feature_lens += params.left_context
        feature = torch.nn.functional.pad(
            feature,
            pad=(0, 0, 0, params.left_context),
            value=LOG_EPS,
        )
        encoder_out, encoder_out_lens, _ = model.encoder.streaming_forward(
            x=feature,
            x_lens=feature_lens,
            chunk_size=params.decode_chunk_size,
            left_context=params.left_context,
            simulate_streaming=True,
        )
    else:
        encoder_out, encoder_out_lens = model.encoder(x=feature, x_lens=feature_lens, prompt=model.prompt)
    
    hyps = []

    if params.decoding_method == "fast_beam_search":
        hyp_tokens = fast_beam_search_one_best(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "fast_beam_search_nbest_LG":
        hyp_tokens = fast_beam_search_nbest_LG(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
        )
        for hyp in hyp_tokens:
            hyps.append([word_table[i] for i in hyp])
    elif params.decoding_method == "fast_beam_search_nbest":
        hyp_tokens = fast_beam_search_nbest(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            nbest_scale=params.nbest_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "fast_beam_search_nbest_oracle":
        hyp_tokens = fast_beam_search_nbest_oracle(
            model=model,
            decoding_graph=decoding_graph,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam,
            max_contexts=params.max_contexts,
            max_states=params.max_states,
            num_paths=params.num_paths,
            ref_texts=sp.encode(supervisions["text"]),
            nbest_scale=params.nbest_scale,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "greedy_search" and params.max_sym_per_frame == 1:
        hyp_tokens = greedy_search_batch(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    elif params.decoding_method == "modified_beam_search":
        hyp_tokens = modified_beam_search(
            model=model,
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            beam=params.beam_size,
        )
        for hyp in sp.decode(hyp_tokens):
            hyps.append(hyp.split())
    else:
        batch_size = encoder_out.size(0)

        for i in range(batch_size):
            # fmt: off
            encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
            # fmt: on
            if params.decoding_method == "greedy_search":
                hyp = greedy_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    max_sym_per_frame=params.max_sym_per_frame,
                )
            elif params.decoding_method == "beam_search":
                hyp = beam_search(
                    model=model,
                    encoder_out=encoder_out_i,
                    beam=params.beam_size,
                )
            else:
                raise ValueError(
                    f"Unsupported decoding method: {params.decoding_method}"
                )
            hyps.append(sp.decode(hyp).split())

    if params.decoding_method == "greedy_search":
        return {"greedy_search": hyps}
    elif "fast_beam_search" in params.decoding_method:
        key = f"beam_{params.beam}_"
        key += f"max_contexts_{params.max_contexts}_"
        key += f"max_states_{params.max_states}"
        if "nbest" in params.decoding_method:
            key += f"_num_paths_{params.num_paths}_"
            key += f"nbest_scale_{params.nbest_scale}"
            if "LG" in params.decoding_method:
                key += f"_ngram_lm_scale_{params.ngram_lm_scale}"

        return {key: hyps}
    else:
        return {f"beam_size_{params.beam_size}": hyps}
    
def decode_and_adapt(
    params: AttributeDict,
    model: Union[nn.Module, DDP],
    optimizer: torch.optim.Optimizer,
    sp: spm.SentencePieceProcessor,
    batch: dict,
    is_training: bool,
    num_iter: int,
    ema_model: nn.Module=None,
    ema_args: dict=None,
) -> Tuple[Tensor, MetricsTracker]:
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
    texts = [text.upper() for text in texts]
    
    token_ids = sp.encode(texts, out_type=int)
    y = k2.RaggedTensor(token_ids).to(device)
    
    if len(token_ids[0]) > 0:
        model.train()
        for i in range(num_iter):
            with torch.set_grad_enabled(is_training):
                # logits : [B, T, prune_range, vocab_size]
                '''
                simple_loss, pruned_loss, ctc_output, logits = model(
                    x=feature,
                    x_lens=feature_lens,
                    y=y,
                    prune_range=params.prune_range,
                    am_scale=params.am_scale,
                    lm_scale=params.lm_scale,
                    return_logits=True,
                )
                '''
                encoder_out, encoder_out_lens = model.encoder(
                        x=feature, 
                        x_lens=feature_lens, 
                        prompt=model.prompt
                )
                batch_size = encoder_out.size(0)

                logits = None
                for i in range(batch_size):
                    encoder_out_i = encoder_out[i:i+1, :encoder_out_lens[i]]
                    hyp, logit = greedy_search(
                        model=model,
                        encoder_out=encoder_out_i,
                        max_sym_per_frame=params.max_sym_per_frame,
                        return_logits=True,
                    )
                    
                    if logits is None:
                        logits = logit.unsqueeze(0)
                    else:
                        logits = torch.cat([logits, logit.unsqueeze(0)], dim=0)
                
                probas = logits
                probas /= 2.5
                probas = torch.nn.functional.softmax(probas, dim=-1)
                #print('1', probas.size())
                probas = probas.flatten(start_dim=0, end_dim=1).contiguous()
                #print('2', probas.size())

                predicted_ids = torch.argmax(probas, dim=-1)
                #print('3', predicted_ids.size())
                non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
                #print('4', non_blank.size())
                
                #em
                log_probas = torch.log(probas + 1e-10)
                #print('5', log_probas.size())
                entropy = -(probas * log_probas).sum(-1)[non_blank] # (L)
                #print('6', entropy.size())
                probas = probas[non_blank]
                #print('7', probas.size())
                loss_em = entropy.mean(-1)
                
                #mcc
                target_entropy_weight = 1 + torch.exp(-entropy).unsqueeze(0) # (1, L)
                #print('9', target_entropy_weight.size())
                target_entropy_weight = probas.shape[0] * target_entropy_weight / torch.sum(target_entropy_weight)
                #print('10', target_entropy_weight.size())
                cov_matrix_t = probas.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(probas) # Y x W.T x Y
                #print('11', cov_matrix_t.size())

                cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
                #print('12', cov_matrix_t.size())
                loss_mcc = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / probas.shape[-1]
                loss = loss_mcc * 0.7 + loss_em * 0.3
                assert loss.requires_grad == is_training

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def decode_dataset(
    dl: torch.utils.data.DataLoader,
    params: AttributeDict,
    model: nn.Module,
    sp: spm.SentencePieceProcessor,
    word_table: Optional[k2.SymbolTable] = None,
    decoding_graph: Optional[k2.Fsa] = None,
    ema_model: nn.Module=None,
) -> Dict[str, List[Tuple[str, List[str], List[str]]]]:
    num_cuts = 0

    try:
        num_batches = len(dl)
    except TypeError:
        num_batches = "?"

    if params.decoding_method == "greedy_search":
        log_interval = 1
    else:
        log_interval = 20

    results = defaultdict(list)

    parameters = []
    parameters_name = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            parameters.append(p)
            parameters_name.append(n)

    optimizer = ScaledAdam(
        parameters,
        lr=params.base_lr,
        clipping_scale=2.0,
        parameters_names=[parameters_name],
    )
    
    for batch_idx, batch in enumerate(dl):
        model.eval()
        texts = batch["supervisions"]["text"]
        texts = [text.upper() for text in texts]

        cut_ids = [cut.id for cut in batch["supervisions"]["cut"]]

        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
        )
        
        # if tta
        # replace the supervision to pseudo labels
        pseudo_batch = deepcopy(batch)

        assert len(hyps_dict[params.decoding_method]) == 1 # shoud use the single utterance sampler
        pseudo_batch["supervisions"]["text"] = [" ".join(hyps_dict[params.decoding_method][0]).lower()] * params.num_augment

        # augment the single utterance (augmentation automatically excued in d2v model)
        pseudo_batch["inputs"] = batch["inputs"].repeat(params.num_augment, 1)
        pseudo_batch["supervisions"]["sequence_idx"] = batch["supervisions"]["sequence_idx"].repeat(params.num_augment)
        pseudo_batch["supervisions"]['cut'] = batch["supervisions"]['cut'] * params.num_augment 
        assert "start_frame" not in pseudo_batch["supervisions"].keys()
        assert "num_frames" not in pseudo_batch["supervisions"].keys()

        # model.train() is excuted in the decoder and adpt fucntion
        decode_and_adapt(
            params, 
            model, 
            optimizer, 
            sp, 
            pseudo_batch, 
            is_training=True, 
            num_iter=params.num_iter, 
            ema_model=ema_model,
            ema_args={"decoding_graph":decoding_graph, "word_table":word_table}
        )

        model.eval()
        hyps_dict = decode_one_batch(
            params=params,
            model=model,
            sp=sp,
            decoding_graph=decoding_graph,
            word_table=word_table,
            batch=batch,
        )
        #end tta
        
        for name, hyps in hyps_dict.items():
            this_batch = []
            assert len(hyps) == len(texts)
            for cut_id, hyp_words, ref_text in zip(cut_ids, hyps, texts):
                ref_words = ref_text.split()
                this_batch.append((cut_id, ref_words, hyp_words))

            results[name].extend(this_batch)

        num_cuts += len(texts)

        if batch_idx % log_interval == 0:
            batch_str = f"{batch_idx}/{num_batches}"

            logging.info(f"batch {batch_str}, cuts processed until now is {num_cuts}")
    return results


def save_results(
    params: AttributeDict,
    test_set_name: str,
    results_dict: Dict[str, List[Tuple[str, List[str], List[str]]]],
):
    test_set_wers = dict()
    for key, results in results_dict.items():
        recog_path = (
            params.res_dir / f"recogs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        results = sorted(results)
        store_transcripts(filename=recog_path, texts=results)
        logging.info(f"The transcripts are stored in {recog_path}")

        # The following prints out WERs, per-word error statistics and aligned
        # ref/hyp pairs.
        errs_filename = (
            params.res_dir / f"errs-{test_set_name}-{key}-{params.suffix}.txt"
        )
        with open(errs_filename, "w") as f:
            wer = write_error_stats(
                f, f"{test_set_name}-{key}", results, enable_log=True
            )
            test_set_wers[key] = wer

        logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])
    errs_info = (
        params.res_dir / f"wer-summary-{test_set_name}-{key}-{params.suffix}.txt"
    )
    with open(errs_info, "w") as f:
        print("settings\tWER", file=f)
        for key, val in test_set_wers:
            print("{}\t{}".format(key, val), file=f)
    
    spk = None
    wer = None
    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
        spk = str(test_set_name)
        wer = str(val)
    logging.info(s)
    with open(f'./{params.res_name}.txt', 'a') as f:
        f.write(f"{spk} {wer}\n")

from fairseq.modules import EMAModule, EMAModuleConfig
def make_ema_teacher(cfg, model):
    ema_config = EMAModuleConfig(
        ema_decay=cfg.ema_decay,
        ema_fp32=True,
    )
    skip_keys = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            skip_keys.add(n)
        else:
            logging.info(f"{n} is copied to the ema model")

    ema = EMAModule(
        model,
        ema_config,
        skip_keys=skip_keys,
    )
    return ema

@torch.no_grad()
def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    params = get_params()
    params.update(vars(args))

    assert params.decoding_method in (
        "greedy_search",
        "beam_search",
        "fast_beam_search",
        "fast_beam_search_nbest",
        "fast_beam_search_nbest_LG",
        "fast_beam_search_nbest_oracle",
        "modified_beam_search",
    )
    params.res_dir = params.exp_dir / params.decoding_method

    if params.iter > 0:
        params.suffix = f"iter-{params.iter}-avg-{params.avg}"
    else:
        params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"

    if params.simulate_streaming:
        params.suffix += f"-streaming-chunk-size-{params.decode_chunk_size}"
        params.suffix += f"-left-context-{params.left_context}"

    if "fast_beam_search" in params.decoding_method:
        params.suffix += f"-beam-{params.beam}"
        params.suffix += f"-max-contexts-{params.max_contexts}"
        params.suffix += f"-max-states-{params.max_states}"
        if "nbest" in params.decoding_method:
            params.suffix += f"-nbest-scale-{params.nbest_scale}"
            params.suffix += f"-num-paths-{params.num_paths}"
            if "LG" in params.decoding_method:
                params.suffix += f"-ngram-lm-scale-{params.ngram_lm_scale}"
    elif "beam_search" in params.decoding_method:
        params.suffix += f"-{params.decoding_method}-beam-size-{params.beam_size}"
    else:
        params.suffix += f"-context-{params.context_size}"
        params.suffix += f"-max-sym-per-frame-{params.max_sym_per_frame}"

    if params.use_averaged_model:
        params.suffix += "-use-averaged-model"

    setup_logger(f"{params.res_dir}/log-decode-{params.suffix}")
    logging.info("Decoding started")

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    logging.info(f"Device: {device}")

    sp = spm.SentencePieceProcessor()
    sp.load(params.bpe_model)

    # <blk> and <unk> are defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.unk_id = sp.piece_to_id("<unk>")
    params.vocab_size = sp.get_piece_size()

    if params.simulate_streaming:
        assert (
            params.causal_convolution
        ), "Decoding in streaming requires causal convolution"

    logging.info(params)

    logging.info("About to create model")
    model = get_transducer_model(params)

    load_checkpoint(f"{params.exp_dir}/{params.model_name}", model)

    # for tta
    parameters = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if ("bias" in n) and ("encoder.layers" in n):
                logging.info(f"{n} is free!")
                parameters.append(p)
            else:
                p.requires_grad = False
    sizes = [p.numel() for p in parameters]
    logging.info(f"total trainable parameter size : {sum(sizes)}")

    model.to(device)
    ema_model = make_ema_teacher(params, model)
    model.eval()
    ema_model.model.eval()

    if "fast_beam_search" in params.decoding_method:
        if params.decoding_method == "fast_beam_search_nbest_LG":
            lexicon = Lexicon(params.lang_dir)
            word_table = lexicon.word_table
            lg_filename = params.lang_dir / "LG.pt"
            logging.info(f"Loading {lg_filename}")
            decoding_graph = k2.Fsa.from_dict(
                torch.load(lg_filename, map_location=device)
            )
            decoding_graph.scores *= params.ngram_lm_scale
        else:
            word_table = None
            decoding_graph = k2.trivial_graph(params.vocab_size - 1, device=device)
    else:
        decoding_graph = None
        word_table = None

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    # we need cut ids to display recognition results.
    args.return_cuts = True
    args.bucketing_sampler= False

    librispeech = LibriSpeechAsrDataModule(args)

    test_clean_cuts = librispeech.userlibri_cuts(option=params.spk_id)
    test_clean_dl = librispeech.test_dataloaders(test_clean_cuts)
    test_sets = [f"{params.spk_id}"]
    test_dl = [test_clean_dl]
    
    for test_set, test_dl in zip(test_sets, test_dl):
        results_dict = decode_dataset(
            dl=test_dl,
            params=params,
            model=model,
            sp=sp,
            word_table=word_table,
            decoding_graph=decoding_graph,
            ema_model=ema_model,
        )
        
        save_results(
            params=params,
            test_set_name=test_set + str(params.spk_id),
            results_dict=results_dict,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
