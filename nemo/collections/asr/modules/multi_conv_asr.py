# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Set, Union

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from nemo.collections.asr.parts.submodules.jasper import (
    JasperBlock,
    MaskedConv1d,
    ParallelBlock,
    SqueezeExcite,
    init_weights,
    jasper_activations,
)
from nemo.collections.asr.parts.submodules.tdnn_attention import (
    AttentivePoolLayer,
    StatsPoolLayer,
    TDNNModule,
    TDNNSEModule,
)
from nemo.collections.asr.parts.utils import adapter_utils
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.mixins import AccessMixin, adapter_mixins
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import (
    AcousticEncodedRepresentation,
    LengthsType,
    LogitsType,
    LogprobsType,
    NeuralType,
    SpectrogramType,
    StringType,
)
from nemo.utils import logging

__all__ = ['MultiConvASRDecoder']


class MultiConvASRDecoder(NeuralModule, Exportable, adapter_mixins.AdapterModuleMixin):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
        https://arxiv.org/pdf/2005.04290.pdf
    """

    @property
    def input_types(self):
        return OrderedDict({
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            'language_ids': [NeuralType(('B'), StringType(), optional=True)],
        })

    @property
    def output_types(self):
        return OrderedDict({"logprobs": NeuralType(('B', 'T', 'D'), LogprobsType())})

    def __init__(self, feat_in, languages, num_classes_per_lang, init_mode="xavier_uniform", num_classes=None, vocabulary=None, multisoftmax=True):
        super().__init__()

        # if vocabulary is None and num_classes < 0:
        #     raise ValueError(
        #         f"Neither of the vocabulary and num_classes are set! At least one of them need to be set."
        #     )

        # if num_classes <= 0:
        #     num_classes = len(vocabulary)
        #     logging.info(f"num_classes of ConvASRDecoder is set to the size of the vocabulary: {num_classes}.")

        if vocabulary is not None:
            # if num_classes != len(vocabulary):
            #     raise ValueError(
            #         f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
            #     )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes_per_lang = []
        self.languages = languages
        for num_classes in num_classes_per_lang:
            self._num_classes_per_lang.append(num_classes + 1)
        self._num_classes = self._num_classes_per_lang[0]

        
        self.decoder_layers = {}
        for lang, num_classes in zip(self.languages, self._num_classes_per_lang):
            self.decoder_layers[lang] = torch.nn.Sequential(
                torch.nn.Conv1d(self._feat_in, num_classes, kernel_size=1, bias=True)
            )
        self.decoder_layers = torch.nn.ModuleDict(self.decoder_layers)
        self.apply(lambda x: init_weights(x, mode=init_mode))

        accepted_adapters = [adapter_utils.LINEAR_ADAPTER_CLASSPATH]
        self.set_accepted_adapter_types(accepted_adapters)

        # to change, requires running ``model.temperature = T`` explicitly
        self.temperature = 1.0
        
    @typecheck()
    def forward(self, encoder_output, language_ids):
        # Adapter module forward step
        if self.is_adapter_available():
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            encoder_output = self.forward_enabled_adapters(encoder_output)
            encoder_output = encoder_output.transpose(1, 2)  # [B, C, T]
        
        language = language_ids[0]
        if self.temperature != 1.0:
            decoder_output = self.decoder_layers[language](encoder_output).transpose(1, 2) / self.temperature
        else:
            decoder_output = self.decoder_layers[language](encoder_output).transpose(1, 2)
        
        return torch.nn.functional.log_softmax(decoder_output, dim=-1)

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    def _prepare_for_export(self, **kwargs):
        m_count = 0
        for m in self.modules():
            if type(m).__name__ == "MaskedConv1d":
                m.use_mask = False
                m_count += 1
        if m_count > 0:
            logging.warning(f"Turned off {m_count} masked convolutions")
        Exportable._prepare_for_export(self, **kwargs)

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        # Update the config with correct input dim
        cfg = self._update_adapter_cfg_input_dim(cfg)
        # Add the adapter
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self._feat_in)
        return cfg

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes
