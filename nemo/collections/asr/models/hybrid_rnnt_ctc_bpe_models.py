# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.data.audio_to_text_dali import AudioToBPEDALIDataset
from nemo.collections.asr.data.audio_to_text_lhotse import LhotseSpeechToTextBpeDataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.core.classes.mixins import AccessMixin
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCBPEDecoding, CTCBPEDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTBPEDecoding, RNNTBPEDecodingConfig
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs


class EncDecHybridRNNTCTCBPEModel(EncDecHybridRNNTCTCModel, ASRBPEMixin):
    """Base class for encoder decoder RNNT-based models with auxiliary CTC decoder/loss and subword tokenization."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        # Tokenizer is necessary for this model
        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.labels = ListConfig(list(vocabulary))

        with open_dict(cfg.decoder):
            cfg.decoder.vocab_size = len(vocabulary)

        with open_dict(cfg.joint):
            cfg.joint.num_classes = len(vocabulary)
            cfg.joint.vocabulary = ListConfig(list(vocabulary))
            cfg.joint.jointnet.encoder_hidden = cfg.model_defaults.enc_hidden
            cfg.joint.jointnet.pred_hidden = cfg.model_defaults.pred_hidden

        # setup auxiliary CTC decoder
        if 'aux_ctc' not in cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )

        with open_dict(cfg):
            if self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual": #CTEMO
                cfg.aux_ctc.decoder.vocabulary = ListConfig(vocabulary)
            else:
                cfg.aux_ctc.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        if cfg.aux_ctc.decoder["num_classes"] < 1:
            logging.info(
                "\nReplacing placholder number of classes ({}) with actual number of classes - {}".format(
                    cfg.aux_ctc.decoder["num_classes"], len(vocabulary)
                )
            )
            cfg.aux_ctc.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        self.cfg.decoding = self.set_decoding_type_according_to_loss(self.cfg.decoding)
        # Setup decoding object
        self.decoding = RNNTBPEDecoding(
            decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
        )

        # Multisoftmax #CTEMO
        self.language_masks = None
        if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in cfg.decoder:
            logging.info("Creating masks for multi-softmax layer.")
            self.language_masks = {}
            self.token_id_offsets = self.tokenizer.token_id_offset
            self.offset_token_ids_by_token_id = self.tokenizer.offset_token_ids_by_token_id
            for language in self.tokenizer.tokenizers_dict.keys():
                self.language_masks[language] = [(token_language == language)  for _, token_language in self.tokenizer.langs_by_token_id.items()]
                self.language_masks[language].append(True) # Insert blank token
            self.ctc_loss = CTCLoss(
                num_classes=self.ctc_decoder._num_classes // len(self.tokenizer.tokenizers_dict.keys()),
                zero_infinity=True,
                reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
            )
            # Setup RNNT Loss
            loss_name, loss_kwargs = self.extract_rnnt_loss_cfg(self.cfg.get("loss", None))
            self.loss = RNNTLoss(
                num_classes=self.ctc_decoder._num_classes // len(self.tokenizer.tokenizers_dict.keys()),
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                reduction=self.cfg.get("rnnt_reduction", "mean_batch"),
            )
            # Setup decoding object
            self.decoding = RNNTBPEDecoding(
                decoding_cfg=self.cfg.decoding, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes // len(self.tokenizer.tokenizers_dict.keys())
            )
            
            self.decoder.language_masks = self.language_masks
            self.joint.language_masks = self.language_masks
            self.joint.token_id_offsets = self.token_id_offsets
            self.joint.offset_token_ids_by_token_id = self.offset_token_ids_by_token_id
            self.ctc_decoder.language_masks = self.language_masks

        # Setup wer object
        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=0,
            use_cer=self.cfg.get('use_cer', False),
            log_prediction=self.cfg.get('log_prediction', True),
            dist_sync_on_step=True,
        )

        # Setup fused Joint step if flag is set
        if self.joint.fuse_loss_wer:
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Setup CTC decoding
        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCBPEDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg
        if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in cfg.decoder: #CTEMO
            self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes//len(self.tokenizer.tokenizers_dict.keys()))
        else:
            self.ctc_decoding = CTCBPEDecoding(self.cfg.aux_ctc.decoding, tokenizer=self.tokenizer)

        # Setup CTC WER
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "ctc"

    def _setup_dataloader_from_config(self, config: Optional[Dict]):

        if config.get("use_lhotse"):
            return get_lhotse_dataloader_from_config(
                config,
                global_rank=self.global_rank,
                world_size=self.world_size,
                dataset=LhotseSpeechToTextBpeDataset(tokenizer=self.tokenizer,),
            )

        dataset = audio_to_text_dataset.get_audio_to_text_bpe_dataset_from_config(
            config=config,
            local_rank=self.local_rank,
            global_rank=self.global_rank,
            world_size=self.world_size,
            tokenizer=self.tokenizer,
            preprocessor_cfg=self.cfg.get("preprocessor", None),
        )

        if dataset is None:
            return None

        if isinstance(dataset, AudioToBPEDALIDataset):
            # DALI Dataset implements dataloader interface
            return dataset

        shuffle = config['shuffle']
        if isinstance(dataset, torch.utils.data.IterableDataset):
            shuffle = False

        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        elif hasattr(dataset.datasets[0], 'collate_fn'):
            # support datasets that are lists of entries
            collate_fn = dataset.datasets[0].collate_fn
        else:
            # support datasets that are lists of lists
            collate_fn = dataset.datasets[0].datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """

        if 'manifest_filepath' in config:
            manifest_filepath = config['manifest_filepath']
            batch_size = config['batch_size']
        else:
            manifest_filepath = os.path.join(config['temp_dir'], 'manifest.json')
            batch_size = min(config['batch_size'], len(config['paths2audio_files']))

        dl_config = {
            'manifest_filepath': manifest_filepath,
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'channel_selector': config.get('channel_selector', None),
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        if config.get("augmentor"):
            dl_config['augmentor'] = config.get("augmentor")

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(
        self,
        new_tokenizer_dir: Union[str, DictConfig],
        new_tokenizer_type: str,
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Directory path to tokenizer or a config for a new tokenizer (if the tokenizer type is `agg`)
            new_tokenizer_type: Type of tokenizer. Can be either `agg`, `bpe` or `wpe`.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for auxiliary CTC decoding, which is optional and can be used to change the decoding type.

        Returns: None

        """
        if isinstance(new_tokenizer_dir, DictConfig):
            if new_tokenizer_type == 'agg':
                new_tokenizer_cfg = new_tokenizer_dir
            else:
                raise ValueError(
                    f'New tokenizer dir should be a string unless the tokenizer is `agg`, but this tokenizer type is: {new_tokenizer_type}'
                )
        else:
            new_tokenizer_cfg = None

        if new_tokenizer_cfg is not None:
            tokenizer_cfg = new_tokenizer_cfg
        else:
            if not os.path.isdir(new_tokenizer_dir):
                raise NotADirectoryError(
                    f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
                )

            if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
                raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

            tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        joint_config = self.joint.to_config_dict()
        new_joint_config = copy.deepcopy(joint_config)
        if self.tokenizer_type == "agg":
            new_joint_config["vocabulary"] = ListConfig(vocabulary)
        else:
            new_joint_config["vocabulary"] = ListConfig(list(vocabulary.keys()))

        new_joint_config['num_classes'] = len(vocabulary)
        del self.joint
        self.joint = EncDecHybridRNNTCTCBPEModel.from_config_dict(new_joint_config)

        decoder_config = self.decoder.to_config_dict()
        new_decoder_config = copy.deepcopy(decoder_config)
        new_decoder_config.vocab_size = len(vocabulary)
        del self.decoder
        self.decoder = EncDecHybridRNNTCTCBPEModel.from_config_dict(new_decoder_config)

        del self.loss
        self.loss = RNNTLoss(num_classes=self.joint.num_classes_with_blank - 1)

        if decoding_cfg is None:
            # Assume same decoding config as before
            decoding_cfg = self.cfg.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
        decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

        if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in self.cfg.decoder: #CTEMO
            self.decoding = RNNTBPEDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes // len(self.tokenizer.tokenizers_dict.keys())
            )
        else:
            self.decoding = RNNTBPEDecoding(
                decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
            )

        self.wer = WER(
            decoding=self.decoding,
            batch_dim_index=self.wer.batch_dim_index,
            use_cer=self.wer.use_cer,
            log_prediction=self.wer.log_prediction,
            dist_sync_on_step=True,
        )

        # Setup fused Joint step
        if self.joint.fuse_loss_wer or (
            self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
        ):
            self.joint.set_loss(self.loss)
            self.joint.set_wer(self.wer)

        # Update config
        with open_dict(self.cfg.joint):
            self.cfg.joint = new_joint_config

        with open_dict(self.cfg.decoder):
            self.cfg.decoder = new_decoder_config

        with open_dict(self.cfg.decoding):
            self.cfg.decoding = decoding_cfg

        logging.info(f"Changed tokenizer of the RNNT decoder to {self.joint.vocabulary} vocabulary.")

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            ctc_decoder_config = copy.deepcopy(self.ctc_decoder.to_config_dict())
            # sidestepping the potential overlapping tokens issue in aggregate tokenizers
            if self.tokenizer_type == "agg":
                ctc_decoder_config.vocabulary = ListConfig(vocabulary)
            else:
                ctc_decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

            decoder_num_classes = ctc_decoder_config['num_classes']
            # Override number of classes if placeholder provided
            logging.info(
                "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                    decoder_num_classes, len(vocabulary)
                )
            )
            ctc_decoder_config['num_classes'] = len(vocabulary)

            del self.ctc_decoder
            self.ctc_decoder = EncDecHybridRNNTCTCBPEModel.from_config_dict(ctc_decoder_config)
            del self.ctc_loss
            self.ctc_loss = CTCLoss(
                num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                zero_infinity=True,
                reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
            )

            if ctc_decoding_cfg is None:
                # Assume same decoding config as before
                ctc_decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            ctc_decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
            ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

            if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in self.cfg.decoder: #CTEMO
                self.ctc_decoding = CTCBPEDecoding(decoding_cfg=ctc_decoding_cfg, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes//len(self.tokenizer.tokenizers_dict.keys()))
            else:
                self.ctc_decoding = CTCBPEDecoding(decoding_cfg=ctc_decoding_cfg, tokenizer=self.tokenizer)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.cfg.aux_ctc.get('use_cer', False),
                log_prediction=self.cfg.get("log_prediction", False),
                dist_sync_on_step=True,
            )

            # Update config
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoder = ctc_decoder_config

            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

            logging.info(f"Changed tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(self, decoding_cfg: DictConfig = None, decoder_type: str = None, lang_id: str=None):
        """
        Changes decoding strategy used during RNNT decoding process.
        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having both RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
        """
        if decoder_type is None or decoder_type == 'rnnt':
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(RNNTBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)
            decoding_cfg = self.set_decoding_type_according_to_loss(decoding_cfg)

            if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in self.cfg.decoder: #CTEMO
                self.decoding = RNNTBPEDecoding(
                    decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes // len(self.tokenizer.tokenizers_dict.keys()),
                    # lang_id=lang_id
                )
            else:
                self.decoding = RNNTBPEDecoding(
                    decoding_cfg=decoding_cfg, decoder=self.decoder, joint=self.joint, tokenizer=self.tokenizer,
                )

            self.wer = WER(
                decoding=self.decoding,
                batch_dim_index=self.wer.batch_dim_index,
                use_cer=self.wer.use_cer,
                log_prediction=self.wer.log_prediction,
                dist_sync_on_step=True,
            )

            # Setup fused Joint step
            if self.joint.fuse_loss_wer or (
                self.decoding.joint_fused_batch_size is not None and self.decoding.joint_fused_batch_size > 0
            ):
                self.joint.set_loss(self.loss)
                self.joint.set_wer(self.wer)

            self.joint.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.decoding):
                self.cfg.decoding = decoding_cfg

            self.cur_decoder = "rnnt"
            logging.info(f"Changed decoding strategy of the RNNT decoder to \n{OmegaConf.to_yaml(self.cfg.decoding)}")
        
        elif decoder_type == 'ctc':
            if not hasattr(self, 'ctc_decoding'):
                raise ValueError("The model does not have the ctc_decoding module and does not support ctc decoding.")
            if decoding_cfg is None:
                # Assume same decoding config as before
                logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
                decoding_cfg = self.cfg.aux_ctc.decoding

            # Assert the decoding config with all hyper parameters
            decoding_cls = OmegaConf.structured(CTCBPEDecodingConfig)
            decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
            decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

            if (self.tokenizer_type == "agg" or self.tokenizer_type == "multilingual") and "multisoftmax" in self.cfg.decoder: #CTEMO
                self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer, blank_id=self.ctc_decoder._num_classes//len(self.tokenizer.tokenizers_dict.keys()), lang_id=lang_id)
            else:
                self.ctc_decoding = CTCBPEDecoding(decoding_cfg=decoding_cfg, tokenizer=self.tokenizer, lang_id=lang_id)

            self.ctc_wer = WER(
                decoding=self.ctc_decoding,
                use_cer=self.ctc_wer.use_cer,
                log_prediction=self.ctc_wer.log_prediction,
                dist_sync_on_step=True,
            )

            self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

            # Update config
            with open_dict(self.cfg.aux_ctc.decoding):
                self.cfg.aux_ctc.decoding = decoding_cfg

            self.cur_decoder = "ctc"
            logging.info(
                f"Changed decoding strategy of the CTC decoder to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}"
            )
        else:
            raise ValueError(f"decoder_type={decoder_type} is not supported. Supported values: [ctc,rnnt]")

    @classmethod
    def list_available_models(cls) -> List[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_en_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_de_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_it_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_it_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_it_fastconformer_hybrid_large_pc/versions/1.20.0/files/stt_it_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_es_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_hr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_hr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_hr_fastconformer_hybrid_large_pc/versions/1.21.0/files/FastConformer-Hybrid-Transducer-CTC-BPE-v256-averaged.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ua_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ua_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ua_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ua_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_pl_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_pl_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_pl_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_pl_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_by_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_by_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_by_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_by_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_ru_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_ru_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_ru_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_ru_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_fr_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_fr_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc_blend_eu",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc_blend_eu.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_multilingual_fastconformer_hybrid_large_pc",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_multilingual_fastconformer_hybrid_large_pc",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_multilingual_fastconformer_hybrid_large_pc/versions/1.21.0/files/stt_multilingual_fastconformer_hybrid_large_pc.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_80ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_80ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_80ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_80ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_480ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_480ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_480ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_480ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_1040ms",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_1040ms",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_1040ms/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_1040ms.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_fastconformer_hybrid_large_streaming_multi",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_fastconformer_hybrid_large_streaming_multi",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_hybrid_large_streaming_multi/versions/1.20.0/files/stt_en_fastconformer_hybrid_large_streaming_multi.nemo",
        )
        results.append(model)

        return results

class EncDecHybridRNNTCTCBPEModelEWC(EncDecHybridRNNTCTCBPEModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)
    
    def set_cl_params(self,cl_params):
        self.lda = cl_params['lamda']
        self.alpha = cl_params['alpha']
        self.fisher = cl_params['fisher']
        self.old_params = cl_params['params']

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if "multisoftmax" not in self.cfg.decoder: #CTEMO
            signal, signal_len, transcript, transcript_len = batch
            language_ids = None
        else:
            signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids) #CTEMO
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
                language_ids=language_ids
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                if "multisoftmax" in self.cfg.decoder:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                        lang_ids=language_ids,
                    )
                else:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                    )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )

        # EWC related changes
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None or name not in self.fisher:
                continue
            loss_value += (
                (
                    self.fisher[name].to(self.device)
                    * (self.old_params[name].to(self.device) - param) ** 2
                )
                .sum()
                # .to(self.device)
                * self.alpha
                * self.lda
            )

        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

class EncDecHybridRNNTCTCBPEModelMAS(EncDecHybridRNNTCTCBPEModel):
    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        super().__init__(cfg=cfg, trainer=trainer)
    
    def set_cl_params(self,cl_params):
        self.lda = cl_params['lamda']
        # self.alpha = cl_params['alpha']
        self.importance = cl_params['mas_importance']
        self.old_params = cl_params['params']

    def training_step(self, batch, batch_nb):
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        if "multisoftmax" not in self.cfg.decoder: #CTEMO
            signal, signal_len, transcript, transcript_len = batch
            language_ids = None
        else:
            signal, signal_len, transcript, transcript_len, sample_ids, language_ids = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=encoded, decoder_outputs=decoder, language_ids=language_ids) #CTEMO
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
                language_ids=language_ids
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded, language_ids=language_ids)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                if "multisoftmax" in self.cfg.decoder:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                        lang_ids=language_ids,
                    )
                else:
                    self.ctc_wer.update(
                        predictions=log_probs,
                        targets=transcript,
                        targets_lengths=transcript_len,
                        predictions_lengths=encoded_len,
                    )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )

        # EWC related changes
        for name, param in self.named_parameters():
            if not param.requires_grad or param.grad is None or name not in self.importance:
                continue
            loss_value += (
                (
                    self.importance[name].to(self.device)
                    * (self.old_params[name].to(self.device) - param) ** 2
                )
                .sum()
                # .to(self.device)
                * self.lda
            )

        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}