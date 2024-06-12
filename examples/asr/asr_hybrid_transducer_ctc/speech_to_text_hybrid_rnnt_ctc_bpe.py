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

"""
# Preparing the Tokenizer for the dataset
Use the `process_asr_text_tokenizer.py` script under <NEMO_ROOT>/scripts/tokenizers/ in order to prepare the tokenizer.

```sh
python <NEMO_ROOT>/scripts/tokenizers/process_asr_text_tokenizer.py \
        --manifest=<path to train manifest files, seperated by commas>
        OR
        --data_file=<path to text data, seperated by commas> \
        --data_root="<output directory>" \
        --vocab_size=<number of tokens in vocabulary> \
        --tokenizer=<"spe" or "wpe"> \
        --no_lower_case \
        --spe_type=<"unigram", "bpe", "char" or "word"> \
        --spe_character_coverage=1.0 \
        --log
```

# Training the model
```sh
python speech_to_text_hybrid_rnnt_ctc_bpe.py \
    # (Optional: --config-path=<path to dir of configs> --config-name=<name of config without .yaml>) \
    model.train_ds.manifest_filepath=<path to train manifest> \
    model.validation_ds.manifest_filepath=<path to val/test manifest> \
    model.tokenizer.dir=<path to directory of tokenizer (not full path to the vocab file!)> \
    model.tokenizer.type=<either bpe or wpe> \
    model.aux_ctc.ctc_loss_weight=0.3 \
    trainer.devices=-1 \
    trainer.max_epochs=100 \
    model.optim.name="adamw" \
    model.optim.lr=0.001 \
    model.optim.betas=[0.9,0.999] \
    model.optim.weight_decay=0.0001 \
    model.optim.sched.warmup_steps=2000
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name="<Name of experiment>" \
    exp_manager.wandb_logger_kwargs.project="<Name of project>"
```

# Fine-tune a model

For documentation on fine-tuning this model, please visit -
https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/configs.html#fine-tuning-configurations

"""

import torch, gc,os, pickle
import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecHybridRNNTCTCBPEModelEWC, EncDecHybridRNNTCTCBPEModelMAS
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import tqdm


def compute_ewc_params(model, dataloader,device):
    model.to(device)
    params, fisher = {}, {}
    nb = 0
    for e, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        cuda_batch = [x.to(device) for x in batch[:4]] + batch[4:]
        loss = model.training_step_custom(cuda_batch,e)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                if name not in params:
                    params[name] = param.clone().cpu()
                if name not in fisher:
                    fisher[name] = (param.grad.clone() ** 2).cpu()
                else:
                    fisher[name] += (param.grad.clone() ** 2).cpu()

        model.zero_grad()
        nb += 1
        # loss.detach().cpu()
        # if e == 5:
        #     del cuda_batch, model, loss
        #     break

    for name in fisher:
        fisher[name] /= nb

    return params, fisher

def compute_mas_params(model, dataloader,device):
    model.to(device)
    params, importance = {}, {}
    nb = 0
    for e, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        cuda_batch = [x.to(device) for x in batch[:4]] + batch[4:]
        loss = model.training_step_custom(cuda_batch,e)
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                if name not in params:
                    params[name] = param.clone().cpu()
                if name not in importance:
                    importance[name] = param.grad.clone().abs().cpu()
                else:
                    importance[name] += param.grad.clone().abs().cpu()

        model.zero_grad()
        nb += 1
        # loss.detach().cpu()
        # if e == 5:
        #     del cuda_batch, model, loss
        #     break

    for name in importance:
        importance[name] /= nb

    return params, importance

@hydra_runner(
    config_path="../conf/conformer/hybrid_transducer_ctc/", config_name="conformer_hybrid_transducer_ctc_bpe"
)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    # For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision("medium")
    
    cl_config = cfg.get('continual_learning_strategy',None)
    
    if cl_config is None:
        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))
        asr_model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)

        # Initialize the weights of the model from another model, if provided via config
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)
        trainer.fit(asr_model)

    elif cl_config.name == 'EWC':
        # EWC Related things
        log_dir = f'{cfg.exp_manager.explicit_log_dir}/checkpoints'
        os.makedirs(log_dir,exist_ok=True)
        if not os.path.exists(f'{log_dir}/ewc.pkl'):
            # load the previous checkpoint with the old dataloader
            prev_cfg = OmegaConf.load(cl_config.ewc_params.old_config)
            trainer = pl.Trainer(**cfg.trainer)
            # prev_cfg.model.train_ds.batch_size = 16
            
            ## model contains dataset, this means that this line has loaded all the data of the previous episode
            asr_model_old = EncDecHybridRNNTCTCBPEModel(cfg=prev_cfg.model, trainer=trainer) 

            ## load the model weights from the current config
            asr_model_old.maybe_init_from_pretrained_checkpoint(cfg)
            # asr_model.setup_optimization()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            ## computing the fisher and storing the old params
            params, fisher = compute_ewc_params(asr_model_old,asr_model_old._train_dl,device)

            ## load the old params (if applicable)
            old_param_path = f"{os.path.abspath(os.path.join(log_dir,'../../'))}/{os.path.split(cl_config.ewc_params.old_config)[-1].split('.')[0]}/checkpoints/ewc.pkl"
            if 'full_finetune' not in old_param_path:
                assert os.path.exists(old_param_path),'Old param path is required'
                
                with open(f'{old_param_path}','rb') as reader:
                    saved = pickle.load(reader)
                old_fisher = saved['fisher']
                
                # do the necessary scaling
                for name in fisher:
                    if name in old_fisher:
                        old_importance = old_fisher[name]
                        fisher[name] *= 1 - cl_config.ewc_params.alpha
                        fisher[name] += (
                            cl_config.ewc_params.alpha * old_importance
                        ) 
                
            # assert os.path.exists(old_param_path) if 'ep0' not in old_param_path,
            with open(f'{log_dir}/ewc.pkl','wb') as writer:
                pickle.dump({'params':params,'fisher':fisher},writer)

            del asr_model_old, trainer

            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            ## load the param and fisher
            with open(f'{log_dir}/ewc.pkl','rb') as reader:
                saved = pickle.load(reader)
            params, fisher = saved['params'],saved['fisher']
            
        cl_params = {
            'alpha': cl_config.ewc_params.alpha,
            'lamda': cl_config.ewc_params.lda,
            'params': params,
            'fisher': fisher 
            }

        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))
        
        asr_model = EncDecHybridRNNTCTCBPEModelEWC(cfg=cfg.model, trainer=trainer) 
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)
        
        # setting cl params
        asr_model.set_cl_params(cl_params)
        
        trainer.fit(asr_model)
        
    elif cl_config.name == 'MAS':
        # MAS Related things
        log_dir = f'{cfg.exp_manager.explicit_log_dir}/checkpoints'
        os.makedirs(log_dir,exist_ok=True)
        if not os.path.exists(f'{log_dir}/mas.pkl'):
            # load the previous checkpoint with the old dataloader
            prev_cfg = OmegaConf.load(cl_config.mas_params.old_config)
            trainer = pl.Trainer(**cfg.trainer)
            # prev_cfg.model.train_ds.batch_size = 16
            
            ## model contains dataset, this means that this line has loaded all the data of the previous episode
            asr_model_old = EncDecHybridRNNTCTCBPEModel(cfg=prev_cfg.model, trainer=trainer) 

            ## load the model weights from the current config
            asr_model_old.maybe_init_from_pretrained_checkpoint(cfg)
            # asr_model.setup_optimization()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            ## computing the mas_importance and storing the old params
            params, mas_importance = compute_mas_params(asr_model_old,asr_model_old._train_dl,device)

            ## load the old params (if applicable)
            old_param_path = f"{os.path.abspath(os.path.join(log_dir,'../../'))}/{os.path.split(cl_config.mas_params.old_config)[-1].split('.')[0]}/checkpoints/mas.pkl"
            if 'full_finetune' not in old_param_path:
                assert os.path.exists(old_param_path),'Old param path is required'
                
                with open(old_param_path,'rb') as reader:
                    saved = pickle.load(reader)
                old_mas_importance = saved['mas_importance']
                
                # do the necessary scaling
                for name in mas_importance:
                    if name in old_mas_importance:
                        old_importance = old_mas_importance[name]
                        mas_importance[name] *= 1 - cl_config.mas_params.alpha
                        mas_importance[name] += (
                            cl_config.mas_params.alpha * old_importance
                        ) 
                
            # assert os.path.exists(old_param_path) if 'ep0' not in old_param_path,
            with open(f'{log_dir}/mas.pkl','wb') as writer:
                pickle.dump({'params':params,'mas_importance':mas_importance},writer)

            del asr_model_old, trainer

            gc.collect()
            torch.cuda.empty_cache()
            
        else:
            ## load the param and mas_importance
            with open(f'{log_dir}/mas.pkl','rb') as reader:
                saved = pickle.load(reader)
            params, mas_importance = saved['params'],saved['mas_importance']
            
        cl_params = {
            'alpha': cl_config.mas_params.alpha,
            'lamda': cl_config.mas_params.lda,
            'params': params,
            'mas_importance': mas_importance 
            }

        trainer = pl.Trainer(**cfg.trainer)
        exp_manager(trainer, cfg.get("exp_manager", None))
        asr_model = EncDecHybridRNNTCTCBPEModelMAS(cfg=cfg.model, trainer=trainer) 
        asr_model.maybe_init_from_pretrained_checkpoint(cfg)
        
        # setting cl params
        asr_model.set_cl_params(cl_params)
        
        trainer.fit(asr_model)
    elif cl_config.name == 'LWF':
        pass
    else:
        raise 'Error'

    if hasattr(cfg.model, 'test_ds') and cfg.model.test_ds.manifest_filepath is not None:
        if asr_model.prepare_test(trainer):
            trainer.test(asr_model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
