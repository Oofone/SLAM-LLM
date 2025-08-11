# os
import os
import fire
import deepspeed
import random
import numpy as np
import importlib

# nn
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# opt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from slam_llm.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

# config
# from llama_recipes.configs import fsdp_config as FSDP_CONFIG
# from llama_recipes.configs import train_config as TRAIN_CONFIG
# from llama_recipes.configs import model_config as MODEL_CONFIG
# from llama_recipes.configs import log_config as LOG_CONFIG
from slam_llm.data.concatenator import ConcatDataset

# util
from slam_llm.utils import fsdp_auto_wrap_policy
from slam_llm.utils.config_utils import get_dataloader_kwargs

from slam_llm.utils.dataset_utils import get_preprocessed_dataset, load_module_from_py_file
from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.deepspeed_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
)

import sys
import logging
import wandb

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path


class SkipDataLoader:
    def __init__(self, dataloader, skip_steps):
        self.dataloader = dataloader
        self.skip_steps = skip_steps
        self._skipped = False
        self.logger = logging.getLogger()  
        self.logger.setLevel(logging.INFO)

    def __iter__(self):
        it = iter(self.dataloader)
        if not self._skipped and self.skip_steps > 0:
            self.logger(f"Skipping {self.skip_steps} batches to resume mid-epoch...")
            for _ in range(self.skip_steps):
                try:
                    next(it)
                except StopIteration:
                    break
            self._skipped = True
        return it

    def __len__(self):
        return len(self.dataloader)


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    # kwargs = to_plain_list(cfg)
    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    main(kwargs)


def main(kwargs: DictConfig):
    # Update the configuration for the training and sharding process
    # train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
    # update_config((train_config, fsdp_config, model_config, log_config), **kwargs)

    train_config, model_config, log_config, dataset_config, deepspeed_config = kwargs.train_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config, \
                                                                          kwargs.deepspeed_config
    del kwargs.train_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    
    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    setup_environ_flags(rank)

    if rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))

    # Set wandb
    if rank == 0:
        if log_config.use_wandb:
            if not os.path.exists(log_config.wandb_dir):
                os.makedirs(log_config.wandb_dir, exist_ok=True)
            wandb_config={"train_config": train_config, "model_config": model_config, "log_config": log_config}
            wandb.init(dir=log_config.wandb_dir, entity=log_config.wandb_entity_name, project=log_config.wandb_project_name,name=log_config.wandb_exp_name ,config=wandb_config)


    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)

    print("Pre-Deepspeed")
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"{name} not expected dtype; is {param.dtype}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If you are facing problem from limited memory(<=256GB), you can try to replace the above code with the following code
    # for i in range(rank):
    #     while not os.path.isfile(f".{i}.done"):
    #         pass
    # assert not os.path.isfile(f".{rank}.done"), f".{rank}.done already exists!"
    # model_factory = get_custom_model_factory(model_config, logger)
    # model, tokenizer = model_factory(train_config, model_config, **kwargs)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.half()
    # with open(f".{rank}.done", "w"):
    #     pass

    # Initialize the optimizer and learning rate scheduler
    lr = train_config.lr
    lr_encoder = train_config.lr_encoder
    lr_projector = train_config.lr_projector
    lr_llm = train_config.lr_llm
    my_optim = None

    if lr_encoder or lr_projector or lr_llm:
        logger.info("Using specific lr for each component")
        logger.info(
            "lr_encoder: {}; lr_llm (LoRA): {}; lr_projector: {}.".format(
                str(lr_encoder), str(lr_llm), str(lr_projector)))
        param_groups = []
        enc_param_names = []
        projector_param_names = []
        lora_param_names = []
        if lr_encoder:
            enc_params = []
            for n, p in model.named_parameters():
                if n.startswith("encoder.") and p.requires_grad:
                    enc_params.append(p)
                    enc_param_names.append(n)
            logger.info(f"Trainable encoder params with LR {lr_encoder}: {len(enc_param_names)}")
            param_groups.append({"params": enc_params, "lr": lr_encoder})
        if lr_projector:
            projector_params = []
            for n, p in model.named_parameters():
                if n.startswith("encoder_projector.") and p.requires_grad:
                    projector_params.append(p)
                    projector_param_names.append(n)
            logger.info(f"Trainable adapter params with LR {lr_projector}: {len(projector_param_names)}")
            param_groups.append({"params": projector_params, "lr": lr_projector})
        if lr_llm:
            lora_params = []
            for n, p in model.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    lora_params.append(p)
                    lora_param_names.append(n)
            logger.info(f"Trainable LLM LoRA params with LR {lr_llm}: {len(lora_param_names)}")
            param_groups.append({"params": lora_params, "lr": lr_llm})
        other_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in enc_param_names + projector_param_names + lora_param_names]
        if other_params:
            param_groups.append({"params": other_params, "lr": lr})
        # Default is Adam
        opt_fn = torch.optim.Adam
        if train_config.custom_optimizer == "adamw":
            opt_fn = torch.optim.AdamW
        my_optim = opt_fn(
            params=param_groups,
            weight_decay=train_config.weight_decay
        )
        logger.info("Original optimizer pre-deepspeed groups:")
        for i, g in enumerate(my_optim.param_groups):
            logger.info(f"Group {i}: lr = {g['lr']}, #params = {len(g['params'])}")
        model_engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=my_optim,
            model_parameters=param_groups,
            config=deepspeed_config,
        )
        logger.info("Original optimizer groups:")
        for i, g in enumerate(my_optim.param_groups):
            logger.info(f"Group {i}: lr = {g['lr']}, #params = {len(g['params'])}")
        logger.info("Deepspeed model.optimizer groups:")
        for i, g in enumerate(model_engine.optimizer.param_groups):
            logger.info(f"Group {i}: lr = {g['lr']}, #params = {len(g['params'])}")
    else:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_engine, _, _, _ = deepspeed.initialize(
            model=model, model_parameters=parameters, config=deepspeed_config
        )
    logger.info(f"LR Scheduler OG LRs: {model_engine.lr_scheduler.org_lrs}")

    resume_skip_steps = 0
    resume_epoch = 0
    if train_config.get("resume_deepspeed_dir", None):
        logger.info("Resuming Deepspeed Checkpoint with the following params:")
        logger.info(f"load_module_strict=False")
        logger.info(f"load_dir={train_config.get('resume_deepspeed_dir')}")
        logger.info(f"load_module_only={train_config.get('resume_deepspeed_only_module', True)}")
        logger.info(f"load_optimizer_states={train_config.get('resume_deepspeed_optimizer', False)}")
        logger.info(f"load_lr_scheduler_states={train_config.get('resume_deepspeed_lr_scheduler', False)}")
        model_engine.load_checkpoint(
            load_module_strict=False,
            load_dir=train_config.get("resume_deepspeed_dir"),
            load_module_only=train_config.get("resume_deepspeed_only_module", True),
            load_optimizer_states=train_config.get("resume_deepspeed_optimizer", False),
            load_lr_scheduler_states=train_config.get("resume_deepspeed_lr_scheduler", False))

        # Attempt restore training state
        training_state_path = os.path.join(train_config.resume_deepspeed_dir, "training_state.pt")
        if train_config.get("resume_training_state", False) and os.path.isfile(training_state_path):
            state = torch.load(training_state_path, map_location="cpu", weights_only=True)
            resume_epoch = state["epoch"]
            resume_skip_steps = state["step"] + 1  # skip *after* this step

            # Restore RNG states
            torch.set_rng_state(state["rng_state"]["torch"])
            torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
            np.random.set_state(state["rng_state"]["numpy"])
            random.setstate(state["rng_state"]["python"])

            logger.info(f"Resuming from epoch={resume_epoch}, step={resume_skip_steps}")
        else:
            logger.warning(f"No training_state.pt found in {train_config.resume_deepspeed_dir}")

    else:
        logger.info("Skipping Resume Deepspeed Checkpoint")

    print("Post-Deepspeed")
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"{name} not expected dtype; is {param.dtype}")

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # if (train_config.enable_fsdp or train_config.enable_ddp) and fsdp_config.pure_bf16:
    #     model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    # if train_config.enable_ddp:
    #     model = model.cuda(local_rank)
    #     model = DDP(model, device_ids=[local_rank],
    #                 find_unused_parameters=kwargs.get("train_conf", {}).get("find_unused_parameters", False))
    # elif not train_config.quantization:
    #     model.to(device)

    # dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    if rank == 0:
        if log_config.use_wandb:
            wandb.config.update({"dataset_config": dataset_config})
    
    # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )
    if (not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0) and train_config.batching_strategy != "dynamic":
        logger.info(f"--> Training Set Length = {len(dataset_train)}")
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="val",
    )
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0 and train_config.batching_strategy != "dynamic":
        logger.info(f"--> Validation Set Length = {len(dataset_val)}")
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    if resume_skip_steps > 0:
        train_dataloader = SkipDataLoader(train_dataloader, resume_skip_steps)

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )


    # Start the training process
    results = train(
        model_engine,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        train_config.gradient_accumulation_steps,
        train_config,
        log_config,
        local_rank,
        rank,
        custom_optimizer=my_optim,
        resume_epoch=resume_epoch,
        resume_epoch_step=resume_skip_steps,
    )
    if rank==0:
        [logger.info(f'Key: {k}, Value: {v}') for k, v in results.items()]

    if rank == 0:
        if log_config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main_hydra()