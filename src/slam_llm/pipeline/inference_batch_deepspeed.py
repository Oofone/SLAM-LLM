# os
import re
import os
import fire
import deepspeed
import random
import importlib
import jiwer
from tqdm import tqdm
# nn
import torch

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.utils.data import DistributedSampler
# opt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from slam_llm.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
import torch.distributed as dist
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
import json
import logging
import wandb

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path
import evaluate


# Ensure Hugging Face metrics are loaded only once
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
squad_metric = evaluate.load("squad")


def clean_text_transcript(text: str) -> str:
    # 1. Remove XML-style tags like <s/>, <en>, </en>
    text = re.sub(r"<[^/>]+/>", "", text)       # Remove self-closing tags
    text = re.sub(r"<[^>]+>", "", text)         # Remove open/close tags

    # 2. Remove non-semantic (xxx) phrases like (ppl), (laugh)
    text = re.sub(r"\([^)]*\)", "", text)

    # 3. Replace [xxx] with xxx (keep filler content, drop brackets)
    text = re.sub(r"\[([^\]]+)\]", r"\1", text)

    # 4. Remove garbage tokens:
    #    - Alphanumeric clusters (e.g., a1b2c3, 9ppb)
    #    - Letter-symbol mixtures (e.g., s/<", x#x)
    text = re.sub(r"\b[a-zA-Z]*[\d]+[a-zA-Z\d]*\b", "", text)    # alphanum
    text = re.sub(r"\b[a-zA-Z]*[^\w\s]+[a-zA-Z]*\b", "", text)   # letter-symbol

    # 5. Remove placeholder tokens
    text = text.replace("--EMPTY--", "").replace("<unk>", "")

    # 6. Normalize whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()

    # 7. Remove remaining non-alphanumeric characters (except spaces)
    text = re.sub(r"[^a-z0-9\s]", "", text)

    return text


@hydra.main(config_name=None, version_base=None)  # strict=False 允许忽略未知参数)
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

    train_config, model_config, log_config, dataset_config = kwargs.train_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config
    name = kwargs.name
    del kwargs.train_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    del kwargs.name

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
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    deepspeed.init_distributed(
        dist_backend='nccl',    # 使用NCCL后端（GPU场景）
    )

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
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu") # FIX(MZY): put the whole model to device.
    model.to(device)
    model.eval()
    logger.info("dataset_config: {}".format(dataset_config))
    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )
    # sampler = DistributedSampler(
    #             dataset_test,
    #             rank=dist.get_rank(),
    #             num_replicas=dist.get_world_size(),
    #         )
    test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            shuffle=False,
            batch_size=train_config.val_batch_size,
            drop_last=False,
            collate_fn=dataset_test.collator,
            # sampler=sampler
            # multiprocessing_context=mp.get_context("spawn")
        )

    logger.info("=====================================")
    os.makedirs(kwargs.get('decode_log'), exist_ok=True)
    log_path = os.path.join(kwargs.get('decode_log'), "inference_log.jsonl")
    error_log_path = os.path.join(kwargs.get('decode_log'), "inference_error_log.jsonl")
    preds = []
    preds_norm = []
    targets = []
    targets_norm = []
    counter = 0
    with open(error_log_path, "w", encoding="utf-8", newline="\n", buffering=1) as log_error_writer:
        with open(log_path, "w", encoding="utf-8", newline="\n", buffering=1) as log_writer:
            with torch.no_grad():
                for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
                    for key in batch.keys():
                        batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                    model_outputs = model.generate(**batch)
                    if hasattr(model, 'tokenizer'):
                        output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
                    else:
                        output_text = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
                    for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
                        text = (text or "").strip()
                        text_norm = clean_text_transcript(text)
                        target = (target or "").strip()
                        target_norm = clean_text_transcript(target)
                        if counter % 5 == 0:
                            print(f"{key}:")
                            print("Predictions: ", text)
                            print("Targets: ", target)
                        if not target:
                            error_log = {
                                "key": key,
                                "pred": text,
                                "target": target,
                                "reason": "EMPTY_REFERENCE_AFTER_STRIP"
                            }
                            log_error_writer.write(f"{json.dumps(error_log)}\n")
                        else:
                            preds.append(text)
                            preds_norm.append(text_norm)
                            targets.append(target)
                            targets_norm.append(target_norm)
                            log = {
                                "key": key,
                                "pred": text,
                                "pred_norm": text_norm,
                                "target": target,
                                "target_norm": target_norm,
                            }
                            log_writer.write(f"{json.dumps(log)}\n")
                    counter += 1
                    if counter % 100 == 0:
                        log_writer.flush()
                        os.fsync(log_writer.fileno())
                        log_error_writer.flush()
                        os.fsync(log_error_writer.fileno())
            log_writer.flush()
            os.fsync(log_writer.fileno())
            log_error_writer.flush()
            os.fsync(log_error_writer.fileno())

    if train_config.get("metric").lower() == "wer":
        wer = jiwer.wer(targets, preds)
        wer_norm = jiwer.wer(targets_norm, preds_norm)
        logger.info("=====================================")
        logger.info(f"Done evaluation:")
        logger.info(f"WER: {wer}")
        logger.info(f"WER_Norm: {wer_norm}")
        wer_path = os.path.join(kwargs.get('decode_log'), "wer.txt")
        with open(wer_path, 'w', encoding="utf-8", newline="\n") as wer_writer:
            wer_writer.write(f"{name} WER: {wer}\n")
            wer_writer.write(f"{name} WER_Norm: {wer_norm}\n")
    elif train_config.get("metric").lower() == "sqa_all":
        # Targets and preds are lists of strings
        assert isinstance(preds_norm, list) and isinstance(targets_norm, list)

        # For BLEU, each reference must be a list of references (even if one)
        bleu_result = bleu_metric.compute(predictions=preds, references=[[ref] for ref in targets])

        # ROUGE handles single-reference and multi-reference
        rouge_result = rouge_metric.compute(predictions=preds, references=targets)

        # METEOR expects strings
        meteor_result = meteor_metric.compute(predictions=preds, references=targets)

        # F1 from squad
        squad_result = squad_metric.compute(predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(preds)],
                                            references=[{"id": str(i), "answers": {"text": [t], "answer_start": [0]}} for i, t in enumerate(targets)])

        # WER
        wer = jiwer.wer(targets, preds)
        wer_norm = jiwer.wer(targets_norm, preds_norm)

        # Logging results
        logger.info("=====================================")
        logger.info(f"Done evaluation:")
        logger.info(f"F1 (SQuAD): {squad_result['f1']:.2f}")
        logger.info(f"Exact Match (SQuAD): {squad_result['exact_match']:.2f}")
        logger.info(f"BLEU: {bleu_result['bleu']:.4f}")
        logger.info(f"METEOR: {meteor_result['meteor']:.4f}")
        logger.info(f"WER: {wer:.4f}")
        logger.info(f"WER_Norm: {wer_norm:.4f}")
        for k, v in rouge_result.items():
            logger.info(f"{k.upper()}: {v:.4f}")

        # Optionally write to file
        score_path = os.path.join(kwargs.get('decode_log'), "sqa_all_metrics.txt")
        with open(score_path, 'w', encoding='utf-8') as f:
            f.write(f"F1 (SQuAD): {squad_result['f1']:.2f}\n")
            f.write(f"Exact Match (SQuAD): {squad_result['exact_match']:.2f}\n")
            f.write(f"BLEU: {bleu_result['bleu']:.4f}\n")
            f.write(f"METEOR: {meteor_result['meteor']:.4f}\n")
            f.write(f"WER: {wer:.4f}\n")
            f.write(f"WER_Norm: {wer_norm:.4f}\n")
            for k, v in rouge_result.items():
                f.write(f"{k.upper()}: {v:.4f}\n")


if __name__ == "__main__":
    main_hydra()
