"""
Offline Distillation
We use Transformer-big as Teacher and Transformer-base as Student.
"""

import datasets
import json
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import logging
from datasets import load_dataset, dataset_dict, load_metric
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    TrainingArguments,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartModel,
    MBartConfig,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MBartTokenizer,
    MBartTokenizerFast,
    BertTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    Trainer,
)
import os
import numpy as np
from huggingface_hub import HfFolder

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# init tokenizer
teacher_tokenizer = BertTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")
student_tokenizer = BertTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")
tokenizer = BertTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")

logger = logging.getLogger(__name__)

# Set seed before initializing model.
set_seed(42)


class DistillationTrainingArguments(Seq2SeqTrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature


class DistillationTrainer(Seq2SeqTrainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(**inputs)

        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss


# define teacher model
teacher_model_checkpoint = "/data0/xp/gec/ChineseNMT/checkpoint/pretrain/wiki1000_big"
teacher_configuration = MBartConfig(
    vocab_size=50000,
    d_model=512,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=1024,
    encoder_ffn_dim=1024,
    dropout=0.3,
    activation_function='gelu'
)
teacher_model = MBartForConditionalGeneration.from_pretrained(teacher_model_checkpoint,
                                                              cache_dir="/data0/xp/gec/model")
teacher_model.resize_token_embeddings(len(teacher_tokenizer))
teacher_configuration = teacher_model.config

# define student model
student_model_checkpoint = "/data0/xp/gec/ChineseNMT/checkpoint/pretrain/wiki1000"
student_configuration = MBartConfig(
    vocab_size=80000,
    d_model=1024,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    decoder_ffn_dim=2048,
    encoder_ffn_dim=2048,
    dropout=0.3,
    activation_function='gelu'
)
student_model = MBartForConditionalGeneration.from_pretrained(student_model_checkpoint,
                                                              cache_dir="/data0/xp/gec/model")
student_model.resize_token_embeddings(len(student_tokenizer))
student_configuration = student_model.config

# Load the dataset
data_files = {}
data_path = "/data0/xp/gec/data/nlpcc2018+hsk/"
train_file = data_path + "train.json"
data_files["train"] = train_file
extension = train_file.split(".")[-1]

valid_file = data_path + "dev.json"
data_files["validation"] = valid_file

raw_datasets = load_dataset(extension, data_files=data_files)
teacher_model.resize_token_embeddings(len(tokenizer))
student_model.resize_token_embeddings(len(tokenizer))


# Preprocessing the datasets.
# We need to tokenize inputs and targets.

def flatten(example):
    return {
        "original_text": example["original_text"],
        "correct_text": example["correct_text"],
        "wrong_ids": example["wrong_ids"]
    }


dataset = raw_datasets["train"].map(flatten,
                                    remove_columns=["original_text", "correct_text"])
max_length = 128
num_beams = 5


def preprocess_function(examples):
    input = examples['original_text']
    target = examples['correct_text']
    inputs = [inp for inp in input]
    targets = [outp for outp in target]
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)
    labels = tokenizer(inputs, targets, max_length=max_length, truncation=True)

    model_inputs.setdefault('labels', 0)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = raw_datasets["train"]

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on train dataset",
)

eval_dataset = raw_datasets["validation"]
eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on validation dataset",
)

# define data_collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer
)

# define training args
training_args = DistillationTrainingArguments(
    output_dir='/data0/xp/gec/model/checkpoint/KD/wiki1000',
    num_train_epochs=30,
    weight_decay=0.01,
    predict_with_generate=True,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    fp16=True,
    learning_rate=4e-6,
    # logging & evaluation strategies
    logging_dir='/data0/xp/gec/model/checkpoint/KD/wiki1000/logs',
    logging_strategy="epoch",  # to get more information to TB
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=4,
    load_best_model_at_end=True,
    # distilation parameters
    alpha=0.5,
    temperature=4.0
)


# Training
trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()
