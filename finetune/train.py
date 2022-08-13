import datasets
import json
import torch
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import logging
from datasets import load_dataset, dataset_dict
from transformers.trainer_utils import get_last_checkpoint
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

os.environ['CUDA_VISIBLE_DEVICES'] = '7'


model_checkpoint = "/data0/xp/gec/ChineseNMT/checkpoint/pretrain/wiki600_reverse"

# tokenizer = MBartTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")
tokenizer = BertTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")

configuration = MBartConfig(
    vocab_size=50000,
    d_model=512,
    encoder_layers=6,
    decoder_layers=6,
    encoder_attention_heads=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=2048,
    encoder_ffn_dim=2048,
    dropout=0.3,
    activation_function='gelu'
)
model = MBartForConditionalGeneration.from_pretrained(model_checkpoint,
                                                      cache_dir="/data0/xp/gec/model")
model.resize_token_embeddings(len(tokenizer))
configuration = model.config


logger = logging.getLogger(__name__)

# Set seed before initializing model.
set_seed(42)

# Load the dataset
data_files = {}

data_path = "/data0/xp/gec/data/nlpcc2018+hsk/"

train_file = data_path + "train_reverse.json"
data_files["train"] = train_file
extension = train_file.split(".")[-1]

valid_file = data_path + "dev_reverse.json"
data_files["validation"] = valid_file

# test_file = data_path + "test.json"
# data_files["test"] = test_file

raw_datasets = load_dataset(extension, data_files=data_files)
model.resize_token_embeddings(len(tokenizer))


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

# predict_dataset = raw_datasets["test"]
# predict_dataset = predict_dataset.map(
#     preprocess_function,
#     batched=True,
#     # num_proc=None,
#     # remove_columns=column_names,
#     # load_from_cache_file=True,
#     desc="Running tokenizer on test dataset",
# )

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
)

# Definite training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/data0/xp/gec/model/checkpoint/wiki600_reverse',
    learning_rate=1e-7,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=30,
    weight_decay=0.01,
    save_total_limit=4,
    predict_with_generate=True,
)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
)

# Training
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = (len(train_dataset))
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluation
results = {}
logger.info("*** Evaluate ***")

metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
max_eval_samples = len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)