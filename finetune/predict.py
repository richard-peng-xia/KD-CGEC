import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartModel,
    MBartConfig,
    BertTokenizerFast,
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    BertForMaskedLM,
    MBartForConditionalGeneration,
    set_seed,
)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_checkpoint = "/data0/xp/gec/ChineseNMT/checkpoint/"

tokenizer = BertTokenizerFast.from_pretrained("/data0/xp/gec/tokenizer", cache_dir="/data0/xp/gec/model")

model = MBartForConditionalGeneration.from_pretrained(model_checkpoint,
                                                      cache_dir="/data0/xp/gec/model")

# model = torch.load('finetune_csc.pth')

max_length = 128

# Load the dataset
data_files = {}

data_path = "/data0/xp/gec/data/csc/"

train_file = data_path + "train.json"
data_files["train"] = train_file
extension = train_file.split(".")[-1]

valid_file = data_path + "dev.json"
data_files["validation"] = valid_file

test_file = data_path + "test.json"
data_files["test"] = test_file

raw_datasets = load_dataset(extension, data_files=data_files)


def preprocess_function(examples):
    input = examples['original_text']
    target = examples['correct_text']
    inputs = [inp for inp in input]
    targets = [outp for outp in target]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    labels = tokenizer(inputs, targets, max_length=128, truncation=True, padding='max_length')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# train_dataset = raw_datasets["train"]
#
# train_dataset = train_dataset.map(
#     preprocess_function,
#     batched=True,
#     # num_proc=None,
#     # remove_columns=column_names,
#     # load_from_cache_file=True,
#     desc="Running tokenizer on train dataset",
# )
#
# eval_dataset = raw_datasets["validation"]
# eval_dataset = eval_dataset.map(
#     preprocess_function,
#     batched=True,
#     # num_proc=None,
#     # remove_columns=column_names,
#     # load_from_cache_file=True,
#     desc="Running tokenizer on validation dataset",
# )

predict_dataset = raw_datasets["test"]
predict_dataset = predict_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on test dataset",
)
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    # label_pad_token_id=tokenizer.pad_token_id,
)

# Definite training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/data0/xp/gec/model/checkpoint',
    learning_rate=1e-7,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=20,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
)

# Initialize our Trainer
trainer = Seq2SeqTrainer(
    model=model,
    # train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
    # predict_dataset=predict_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    args=training_args,
)

# Test
predict_results = trainer.predict(predict_dataset)
print(predict_results)
print(predict_results.predictions)
output_dir = '/data0/xp/gec/output/'
if trainer.is_world_process_zero():
    predictions = tokenizer.batch_decode(
        predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    predictions = [pred.strip() for pred in predictions]
    # output_prediction_file = os.path.join(output_dir, "test.txt")
    # with open(output_prediction_file, "w", encoding="utf-8") as writer:
    #     writer.write("\n".join(predictions))

# def correct_sentence(test_samples, model):
#     inputs = tokenizer(
#         test_samples,
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#         return_tensors="pt",
#     )
#     input_ids = inputs.input_ids.to(model.device)
#     attention_mask = inputs.attention_mask.to(model.device)
#     outputs = model.generate(input_ids, attention_mask=attention_mask)
#     output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     return outputs, output_str


# print(correct_sentence(predict_dataset, model))
