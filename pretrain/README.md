# KD-CGEC

This repository stores the code for paper `arXiv:2208.00351` [Chinese grammatical error correction based on knowledge distillation](https://arxiv.org/abs/2208.00351) based on [HuggingFace](https://huggingface.co/)🤗.

## Data

Pretrain: [Wikipedia](https://dumps.wikimedia.org/zhwiki/)

Finetune: [NLPCC2018 Dataset](http://tcci.ccf.org.cn/conference/2018/dldoc/trainingdata02.tar.gz) [HSK 动态作文语料](http://ordostsg.org.cn:1080/KCMS/detail/detail.aspx?filename=YYJX200901008&dbcode=CJFR&dbname=)

## Data Process

### Word Segmentation

- **Tool**：[sentencepiece](https://github.com/google/sentencepiece)
- **Preprocess**：Run `./pretrain/data/get_corpus.py` , in which we will get bilingual data to build our training, dev and testing set.  The data will be saved in `corpus.src` and `corpus.trg`, with one sentence in each line.
- **Word segmentation model training**: Run `./pretrain/tokenizer/tokenize.py`, in which the *sentencepiece.SentencePieceTrainer.Train()* mothed is called to train our word segmentation model. After training, `src.model`，`src.vocab`，`trg.model` and `trg.vocab` will be saved in `./pretrain/tokenizer`.  `.model` is the word segmentation model we need and `.vocab` is the vocabulary.

## Transformer Model

We use the open-source code [transformer-pytorch](http://nlp.seas.harvard.edu/2018/04/03/attention.html) developmented by Harvard.

## Requirements

This repo was tested on Python 3.6+ and PyTorch 1.5.1. The main requirements are:

- tqdm
- pytorch >= 1.5.1
- sacrebleu >= 1.4.14
- sentencepiece >= 0.1.94

To get the environment settled quickly, run:

```
pip install -r requirements.txt
```

## Usage

### Pretrain

Hyperparameters can be modified in `./pretrain/config.py`.

- This code supports MultiGPU training. You should modify `device_id` list in  `config.py` and `os.environ['CUDA_VISIBLE_DEVICES']` in `main.py` to use your own GPUs.

To start training, please run:

```
python ./pretrain/main.py
```

The training log is saved in `./pretrain/experiment/train.log`, and the translation results of testing dataset is in `./pretrain/experiment/output.txt`.

### Finetune

To start training, please run:

```
python ./finetune/train.py
```

### Distillation

```
python ./finetune/distillation.py
```

## Reference

```
@article{xia2022chinese,
  title={Chinese grammatical error correction based on knowledge distillation},
  author={Xia, Peng and Zhou, Yuechi and Zhang, Ziyan and Tang, Zecheng and Li, Juntao},
  journal={arXiv preprint arXiv:2208.00351},
  year={2022}
}
```

For any other problems you meet when doing your own project, welcome to issuing or sending emails to me 😊~
