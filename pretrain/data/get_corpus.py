import json

if __name__ == "__main__":
    files = ['train_nlpcc', 'dev_nlpcc']
    src_path = '/data0/xp/gec/ChineseNMT/data/train_nlpcc.src'
    trg_path = '/data0/xp/gec/ChineseNMT/data/train_nlpcc.trg'
    src_lines = []
    trg_lines = []

    for file in files:
        corpus = json.load(open('/data0/xp/gec/ChineseNMT/data/json/' + file + '.json', 'r', encoding="utf-8"))
        for item in corpus:
            src_lines.append(item[0] + '\n')
            trg_lines.append(item[1] + '\n')

    with open(src_path, "w") as fch:
        fch.writelines(src_lines)

    with open(trg_path, "w") as fen:
        fen.writelines(trg_lines)

    # lines of Chinese: 252777
    print("lines of source: ", len(src_lines))
    # lines of English: 252777
    print("lines of target: ", len(trg_lines))
    print("-------- Get Corpus ! --------")
