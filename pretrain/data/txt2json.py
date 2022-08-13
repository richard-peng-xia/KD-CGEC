import json


def txt2json(src, trg):
    # 读取文件
    with open(src, 'r', encoding="utf-8") as f1:
        with open(trg, 'r', encoding="utf-8") as f2:
            result = []
            lst_src = []
            lst_trg = []
            # 逐行读取
            for line1 in f1:
                line1 = line1.split('\n')[0]
                lst_src.append(line1)
                for line2 in f2:
                    line2 = line2.split('\n')[0]
                    lst_trg.append(line2)
            for i in range(len(lst_src)):
                item = [lst_src[i], lst_trg[i]]
                result.append(item)
                # result.append("\n")

    with open('/data0/xp/gec/ChineseNMT/data/json/train_nlpcc.json', 'w', encoding='utf-8') as dump_f:
        dump_f.write('[')
        for s in result:
            json.dump(s, dump_f, ensure_ascii=False)
            # if result.index(s) != len(result)-1:
            dump_f.write(",")
            dump_f.write("\n")
        dump_f.write(']')
        # for s in result:
        #     dump_f.writelines(s)
        # json.dump(s, dump_f, ensure_ascii=False)
        # dump_f.write(',')


txt2json("/data0/xp/zh/all_data/nlpcc2018+hsk/train.src", "/data0/xp/zh/all_data/nlpcc2018+hsk/train.trg")
