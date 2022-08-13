def format_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f1:
        lines = f1.readlines()
        lst = []
        for row in lines:
            row = row.split('\n')[0]
            lst.append(row)
    with open(output_file, "w", encoding="utf-8") as f2:
        f2.writelines(lst)

format_json('/data0/xp/gec/ChineseNMT/data/json/train_wiki_400.json',"/data0/xp/gec/ChineseNMT/data/json/train_wiki400.json")