import argparse
import re, json

def section_to_clst(section):
    clst = []
    idx = 0
    sec_len = 0
    for sec in section:
        c = []
        for _ in sec:
            c.append(idx)
            idx += 1
        clst.append(c)
        sec_len += 1
    return clst, idx, sec_len


def get_hedge(input_path, output_path, task):
    file_path = '{}/{}.txt'.format(input_path, task)
    save_path = '{}/{}.hedges.jsonl'.format(output_path, task)
    fout = open(save_path, 'w')
    count = 0
    line_count = 0
    sec_len_average = 0
    with open(file_path, encoding="utf-8") as fin:
        for line in fin:
            data = json.loads(line)
            article_id = data["article_id"]
            article_text = data["article_text"]
            if len(article_text) <= 1:
                continue
            else:
                sec_clst, sen_num, sec_len = section_to_clst(data["sections"])

            dataset = {}
            dataset["id"] = article_id
            dataset["hedges"] = sec_clst
            dataset["length"] = sen_num
            dataset["section_length"] = sec_len

            fout.write(json.dumps(dataset) + '\n')
            count += sen_num
            sec_len_average += sec_len
            line_count += 1
        count = count/line_count
        sec_len_average = sec_len_average/line_count
    fin.close()
    fout.close()
    return count, sec_len_average

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing dataset for hedges')

    parser.add_argument('--input_path', type=str, default='dataset/arxiv-dataset', help='The dataset directory.')
    parser.add_argument('--output_path', type=str, default='/home/zcl/dataset/preprocessed/pubmed', help='The dataset directory.')
    parser.add_argument('--task', type=str, default='val', help='dataset [train|val|test]')

    args = parser.parse_args()

    average_len, sec_len_average = get_hedge(args.input_path, args.output_path, args.task)