import random

def read_words(input_path):
    lines = open(input_path, encoding='utf-8').read().strip().split('\n')
    words = []
    for i in range(0, len(lines)):
        sentence_pair = lines[i].split('\t')
        for word in sentence_pair[0].split(' '):
            if word not in words:
                words.append(word)
    
    return words

def extract_data(input_path, words, need_num):
    eval_data = []
    lines = open(input_path, encoding='utf-8').read().strip().split('\n')
    for i in range(2000, len(lines)):
        flag = True
        sentence_pair = lines[i].split('\t')
        for word in sentence_pair[0].split(' '):
            if word not in words:
                flag = False
                break
        if flag == True and sentence_pair not in eval_data:
            eval_data.append(sentence_pair)
    eval_data = random.sample(eval_data, need_num)
    return eval_data

def write_eval_data(output_path, eval_data):
    with open(output_path, 'w') as f:
        for pair in eval_data:
            f.write(pair[0] + '\t' + pair[1] + '\n')

input_path1 = '/Users/qianze/graduate_design/code/data/dataset2/train_data.txt'
input_path2 = '/Users/qianze/graduate_design/code/data/dataset2/eval_data.txt'
need_num = 1000
output_path = '/Users/qianze/graduate_design/code/data/dataset2/extracted_eval_data.txt'

words = read_words(input_path1)
eval_data = extract_data(input_path2, words, need_num)
write_eval_data(output_path, eval_data)