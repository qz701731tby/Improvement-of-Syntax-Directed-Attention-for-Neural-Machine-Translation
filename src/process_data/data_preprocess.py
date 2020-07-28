from stanfordcorenlp import StanfordCoreNLP
import os

nlp = StanfordCoreNLP('/Users/qianze/graduate_design/code/stanford-corenlp-full-2016-10-31/', lang='zh')

def read_ch_txt(input_path):
    origin_corpus = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            sentence = line.strip().replace(' ', '')
            origin_corpus.append(sentence)
    
    return origin_corpus

def re_tokenize(datas):
    new_corpus = []
    for sentence in datas:
        new_seq = nlp.word_tokenize(sentence)
        new_sentence = ' '.join(new_seq)
        new_corpus.append(new_sentence)
        
    return new_corpus

def write_new_ch_txt(new_corpus, output_path):
    with open(output_path, 'w') as f:
        for sentence in new_corpus:
            f.write(sentence + '\n')

def merge_txt(input_path1, input_path2, output_path):
    datas1 = []
    datas2 = []
    with open(input_path1, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datas1.append(line.strip())
    with open(input_path1, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datas2.append(line.strip())
    
    with open(output_path, 'w') as f:
        for i in range(len(datas1)):
            f.write(datas1[i] + '\t' + datas2[i] + '\n')

def join_txt(input_folder, output_path):
    dirs = os.listdir(input_folder)
    datas = []
    count = 0
    for file in dirs:
        print(file)
        print(count)
        with open(os.path.join(input_folder, file), 'r', encoding = 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                datas.append(line.strip())
    print(len(datas))
    with open(output_path, 'w', encoding = 'utf-8') as output:
        for data in datas:
            output.write(data+'\n')

origin_corpus = read_ch_txt(input_path)
new_corpus = re_tokenize(origin_corpus)
write_new_ch_txt(new_corpus, ch_output_path)
merge_txt(ch_txt_path, en_txt_path, merge_output_path)


input_folder = '/Users/mtdp/毕业设计/设计代码/output/'
output_path = '/Users/mtdp/毕业设计/设计代码/zh-en.txt'
join_txt(input_folder, output_path)



