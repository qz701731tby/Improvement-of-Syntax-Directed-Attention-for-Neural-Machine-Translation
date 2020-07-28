from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP('/Users/qianze/graduate_design/code/stanford-corenlp-full-2016-10-31/', lang='zh')

def read_txt(input_path):
    origin_corpus = []
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            ch_sentence, en_sentence = line.strip().split('\t')[0], line.strip().split('\t')[1]
            origin_corpus.append([ch_sentence, en_sentence])

    return origin_corpus

def retokenize_filter(origin_corpus, need_num, start_num):
    new_corpus = []
    cnt = 0
    useful_cnt = 0
    for i in range(start_num, len(origin_corpus)):
        cnt += 1
        new_ch_sentence = ' '.join(nlp.word_tokenize(origin_corpus[i][0].replace(' ', '')))
        if cnt % 5000 == 0:
            print(cnt)
            print(useful_cnt)
        if filter([new_ch_sentence, origin_corpus[i][1]]):
            new_corpus.append([new_ch_sentence, origin_corpus[i][1]])
            useful_cnt += 1
        if useful_cnt == need_num:
            break
    
    return new_corpus

def filter(pair):
    ch, en = pair[0], pair[1]
    ch_list, en_list = ch.split(' '), en.split(' ')
    ch_word_num, en_word_num = len(ch_list), len(en_list)
    ch_sign_num, en_sign_num = 0, 0
    sign_set = ['.', ',', ':', '"', '?', '!', "。", "，", "？"] #标点符号

    for word in ch_list:
        if not judge_ch(word) and word not in sign_set:
            return False
        elif word in sign_set:
            ch_sign_num += 1
    
    #标点符号的数量不超过总数的30%
    if ch_sign_num >= ch_word_num * 0.3:
        return False
    
    for word in en_list:
        if not judge_en(word) and word not in sign_set:
            return False
        elif word in sign_set:
            en_sign_num += 1
    
    if en_sign_num >= en_word_num * 0.3:
        return False
    
    #中英文单词数比例不超过2
    if ch_word_num / en_word_num < 0.5 or ch_word_num / en_word_num > 2:
        return False

    return True

#判断字符串是否全为中文字符
def judge_ch(word):
    for _char in word:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def judge_en(word):
    for _char in word:
        if not _char.isalpha():
            return False
    return True
    #todo:

def write(new_corpus, output_path):
    with open(output_path, 'w') as f:
        for pair in new_corpus:
            f.write(pair[0] + '\t' + pair[1] + '\n')

need_num = 15000
input_path = '/Users/qianze/graduate_design/code/data/zh-en_20.txt'
output_path = '/Users/qianze/graduate_design/code/data/filtered_zh-en_' + str(need_num) + '.txt'

origin_corpus = read_txt(input_path)
new_corpus = retokenize_filter(origin_corpus, need_num, 0)
write(new_corpus, output_path)