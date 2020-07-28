from nltk.translate.bleu_score import sentence_bleu

def read_txt(input_path1, input_path2):
    references = []
    candidates = []
    with open(input_path1, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            references.append(lines[i].strip().split('\t')[1].split(' ')) 
    with open(input_path2, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            candidates.append(lines[i].strip().split(' ')) 
    return references, candidates

def BLEU_judge(references, candidates):
    total_score = 0
    cnt = 0
    for i in range(len(references)):
        score = sentence_bleu([references[i]], candidates[i], weights=[1,0,0,0])
        if score > 0.5:
            print(references[i])
            print(candidates[i])
            cnt += 1
        total_score += score
    print(cnt)
    print(total_score/len(references))

input_path1 = '/Users/qianze/graduate_design/code/data/dataset3/syntax_info_eval_data.txt'
input_path2 = '/Users/qianze/graduate_design/code/data/dataset3/global attention/result.txt'
references, candidates = read_txt(input_path1, input_path2)
print(len(references), len(candidates))
BLEU_judge(references, candidates)