# coding: utf-8
import numpy as np
#from stanfordcorenlp import StanfordCoreNLP
import torch

class dependency_tree:
    def __init__(self, sentence_pair, word_num, dependency_list):
        self.word_num = word_num
        self.dependency_list = self.filter_dependency_info(dependency_list) 
        self.sentence_pair = sentence_pair
        #self.sentence = sentence
        self.syntax_matrix = self.get_syntax_matrix()
        #self.root = dependency_list[0][2]

    #获取所有源词的句法距离
    def get_all_distance(self, position):
        visited = []
        inc_list = [position]
        all_distance_dict = {}
        tmp_dependency_list = []
        for i in range(len(self.dependency_list)):
            tmp_dependency_list.append(self.dependency_list[i])
        tmp_distance = 0
        while len(visited) < self.word_num:
            tmp_inc = []
            while len(inc_list) > 0:
                tmp_node = inc_list.pop()
                visited.append(tmp_node)
                all_distance_dict[tmp_node] = tmp_distance
                cnt = 0
                dependency_len = len(tmp_dependency_list)
                for i in range(dependency_len):
                    if tmp_dependency_list[cnt][1] == tmp_node:
                        tmp_inc.append(tmp_dependency_list[cnt][2])
                        tmp_dependency_list.pop(cnt)
                        continue
                    if tmp_dependency_list[cnt][2] == tmp_node:
                        tmp_inc.append(tmp_dependency_list[cnt][1])
                        tmp_dependency_list.pop(cnt)
                        continue
                    cnt += 1
            inc_list = tmp_inc
            tmp_distance += 1

        return all_distance_dict

    #获取限制范围内的源词句法距离
    def get_legal_distance(self, position, constraint):
        all_distance_dict = self.get_all_distance(position)
        legal_distance_dict = {}
        for key in all_distance_dict.keys():
            if all_distance_dict[key] <= constraint:
                legal_distance_dict[key] = all_distance_dict[key]
        return legal_distance_dict
    
    #获取句法指导向量
    def get_syntax_vector(self, position, constraint, sigma):
        legal_distance_dict = self.get_legal_distance(position, constraint)
        syntax_vector = [0 for i in range(self.word_num)]
        for key in legal_distance_dict.keys():
            syntax_vector[key-1] = np.exp(-legal_distance_dict[key]**2/(2*(sigma**2)))
        return syntax_vector

    def filter_dependency_info(self, dependency_list):
        info_len = len(dependency_list)
        index = 0
        for i in range(info_len):
            if 0 in dependency_list[index]:
                dependency_list.pop(index)
                index -= 1 
            index += 1
        
        return dependency_list

    #获取该句话的句法距离矩阵
    def get_syntax_matrix(self):
        #print(self.dependency_list)
        #print(self.word_num)
        syntax_matrix = [[0 for i in range(self.word_num)] for i in range(self.word_num)]
        for i in range(1, self.word_num+1):
            tmp_distance_dict = self.get_all_distance(i)
            #print(tmp_distance_dict)
            for j in range(1, self.word_num+1):
                syntax_matrix[i-1][j-1] = tmp_distance_dict[j]
        return syntax_matrix

    #根据句法距离向量生成句法指导向量
    def generate_guass_vector(self, distance_vector, constraint, sigma):
        for i in range(len(distance_vector)):
            if distance_vector[i] > constraint:
                distance_vector[i] = 0
        syntax_vector = np.exp(-distance_vector**2/(2*(sigma**2)))
        return syntax_vector

class dependency_set:
    def __init__(self, input_path):
        #self.dependency_info是depencency_tree的实例集合
        self.sentence_num, self.dependency_infos = self.generate_sentence_info(input_path)
    
    def get_nlp_info(self, input_path):
        sentence_infos = []
        nlp = StanfordCoreNLP('/Users/qianze/graduate_design/code/stanford-corenlp-full-2016-10-31/', lang='zh')
        num = 0
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                count += 1
                if count % 5000 == 0:
                    print(count)
                origin_sentence = line.strip()
                chinese_sentence = line.split('\t')[0]
                word_num = len(nlp.word_tokenize(chinese_sentence))
                dependency_info, legal_flag = self.judge_legal(word_num, nlp.dependency_parse(chinese_sentence))
                if legal_flag == False:
                    continue
                sentence_infos.append([origin_sentence, word_num, dependency_info])
                num += 1

        return num, sentence_infos

    #判断一个stanfordCoreNLP解析出来的依存结果，是否能生成完整的树
    def judge_legal(self, word_num, dependency_info):
        max_num = 0
        index = 0
        for i in range(len(dependency_info)):
            try:
                if 0 in dependency_info[index]:
                    dependency_info.pop(index)
                    index -= 1
                max_num = max(max_num, dependency_info[index][1], dependency_info[index][2])
                index += 1
            except Exception as e:
                return dependency_info, False
                print(e)
        if max_num != word_num:
            return dependency_info, False

        return dependency_info, True

    #对每一个句子生成一个dependency_tree对象，并添加至self.sentence_infos中，形成一个dependency_tree对象的队列
    def generate_sentence_info(self, input_path):
        dependency_infos = []
        sentence_num, sentence_infos = self.get_nlp_info(input_path)
        print(sentence_num)
        if sentence_num != len(sentence_infos):
            print('number is incorrect!')
            return 
        else:
            print('number is right!')
        for index in range(sentence_num):
            sentence_pair = sentence_infos[index][0]
            word_num = sentence_infos[index][1]
            dependency_list = sentence_infos[index][2]
            tmp_dependency = dependency_tree(sentence_pair, word_num, dependency_list)
            dependency_infos.append(tmp_dependency)
        
        return sentence_num, dependency_infos
    
    def write_dependency_info(self, output_path):
        with open(output_path, 'w') as f:
            for index in range(self.sentence_num):
                f.write(self.dependency_infos[index].sentence_pair)
                f.write('\n')
                f.write(str(self.dependency_infos[index].word_num))
                f.write('\n')
                f.write(str(self.dependency_infos[index].syntax_matrix))
                f.write('\n')
                f.write('\n')
    
    #def read_syntax_matrix(input_path):

if __name__ == "__main__":
    '''test_dependency_list = [('ROOT', 0, 5), ('det', 3, 1), ('compound:nn', 3, 2), ('nsubj', 5, 3), ('advmod', 5, 4), ('amod', 9, 6), ('mark', 6, 7), ('compound:nn', 9, 8), ('dobj', 5, 9)]
    test_dependency_tree = dependency_tree(test_dependency_list, 9)
    print(test_dependency_tree.get_all_distance(5))
    print(test_dependency_tree.get_legal_distance(5,3))
    print(test_dependency_tree.get_syntax_vector(5, 2, 1))
    print(test_dependency_tree.syntax_matrix)'''
    input_path = '/Users/qianze/graduate_design/code/data/zh-en_20.txt'
    output_path = '/Users/qianze/graduate_design/code/data/syntax_info.txt'
    dependency_tree_set = dependency_set(input_path)
    dependency_tree_set.write_dependency_info(output_path)