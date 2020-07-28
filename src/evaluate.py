# coding: utf-8
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from model import EncoderRNN_GRU, EncoderRNN_LSTM, DecoderRNN, AttnDecoderRNN, SyntaxAttnDecoderRNN_1, SyntaxAttnDecoderRNN_2,SyntaxAttnDecoderRNN_3
from train import tensorsFromPair, tensorFromSentence, indexesFromSentence
from utils import prepareData
from dependency_tree import dependency_tree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import sentence_bleu
import random

SOS_token = 0
EOS_token = 1

def read_eval_data(input_path):
    lines = open(input_path, encoding='utf-8').read().strip().split('\n')
    pairs = []
    for i in range(0, len(lines), 2):
        sentence_pair = lines[i].split('\t')
        syntax_matrix = eval(lines[i+1])
        pairs.append([sentence_pair, syntax_matrix])
    return pairs

def evaluateIters(encoder, decoder, pairs, max_length, output_path):
    total_score = 0
    results = []
    #pairs = random.sample(pairs, 20)
    for pair in pairs:
        reference = [pair[0][1].split(' ')]
        input_sentence = pair[0][0]
        syntax_matrix = pair[1]
        output_words, attentions = evaluateSingle(encoder, decoder, input_sentence, syntax_matrix, max_length)
        candidate = output_words[:-1]
        results.append(' '.join(candidate))
        #print(reference)
        #print(candidate)
        score = sentence_bleu(reference, candidate, weights=[0.5,0.5,0,0])
        total_score += score
    average_score = total_score / len(pairs)
    print('average_score =', str(average_score))
    with open(output_path, 'w') as f:
        for result in results:
            f.write(result)
            f.write('\n')
            
    return average_score

def evaluateSingle(encoder, decoder, sentence, syntax_matrix, max_length):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        h_t_tilde = torch.zeros(decoder_hidden[0].shape).to(device)

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, len(syntax_matrix), syntax_matrix)
            decoder_output, decoder_hidden, h_t_tilde, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, h_t_tilde, len(syntax_matrix), syntax_matrix)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate(encoder, decoder, sentence, max_length):
    with torch.no_grad():
        word_tokenize = nlp.word_tokenize(sentence)
        syntax_info = nlp.dependency_parse(sentence)
        sentence = ' '.join(word_tokenize)
        #print(syntax_info)
        #print(len(word_tokenize))
        tmp_dependency_tree = dependency_tree(sentence, len(word_tokenize), syntax_info)
        syntax_matrix = tmp_dependency_tree.get_syntax_matrix()
        #print(syntax_matrix)

        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        h_t_tilde = decoder_hidden[0]

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, len(syntax_matrix), syntax_matrix)
            #decoder_output, decoder_hidden, h_t_tilde, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, h_t_tilde, len(syntax_matrix), syntax_matrix)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence, max_length):
    output_words, attentions = evaluate(encoder1, syntax_attn_decoder1, input_sentence, max_length)
    #output_words, attentions = evaluate(encoder2, syntax_attn_decoder2, input_sentence, max_length)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/'
    eval_data_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/syntax_info_eval_data.txt'
    input_lang, output_lang, pairs, max_length = prepareData(data_path, 'zh', 'en', False)
    MAX_LENGTH = max_length + 1
    hidden_size = 512
    
    encoder_model_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/encoder.pt'
    #attn_decoder_model_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset1/path2 lstm/attn_decoder.pt'
    syntax_decoder_model_path ='/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/syntax_attn_decoder.pt'
    output_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/result.txt'
    eval_data = read_eval_data(eval_data_path)

    #encoder1 = EncoderRNN_GRU(input_lang.n_words, hidden_size).to(device)
    encoder2 = EncoderRNN_LSTM(input_lang.n_words, hidden_size).to(device)
    #attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    #syntax_attn_decoder1 = SyntaxAttnDecoderRNN_1(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    syntax_attn_decoder2 = SyntaxAttnDecoderRNN_2(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    #syntax_attn_decoder3 = SyntaxAttnDecoderRNN_3(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    encoder2.load_state_dict(torch.load(encoder_model_path))
    #syntax_attn_decoder1.load_state_dict(torch.load(syntax_decoder_model_path))
    syntax_attn_decoder2.load_state_dict(torch.load(syntax_decoder_model_path))
    #attn_decoder1.load_state_dict(torch.load(attn_decoder_model_path))

    #output_words, attentions = evaluate(encoder1, syntax_attn_decoder1, '', MAX_LENGTH)
    #evaluateAndShowAttention('新加坡国际电影节在新加坡举行.', MAX_LENGTH)

    score = evaluateIters(encoder2, syntax_attn_decoder2, eval_data, MAX_LENGTH, output_path)
    #score = evaluateIters(encoder1, attn_decoder1, eval_data, MAX_LENGTH, output_path)
    print(score)