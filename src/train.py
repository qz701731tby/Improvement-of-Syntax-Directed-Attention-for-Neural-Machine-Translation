# coding: utf-8

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random
from model import EncoderRNN_GRU, EncoderRNN_LSTM, DecoderRNN, AttnDecoderRNN, SyntaxAttnDecoderRNN_1, SyntaxAttnDecoderRNN_2,SyntaxAttnDecoderRNN_3
from utils import prepareData
from dependency_tree import dependency_tree
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 0

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    #print(pair[0][0])
    #print(pair[0][1])
    input_tensor = tensorFromSentence(input_lang, pair[0][0])
    target_tensor = tensorFromSentence(output_lang, pair[0][1])
    #print(input_tensor, target_tensor)
    return (input_tensor, target_tensor), pair[1]

def label2onehot(label, batch_size, class_num):
    label = torch.LongTensor(label)
    y_one_hot = torch.zeros(batch_size,class_num).scatter_(1,label,1).type(torch.LongTensor)
    return y_one_hot

teacher_forcing_ratio = 0.5

def train(input_tensor, target_tensor, syntax_matrix, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    #input_tensor和target_tensor转化为one-hot编码
    #input_tensor = label2onehot(input_tensor, input_length, input_lang.n_words)
    #target_tensor = label2onehot(target_tensor, target_length, output_lang.n_words)
    loss = 0
    
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    h_t_tilde = torch.zeros(decoder_hidden[0].shape).to(device)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, len(syntax_matrix), syntax_matrix)
            decoder_output, decoder_hidden, h_t_tilde, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, h_t_tilde, len(syntax_matrix), syntax_matrix)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, len(syntax_matrix), syntax_matrix)    
            decoder_output, decoder_hidden, h_t_tilde, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, h_t_tilde, len(syntax_matrix), syntax_matrix)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def ShowandSavePlot(points, fig_path):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(fig_path)

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_losses = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    encoder_scheduler = lr_scheduler.StepLR(encoder_optimizer, step_size=10000, gamma=0.8)
    decoder_scheduler = lr_scheduler.StepLR(decoder_optimizer, step_size=10000, gamma=0.8)

    #获取训练数据
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    begin_time = time.time()
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0][0]
        target_tensor = training_pair[0][1]
        #获取该句话的syntax_matrix
        syntax_matrix = training_pair[1]

        try:
            loss = train(input_tensor, target_tensor, syntax_matrix, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH)
            print_loss_total += loss
            plot_loss_total += loss
        except ValueError:
            print(iter)
            #print(input_tensor)
            #print([input_lang.index2word[index] for index in input_tensor])
        encoder_scheduler.step()
        decoder_scheduler.step()
        if iter % 10000 == 0:
            print(encoder_scheduler.get_last_lr())
            
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_losses.append(print_loss_avg)
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        if iter >= 80000:
            if print_losses[-1] > print_losses[-2] and print_losses[-2] > print_losses[-3]:
                break
    end_time = time.time()
    run_time = end_time-begin_time
    print ('该训练运行时间：',run_time)
    fig_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/loss_dataset3_path2_lstm'
    ShowandSavePlot(plot_losses, fig_path)

if __name__ == "__main__":
    data_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/'
    input_lang, output_lang, pairs, max_length = prepareData(data_path, 'zh', 'en', False)
    MAX_LENGTH = max_length + 1
    hidden_size = 512
    #encoder1 = EncoderRNN_GRU(input_lang.n_words, hidden_size).to(device) #gru
    encoder2 = EncoderRNN_LSTM(input_lang.n_words, hidden_size).to(device) #lstm
    print(output_lang.n_words)
    print(MAX_LENGTH)
    #attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    #syntax_attn_decoder1 = SyntaxAttnDecoderRNN_1(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    syntax_attn_decoder2 = SyntaxAttnDecoderRNN_2(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)
    #syntax_attn_decoder3 = SyntaxAttnDecoderRNN_3(hidden_size, output_lang.n_words, MAX_LENGTH, dropout_p=0.1).to(device)

    #trainIters(encoder1, attn_decoder1, 120000, print_every=2000, plot_every=1000)
    trainIters(encoder2, syntax_attn_decoder2, 120000, print_every=2000, plot_every=1000)
    #trainIters(encoder1, syntax_attn_decoder3, 100000, print_every=2000, plot_every=1000)

    encoder_model_path =  '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/encoder.pt'
    syntax_attn_decoder_model_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset3/path2 lstm/syntax_attn_decoder.pt'
    #attn_decoder_model_path = '/content/drive/My Drive/colab qz_seq2seq/data/dataset1/normal global attention/attn_decoder.pt'
    torch.save(encoder2.state_dict(),encoder_model_path)
    torch.save(syntax_attn_decoder2.state_dict(), syntax_attn_decoder_model_path)
    #torch.save(attn_decoder1.state_dict(), attn_decoder_model_path)

