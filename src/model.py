# coding: utf-8

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers=1):
        super(EncoderRNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = lstm_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class EncoderRNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_layers=1):
        super(EncoderRNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = lstm_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=False)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.lstm(output, hidden)
        return output, hidden
    
    def initHidden(self):
        h0 = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, 1, self.hidden_size).to(device)
        return h0, c0

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1).squeeze(0)
        attn_energies = self.concat_score(h, encoder_outputs)
        #print(F.softmax(attn_energies, dim=0))
        return F.softmax(attn_energies, dim=0)

    def concat_score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 1)))
        #v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        v = self.v.unsqueeze(1)
        energy = torch.mm(energy, v)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]
    
    def dot_score(self, hidden, encoder_outputs):
        energy = (hidden.mul(encoder_outputs)).sum(1)
        #print(energy)
        return energy

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #bmm矩阵乘法
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        
        #torch.cat参数为1，按照列拼接
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#ht−1 → at → ct → ht
class SyntaxAttnDecoderRNN_1(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(SyntaxAttnDecoderRNN_1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding_size = hidden_size
        n_pt_weights = hidden_size

        self.p_t_dense = nn.Linear(self.embedding_size + self.hidden_size, n_pt_weights, bias=False)
        self.p_t_dot = nn.Linear(n_pt_weights, 1, bias=False)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attention = Attention(self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    #根据句法距离向量生成句法指导向量
    def generate_guass_vector(self, distance_vector, max_length, constraint=4, sigma=2):
        for i in range(len(distance_vector)):
            if distance_vector[i] > constraint:
                distance_vector[i] = float("inf")
        for i in range(max_length-len(distance_vector)):
            distance_vector.append(float("inf"))
        distance_vector = np.array(distance_vector)
        syntax_vector = np.exp(-distance_vector**2/(2*(sigma**2)))
        return torch.from_numpy(syntax_vector)

    def forward(self, input, hidden, encoder_outputs, s_length, syntax_matrix):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #todo:自适应距离限制，输入hidden和input，输出一个限制
        #constraint = 
        context_embedding = torch.cat((embedded.squeeze(0), hidden[0]), dim=-1)
        #尚未确定hidden的size
        #p_t = s_length * torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(context_embedding)))).squeeze().type(torch.uint8) //之前的写法
        p_t = round(s_length * torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(context_embedding))))[0][0].item())
        #print(p_t)

        #融入句法知识指导向量:
        syntax_distance_vector = syntax_matrix[p_t]
        #print(syntax_distance_vector)
        syntax_vector = torch.Tensor(self.generate_guass_vector(syntax_distance_vector, self.max_length).type(torch.float)).to(device)

        attn_weights = self.attention(hidden[0][0], encoder_outputs)
        #attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)   

        attn_weights = torch.mul(attn_weights.squeeze(0), syntax_vector).unsqueeze(0)
        #归一化
        attn_weights = attn_weights / attn_weights.sum(1)
        
        #bmm矩阵乘法, attn_applied是语境向量
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        #torch.cat参数为1，按照列拼接
        output = torch.cat((embedded[0], context_vector[0]), 1) #[1, 1, 2*hidden_size]
        output = self.attn_combine(output).unsqueeze(0) #[1, 1, hidden_size]

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1) #[1, output_size]
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

#ht → at → ct → ～ht
class SyntaxAttnDecoderRNN_2(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, lstm_layers=1, dropout_p=0.1):
        super(SyntaxAttnDecoderRNN_2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding_size = hidden_size
        self.n_layers = lstm_layers
        self.batch_size = 1
        n_pt_weights = hidden_size

        self.p_t_dense = nn.Linear(self.hidden_size, n_pt_weights, bias=False)
        self.p_t_dot = nn.Linear(n_pt_weights, 1, bias=False)
        self.attention = Attention(self.hidden_size)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.gru = nn.GRU(self.embedding_size + self.hidden_size, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size + self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.n_layers,
                            bidirectional=False)
    
    #根据句法距离向量生成句法指导向量
    def generate_guass_vector(self, distance_vector, max_length, constraint=4, sigma=2):
        for i in range(len(distance_vector)):
            if distance_vector[i] > constraint:
                distance_vector[i] = float("inf")
        for i in range(max_length-len(distance_vector)):
            distance_vector.append(float("inf"))
        distance_vector = np.array(distance_vector)
        syntax_vector = np.exp(-distance_vector**2/(2*(sigma**2)))
        return torch.from_numpy(syntax_vector)

    def forward(self, input, hidden, encoder_outputs, h_t_tilde, s_length, syntax_matrix):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        context_embedding = torch.cat((embedded, h_t_tilde), dim=-1)

        out, hidden = self.lstm(context_embedding, hidden)

        h_t = hidden[0][0]

        p_t = round(s_length * torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(h_t))))[0].item())
        #print(p_t)

        #在这里融入句法知识
        syntax_distance_vector = syntax_matrix[p_t]
        syntax_vector = torch.Tensor(self.generate_guass_vector(syntax_distance_vector, self.max_length).type(torch.float)).to(device)

        attn_weights = self.attention(h_t, encoder_outputs)  
        attn_weights = torch.mul(attn_weights.squeeze(0), syntax_vector).unsqueeze(0)
        #归一化
        attn_weights = attn_weights / attn_weights.sum(1)
        #print(attn_weights)
        #print(encoder_outputs)
        
        #bmm矩阵乘法, attn_applied是语境向量
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        h_t_tilde = torch.tanh(self.attn_combine(torch.cat((context_vector, h_t.view(1, 1, -1)), dim=-1))) 
        #print(h_t_tilde)

        #print(self.out(h_t_tilde))
        output = F.log_softmax(self.out(h_t_tilde), dim=-1).squeeze(0)
        return output, hidden, h_t_tilde, attn_weights

    '''def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    '''
    def initHidden(self):
        lstm_init_h = nn.Parameter(nn.init.xavier_uniform(
            torch.Tensor(self.n_layers, self.batch_size, self.embedding_size).type(torch.FloatTensor)),
                                   requires_grad=True)
        lstm_init_c = nn.Parameter(nn.init.xavier_uniform(
            torch.Tensor(self.n_layers, self.batch_size, self.embedding_size).type(torch.FloatTensor)),
                                   requires_grad=True)
        return (lstm_init_h, lstm_init_c)

#ht → at → ct → ～ht
class SyntaxAttnDecoderRNN_3(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(SyntaxAttnDecoderRNN_3, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding_size = hidden_size
        n_pt_weights = hidden_size

        self.p_t_dense = nn.Linear(self.hidden_size, n_pt_weights, bias=False)
        self.p_t_dot = nn.Linear(n_pt_weights, 1, bias=False)

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attention = Attention(self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    #根据句法距离向量生成句法指导向量
    def generate_guass_vector(self, distance_vector, max_length, constraint=4, sigma=2):
        for i in range(len(distance_vector)):
            if distance_vector[i] > constraint:
                distance_vector[i] = float("inf")
        for i in range(max_length-len(distance_vector)):
            distance_vector.append(float("inf"))
        distance_vector = np.array(distance_vector)
        syntax_vector = np.exp(-distance_vector**2/(2*(sigma**2)))
        return torch.from_numpy(syntax_vector)

    def forward(self, input, hidden, encoder_outputs, h_t_tilde, s_length, syntax_matrix):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        #print(h_t_tilde.shape)
        #print(embedded.shape)
        context_embedding = torch.cat((embedded, h_t_tilde), dim=-1)

        out, hidden = self.gru(context_embedding, hidden)

        h_t = hidden[0][0]
        #尚未确定hidden的size
        #print(h_t)
        #print(torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(h_t)))))
        p_t = round(s_length * torch.sigmoid(self.p_t_dot(torch.tanh(self.p_t_dense(h_t))))[0].item())
        #print(p_t)

        #在这里融入句法知识
        syntax_distance_vector = syntax_matrix[p_t]
        syntax_vector = torch.Tensor(self.generate_guass_vector(syntax_distance_vector, self.max_length).type(torch.float)).to(device)

        '''print("h_t:")
        print(h_t)
        print("encoder_outputs:")
        print(encoder_outputs)'''
        attn_weights = self.attention(h_t, encoder_outputs)  
        #print(attn_weights)
        attn_weights = torch.mul(attn_weights.squeeze(0), syntax_vector).unsqueeze(0)
        #attention权重归一化
        attn_weights = attn_weights / attn_weights.sum(1)
        #print(attn_weights)
        
        #bmm矩阵乘法, attn_applied是语境向量
        context_vector = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        #print(context_vector)

        h_t_tilde = torch.tanh(self.attn_combine(torch.cat((context_vector, h_t.view(1, 1, -1)), dim=-1))) 

        #print(self.out(h_t_tilde))
        output = F.log_softmax(self.out(h_t_tilde), dim=-1).squeeze(0)
        return output, hidden, h_t_tilde, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)