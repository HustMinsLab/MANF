# coding: UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# module for Word Pair   
class WP(nn.Module):
    def __init__(self, args):
        super(WP, self).__init__()
        
        self.fc = nn.Linear(args.h_dim, args.h_dim*4)
        self.we = nn.Linear(args.h_dim, 1, bias = False)    
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0.5)

        # parameters
        self.len_arg = args.len_arg
        self.in_dim = args.in_dim
        
        # initial        
        nn.init.xavier_normal_(self.fc.weight, 1)
        nn.init.xavier_normal_(self.we.weight, 1)
        
    def forward(self, arg_1, length_x1, arg_2, length_x2):
        
        batch_size = len(length_x1)
        matrix = torch.zeros([batch_size, self.len_arg*self.len_arg, self.in_dim]).cuda() 
        for batch in range(batch_size):
            word_1 = arg_1[batch][:length_x1[batch]].expand(length_x2[batch], length_x1[batch], self.in_dim)
            word_2 = arg_2[batch][:length_x2[batch]].expand(length_x1[batch], length_x2[batch], self.in_dim)
            word_1 = word_1.permute(1, 0, 2)
            offset = word_1 - word_2 
            offset_flat = offset.reshape(length_x1[batch]*length_x2[batch], self.in_dim) 
            pad = torch.zeros(self.len_arg*self.len_arg - length_x1[batch]*length_x2[batch], self.in_dim).cuda()
            offset_pad = torch.cat((offset_flat, pad), dim = 0) 
            matrix[batch] = offset_pad
                        
        We_ = self.we(matrix) 
        mask_we = We_.masked_fill(torch.abs(We_) < 1e-7, -1e7)
        alpha = F.softmax(mask_we, dim=1) 
        matrix_weight = matrix * alpha 
        matrix_weight = torch.sum(matrix_weight, dim=1) 
        
        out = self.fc(matrix_weight)
        out = self.tanh(out)
        out = self.drop(out)
        
        return out

# module for AttBiLSTM
class Att_BiLSTM(nn.Module):
    def __init__(self, args):
        super(Att_BiLSTM, self).__init__()
        
        self.lstm_1 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.lstm_2 = nn.LSTM(args.in_dim, args.h_dim, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.softmax = nn.Softmax(dim=1)

        self.We_arg1 = nn.Linear(args.h_dim *2, 1, bias = False)
        self.We_arg2 = nn.Linear(args.h_dim *2, 1, bias = False)
        
        self.Wg_Att = nn.Linear(args.h_dim *4, args.h_dim *4, bias = False)
        self.Ug_Att = nn.Linear(args.h_dim *4, args.h_dim *4, bias = False)
        self.Att_drop = nn.Dropout(0.5)

        # initial
        nn.init.xavier_normal_(self.We_arg1.weight, 1)
        nn.init.xavier_normal_(self.We_arg2.weight, 1)
        nn.init.xavier_normal_(self.Wg_Att.weight, 1)
        nn.init.xavier_normal_(self.Ug_Att.weight, 1)

    def forward(self, arg_1, length_x1, arg_2, length_x2):
        
        # BiLSTM package
        length_arg1, idx_1 = length_x1.sort(0, descending=True)
        _, un_idx_1 = torch.sort(idx_1, dim=0)
        arg_sort_1 = arg_1[idx_1]
        packed_input_arg1 = pack_padded_sequence(input=arg_sort_1, lengths=length_arg1, batch_first=True)

        length_arg2, idx_2 = length_x2.sort(0, descending=True)
        _, un_idx_2 = torch.sort(idx_2, dim=0)
        arg_sort_2 = arg_2[idx_2]
        packed_input_arg2 = pack_padded_sequence(input=arg_sort_2, lengths=length_arg2, batch_first=True)
                
        # lstm
        packed_output_arg1, _ = self.lstm_1(packed_input_arg1)  
        packed_output_arg2, _ = self.lstm_2(packed_input_arg2)
        
        # batch_unpack
        unpacked_1, _ = pad_packed_sequence(packed_output_arg1, batch_first=True)
        unpacked_2, _ = pad_packed_sequence(packed_output_arg2, batch_first=True)

        out_a1 = torch.index_select(unpacked_1, 0, un_idx_1.cuda())
        out_a2 = torch.index_select(unpacked_2, 0, un_idx_2.cuda()) 
        
        # Interacvite Attention
        out_a1_sum = torch.sum(out_a1, dim=1)  
        out_a2_sum = torch.sum(out_a2, dim=1) 
        alpha_IA_a1 = torch.matmul(out_a1, out_a2_sum.unsqueeze(-1)) 
        alpha_IA_a2 = torch.matmul(out_a2, out_a1_sum.unsqueeze(-1)) 
        alpha_IA_a1.masked_fill_(torch.abs(alpha_IA_a1) < 1e-7, -1e7)
        alpha_IA_a2.masked_fill_(torch.abs(alpha_IA_a2) < 1e-7, -1e7)
        alpha_IA_a1, alpha_IA_a2 = self.softmax(alpha_IA_a1), self.softmax(alpha_IA_a2)  
        out_IA_a1 = torch.squeeze(torch.matmul(alpha_IA_a1.transpose(1,2), out_a1)) 
        out_IA_a2 = torch.squeeze(torch.matmul(alpha_IA_a2.transpose(1,2), out_a2)) 

        # Self-Attention 
        We_a1 = self.We_arg1(out_a1) 
        We_a2 = self.We_arg2(out_a2) 
        We_a1.masked_fill_(torch.abs(We_a1) < 1e-7, -1e7)
        We_a2.masked_fill_(torch.abs(We_a2) < 1e-7, -1e7)
        alpha_SA_a1 = F.softmax(We_a1, dim=1) 
        alpha_SA_a2 = F.softmax(We_a2, dim=1) 
        out_SA_a1 = torch.squeeze(torch.matmul(alpha_SA_a1.transpose(1,2), out_a1)) 
        out_SA_a2 = torch.squeeze(torch.matmul(alpha_SA_a2.transpose(1,2), out_a2)) 
        
        # Cat
        out_SA_cat = torch.cat((out_SA_a1, out_SA_a2), dim=1) 
        out_IA_cat = torch.cat((out_IA_a1, out_IA_a2), dim=1) 
        
        # fusion gate Self-Attention and Interacvite Attention
        gate = torch.sigmoid(self.Wg_Att(out_IA_cat)+self.Ug_Att(out_SA_cat)) 
        out = torch.mul(gate, out_IA_cat)+torch.mul(1.0-gate, out_SA_cat) 
        out = self.Att_drop(out)      
        
        return out
    
# the whole module
class BiLSTM_DA_WP_GATED(nn.Module):
    def __init__(self, args):
        super(BiLSTM_DA_WP_GATED, self).__init__()
        
        self.WP = WP(args)
        self.Att_BiLSTM = Att_BiLSTM(args)

        # fusion gate
        self.Wg = nn.Linear(args.h_dim *4, args.h_dim *4, bias = False)
        self.Ug = nn.Linear(args.h_dim *4, args.h_dim *4, bias = False)
        self.f_drop = nn.Dropout(0.2)

        # classifier
        self.classifier = nn.Linear(args.h_dim*4, args.num_class)

        # initial
        nn.init.xavier_normal_(self.Wg.weight, 1)
        nn.init.xavier_normal_(self.Ug.weight, 1)
        nn.init.xavier_normal_(self.classifier.weight, 1)
        
    def forward(self, arg_1, length_x1, arg_2, length_x2):
        
        # word pair
        out_WP = self.WP(arg_1, length_x1, arg_2, length_x2) 

        # Att-BiLSTM
        out_Att = self.Att_BiLSTM(arg_1, length_x1, arg_2, length_x2) 
       
        # fusion gate
        gate = torch.sigmoid(self.Wg(out_Att)+self.Ug(out_WP)) 
        out = torch.mul(gate, out_Att)+torch.mul(1.0-gate, out_WP) 
        out = self.f_drop(out)
        
        # classification
        out = self.classifier(out)
        out = F.log_softmax(out, dim=1)

        return out