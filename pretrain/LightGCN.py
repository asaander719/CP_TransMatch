import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, user_embedding, item_embedding, pretrain_layer_num, adj_UJ, adj_IJ, device):
        super(LightGCN, self).__init__()
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding
        self.num_layers = pretrain_layer_num
        self.device = device
        self.adj_IJ = adj_IJ.to(self.device)
        self.adj_UJ = adj_UJ.to(self.device)

    def forward(self, Us, Is, Js):
        user_emb = self.user_embedding(Us)
        Is_emb = self.item_embedding(Is)
        Js_emb = self.item_embedding(Js)
        #使用完整的邻接矩阵，但只更新minibatch中的emb.
        for _ in range(self.num_layers):
            user_emb_temp = torch.sparse.mm(self.adj_UJ, self.item_embedding.weight)
            item_emb_temp = torch.sparse.mm(self.adj_IJ.t(), self.user_embedding.weight)

            final_user_emb = user_emb + user_emb_temp(Us)
            final_Is_emb = Is_emb + item_emb_temp(Is)
            final_Js_emb = Js_emb + item_emb_temp(Js)

        return final_user_emb, final_Is_emb, final_Js_emb


class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_emb, top_k_emb):
        # Query: 原始用户, Key/Value: Top-K相似用户
        q = self.query(user_emb).unsqueeze(1)  # [batch_size, 1, embedding_dim]
        k = self.key(top_k_emb)                # [batch_size, k, embedding_dim]
        v = self.value(top_k_emb)              # [batch_size, k, embedding_dim]

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (embedding_dim ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)

        weighted_sum = torch.matmul(attention, v).squeeze(1)
        return weighted_sum

# class SelfAttention(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(SelfAttention, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.query = nn.Linear(input_dim, output_dim)
#         self.key = nn.Linear(input_dim, output_dim)
#         self.value = nn.Linear(input_dim, output_dim)

#     def forward(self, x, mask = None):
#         batch_size, seq_length, input_d = x.size()

#         q = self.query(x) #(batch_size, seq_length, output_d)
#         k = self.key(x)
#         v = self.value(x)

#         attention_score = torch.matmul(q, k.transpose(1,2)) / (self.output_dim ** 0.5) #(batch_size, seq_length, seq_length)
#         if mask is not None:
#             attention_score = attention_score.masked_fill(mask ==0, float('-inf'))
        
#         attention_weight = torch.softmax(attention_score, dim=-1) #(batch_size, seq_length, seq_length)
#         weighted_values = torch.matmul(attention_weight, v) ##(batch_size, seq_length, output_d)

#         output = torch.mean(weighted_values, dim=1) #(bs, out_dim) #average pooling
#         return output