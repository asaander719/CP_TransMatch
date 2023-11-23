import torch
import torch.nn as nn
import torch.nn.functional as F

class LightGCN(nn.Module):
    def __init__(self, user_embedding, item_embedding, num_layers):
        super(LightGCN, self).__init__()
        self.num_layers = num_layers
        self.user_embedding = user_embedding
        self.item_embedding = item_embedding

    def forward(self, user_indices, item_indices):
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        for _ in range(self.num_layers):
            user_emb = torch.matmul(user_emb, self.item_embedding.weight.t())
            item_emb = torch.matmul(item_emb, self.user_embedding.weight.t())

        return user_emb, item_emb


def find_top_k_similar_users(model, k):
    user_embeddings = model.user_embedding.weight
    similarity_matrix = torch.matmul(user_embeddings, user_embeddings.t())
    # 取Top-K，不包括自己
    top_k_users = torch.topk(similarity_matrix, k=k+1, dim=1)[1][:, 1:]
    return top_k_users

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