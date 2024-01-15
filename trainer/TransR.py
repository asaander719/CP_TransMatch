import torch
import torch.nn as nn
import torch.nn.functional as F

class TransRModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, relation_dim, margin):
        super(TransRModel, self).__init__()

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
        self.projection_matrix = nn.Embedding(num_relations, relation_dim * embedding_dim)
        self.margin = margin

    def forward(self, head, relation, tail):
        # Embedding lookup
        head_embedding = self.entity_embeddings(head)
        tail_embedding = self.entity_embeddings(tail)
        relation_embedding = self.relation_embeddings(relation)
        projection_matrix = self.projection_matrix(relation)

        # Projection operation
        head_projection = torch.matmul(head_embedding, projection_matrix.view(-1, self.embedding_dim, self.relation_dim).transpose(1, 2))
        tail_projection = torch.matmul(tail_embedding, projection_matrix.view(-1, self.embedding_dim, self.relation_dim).transpose(1, 2))

        # Distance calculation using L1 norm
        distance = torch.sum(torch.abs(head_projection + relation_embedding - tail_projection), dim=1)

        return distance