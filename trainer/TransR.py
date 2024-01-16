import torch
from torch.nn import *
import torch.nn as nn
from torch.optim import Adam
from torch.nn.init import uniform_, normal_
from torch.nn import functional as F
import pdb
from abc import abstractmethod
import numpy as np

# class TransRModel(nn.Module):
#     def __init__(self, num_entities, num_relations, embedding_dim, relation_dim, margin):
#         super(TransRModel, self).__init__()

#         self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
#         self.relation_embeddings = nn.Embedding(num_relations, relation_dim)
#         self.projection_matrix = nn.Embedding(num_relations, relation_dim * embedding_dim)
#         self.margin = margin

#     def forward(self, head, relation, tail):
#         # Embedding lookup
#         head_embedding = self.entity_embeddings(head)
#         tail_embedding = self.entity_embeddings(tail)
#         relation_embedding = self.relation_embeddings(relation)
#         projection_matrix = self.projection_matrix(relation)

#         # Projection operation
#         head_projection = torch.matmul(head_embedding, projection_matrix.view(-1, self.embedding_dim, self.relation_dim).transpose(1, 2))
#         tail_projection = torch.matmul(tail_embedding, projection_matrix.view(-1, self.embedding_dim, self.relation_dim).transpose(1, 2))

#         # Distance calculation using L1 norm
#         distance = torch.sum(torch.abs(head_projection + relation_embedding - tail_projection), dim=1)

#         return distance


class TransR(Module):
    def __init__(self, conf, visual_features=None, item_cate=None):
        super(TransR, self).__init__()
        self.visual_features = visual_features
        self.hidden_dim = conf["hidden_dim"]
        self.user_num = conf["user_num"]
        self.item_num = conf["item_num"]
        self.batch_size = conf["batch_size"]
        self.device = conf["device"]
        self.score_type = conf["score_type"]
        self.pretrain_mode = conf['pretrain_mode']
        self.use_pretrain = conf['use_pretrain']

        self.userEmb = F.normalize(torch.normal(mean=torch.zeros(self.user_num + 1, self.hidden_dim), std=1/(self.hidden_dim)**0.5), p=2, dim=-1)
        self.itemEmb = F.normalize(torch.normal(mean=torch.zeros(self.item_num + 1, self.hidden_dim), std=1/(self.hidden_dim)**0.5), p=2, dim=-1)
        self.itemB = torch.zeros([self.item_num + 1, 1])

        # self.relationEmb = F.normalize(torch.normal(mean=torch.zeros(self.item_num + 1, self.hidden_dim* self.hidden_dim), std=1/(self.hidden_dim* self.hidden_dim)**0.5), p=2, dim=-1)
        
        self.u_embeddings_l = nn.Embedding.from_pretrained(self.userEmb, freeze=False, padding_idx=self.user_num)
        self.i_bias_l = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.i_embeddings_i = nn.Embedding.from_pretrained(self.itemEmb, freeze=False, padding_idx=self.item_num)
        # self.projection_matrix = nn.Embedding.from_pretrained(self.relationEmb, freeze=False, padding_idx=self.user_num) #(num_relations, relation_dim * embedding_dim)
        self.projection_matrix = nn.Embedding(self.user_num + 1, self.hidden_dim * self.hidden_dim) #(num_relations, relation_dim * embedding_dim)
        nn.init.xavier_uniform_(self.projection_matrix.weight.data)
        if self.score_type == "mlp":
            self.layer = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
            self.scorer = nn.Linear(self.hidden_dim, 1)   

        self.visual_nn_comp = Sequential(
            Linear(conf["visual_feature_dim"], self.hidden_dim),
            nn.Sigmoid())

        self.visual_nn_comp[0].apply(lambda module: normal_(module.weight.data,mean=0,std=1/(self.hidden_dim)**0.5))
        self.visual_nn_comp[0].apply(lambda module: normal_(module.bias.data,mean=0,std=1/(self.hidden_dim)**0.5))

        self.visual_nn_per = Sequential(
            Linear(conf["visual_feature_dim"], self.hidden_dim),
            nn.Sigmoid())
        self.visual_nn_per[0].apply(lambda module: uniform_(module.weight.data,0,0.001))
        self.visual_nn_per[0].apply(lambda module: uniform_(module.bias.data,0,0.001))
        self.i_bias_v = nn.Embedding.from_pretrained(self.itemB, freeze=False, padding_idx=self.item_num)
        self.u_embeddings_v = nn.Embedding.from_pretrained(self.userEmb, freeze=False, padding_idx=self.user_num)
        # self.projection_matrix_v = nn.Embedding.from_pretrained(self.relationEmb, freeze=False, padding_idx=self.user_num)
        self.projection_matrix_v = nn.Embedding(self.user_num + 1, self.hidden_dim * self.hidden_dim) 
        nn.init.xavier_uniform_(self.projection_matrix_v.weight.data)

        # self.T = nn.Parameter(torch.zeros(self.hidden_dim))
        # experiments shows that T (TransRec) dosen't helpful 

        # self.margin = nn.Parameter(torch.tensor(1e-10)) 
        # self.margin = 1e-10

    def transE_predict(self, u_rep, i_rep, j_rep, j_bias):
        '''
            u_rep: The embedding representation of a user,
            i_rep: The embedding representation of positive bottom.
            j_rep: The embedding representation of a negative bottom.
            j_bias: A bias term associated with the negative sample j_rep. 
                    Subtracted from sum, to adjust the score based on some intrinsic property of the negative sample.
        '''
        pred = j_bias - torch.sum(torch.pow((u_rep + i_rep - j_rep), 2), -1, keepdim=True)
        # computes the element-wise square of the difference, output a single score for each triple.
        # experiment shows that j_bias is quite effective, and ||Mrh+r-Mrt|| is effective than h*r*t (DistMult).
        return pred.squeeze(-1)

    def forward(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2] 
        Ks = batch[3] 

        U_latent = self.u_embeddings_l(Us)
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 
        I_latent = self.i_embeddings_i(Is)
        J_latent = self.i_embeddings_i(Js)
        K_latent = self.i_embeddings_i(Ks)

        if self.score_type == "mlp":
            edge_pos_rep = torch.cat([U_latent, I_latent, J_latent], dim=-1)
            edge_neg_rep = torch.cat([U_latent, I_latent, K_latent], dim=-1)
            edge_pos_rep = self.layer(edge_pos_rep)  # [bs, -1, emb_dim]
            edge_pos_rep = F.relu(edge_pos_rep)
            edge_neg_rep = self.layer(edge_neg_rep)  # [bs, -1, emb_dim]
            edge_neg_rep = F.relu(edge_neg_rep)
            R_j = self.scorer(edge_pos_rep).squeeze(-1)
            R_k = self.scorer(edge_neg_rep).squeeze(-1)
            
        elif self.score_type == "transE":
            projection_matrix = self.projection_matrix(Us).view(Us.size(0), self.hidden_dim, self.hidden_dim).transpose(1,2)

            I_latent = torch.matmul(I_latent.unsqueeze(1), projection_matrix).squeeze(1)
            J_latent = torch.matmul(J_latent.unsqueeze(1), projection_matrix).squeeze(1)
            K_latent = torch.matmul(K_latent.unsqueeze(1), projection_matrix).squeeze(1)

            J_bias_l = self.i_bias_l(Js)
            K_bias_l = self.i_bias_l(Ks)
            R_j = self.transE_predict(U_latent, I_latent, J_latent, J_bias_l)
            R_k = self.transE_predict(U_latent, I_latent, K_latent, K_bias_l)
        
        J_bias_v = self.i_bias_v(Js)
        K_bias_v = self.i_bias_v(Ks)
     
        U_visual = self.u_embeddings_v(Us)
        vis_I = self.visual_features[Is]
        vis_J = self.visual_features[Js]
        vis_K = self.visual_features[Ks]
        I_visual = self.visual_nn_comp(vis_I) #bs, hidden_dim
        J_visual = self.visual_nn_comp(vis_J)
        K_visual = self.visual_nn_comp(vis_K)
        if self.score_type == "mlp":
            pass
        elif self.score_type == "transE":
            projection_matrix_v = self.projection_matrix_v(Us).view(Us.size(0), self.hidden_dim, self.hidden_dim).transpose(1,2)

            I_visual = torch.matmul(I_visual.unsqueeze(1), projection_matrix_v).squeeze(1)
            J_visual = torch.matmul(J_visual.unsqueeze(1), projection_matrix_v).squeeze(1)
            K_visual = torch.matmul(K_visual.unsqueeze(1), projection_matrix_v).squeeze(1)

            R_j_v = self.transE_predict(U_visual, I_visual, J_visual, J_bias_v)
            R_k_v = self.transE_predict(U_visual, I_visual, K_visual, K_bias_v)
            
        R_j += R_j_v
        R_k += R_k_v
        loss = bpr_loss(R_j, R_k)
        # loss = -torch.log(self.margin + torch.sigmoid(R_j - R_k)).mean()
        # 只有当R_j比R_k的score至少大出margin值时，loss才会更小,以防止negative sample和positive sample过于close
        # but only useful when those two are close, not effective if most of samples are not close
        if self.pretrain_mode: 
            return loss 
        else:
            return R_j, R_k

    def inference(self, batch):
        Us = batch[0]
        Is = batch[1]
        Js = batch[2] 
        Ks = batch[3]
        J_list = torch.cat([Js.unsqueeze(1), Ks], dim=-1)
        j_num = J_list.size(1)
        Us_exp = Us.unsqueeze(1).expand(-1, j_num) #bs, j_num
        Is_exp= Is.unsqueeze(1).expand(-1, j_num)
        J_bias_l = self.i_bias_l(J_list)
        U_latent = self.u_embeddings_l(Us_exp) 
            # T = self.T.expand_as(U_latent)  # [B H]
            # U_latent = U_latent + T 

        I_latent_ii = self.i_embeddings_i(Is_exp) #[bs, 2, hd]
        Js_latent_ii = self.i_embeddings_i(J_list) 
        if self.score_type == "mlp":
            edge_rep = torch.cat([U_latent, I_latent_ii, Js_latent_ii], dim=-1)
            edge_rep = self.layer(edge_rep)  # [bs, -1, emb_dim]
            edge_rep = F.relu(edge_rep)
            scores = self.scorer(edge_rep).squeeze(-1)
        elif self.score_type == "transE": 
            projection_matrix = self.projection_matrix(Us_exp).view(Us_exp.size(0), Us_exp.size(1), self.hidden_dim, self.hidden_dim).transpose(2,3) 
            #torch.Size([bs, 2, hd*hd]) -> torch.Size([bs, 2, hd, hd]) 
           
            I_latent_ii = torch.matmul(I_latent_ii.unsqueeze(2), projection_matrix).squeeze(2)#torch.Size([1024, 2, 32]) torch.Size([1024, 2, 32, 32])
            Js_latent_ii = torch.matmul(Js_latent_ii.unsqueeze(2), projection_matrix).squeeze(2)

            J_bias_l = self.i_bias_l(J_list)
            scores = self.transE_predict(U_latent, I_latent_ii, Js_latent_ii, J_bias_l)
       
        J_bias_v = self.i_bias_v(J_list)

        U_visual = self.u_embeddings_v(Us_exp)
        vis_I = self.visual_features[Is_exp]
        vis_Js = self.visual_features[J_list]
        I_visual_ii = self.visual_nn_comp(vis_I) #bs, hidden_dim
        Js_visual_ii = self.visual_nn_comp(vis_Js)#bs, j_num, hidden_dim

        if self.score_type == "mlp":
            scores = self.scorer(edge_rep_v).squeeze(-1)
        elif self.score_type == "transE":
            projection_matrix_v = self.projection_matrix_v(Us_exp).view(Us_exp.size(0), Us_exp.size(1), self.hidden_dim, self.hidden_dim).transpose(2,3) 
            #torch.Size([bs, 2, hd*hd]) -> torch.Size([bs, 2, hd, hd])

            I_visual_ii = torch.matmul(I_visual_ii.unsqueeze(2), projection_matrix_v).squeeze(2)
            Js_visual_ii = torch.matmul(Js_visual_ii.unsqueeze(2), projection_matrix_v).squeeze(2)

            J_bias_l = self.i_bias_l(J_list)
            self.vis_scores = self.transE_predict(U_visual, I_visual_ii, Js_visual_ii, J_bias_v)

        scores += self.vis_scores
        return scores

def bpr_loss(pos_score, neg_score):
    loss = - F.logsigmoid(pos_score - neg_score).mean()
    # # loss = torch.mean(loss)
    # loss = torch.mean((- F.logsigmoid(pos_score - neg_score)).sum())
    
    return loss