import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RAMPLayer(nn.Module):
    def __init__(self, dim_in_ent, dim_out_ent, dim_in_rel, dim_out_rel, num_ent, num_rel, \
                 phi = 'LeakyReLU', rho = 'Identity', psi = 'Identity', aggr_method = 'sum'):
        super(RAMPLayer, self).__init__()

        self.dim_in_ent = dim_in_ent
        self.dim_out_ent = dim_out_ent
        self.dim_in_rel = dim_in_rel
        self.dim_out_rel = dim_out_rel

        self.phi = self.str2act(phi)
        self.rho = self.str2act(rho)
        self.psi = self.str2act(psi)

        self.num_ent = num_ent
        self.num_rel = num_rel
        self.aggr_method = aggr_method

        self.res = nn.Linear(dim_in_ent, dim_out_ent, bias = False)

        self.proj_msg_ent = nn.Linear(dim_in_ent, num_rel * dim_out_ent, bias = False)
        self.proj_msg_rel = nn.Linear(dim_in_rel, num_rel * dim_out_ent, bias = False)

        self.proj_rel = nn.Linear(dim_in_rel, dim_out_rel, bias = False)

        self.param_init()
    
    def param_init(self):
        nn.init.xavier_normal_(self.res.weight, gain = nn.init.calculate_gain('relu'))
        
        nn.init.normal_(self.proj_msg_ent.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(self.dim_in_ent + self.dim_out_ent)))
        nn.init.normal_(self.proj_msg_rel.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(self.dim_in_rel + self.dim_out_ent)))
        nn.init.xavier_normal_(self.proj_rel.weight, gain = nn.init.calculate_gain('relu'))

    def str2act(self, string):
        if string == 'LeakyReLU':
            return nn.LeakyReLU()
        elif string == 'Identity':
            return nn.Identity()
        else:
            raise NotImplementedError

    def forward(self, emb_ent, emb_rel, triplets):
        h = triplets[:,0]
        r = triplets[:,1]
        t = triplets[:,2]

        proj_ent = self.proj_msg_ent(self.psi(emb_ent)).reshape((self.num_ent, self.num_rel, -1))
        proj_rel = self.proj_msg_rel(self.psi(emb_rel)).reshape((self.num_rel, self.num_rel, -1))
        triplet_msg = proj_ent[h, r] + proj_rel[r, r]

        if self.aggr_method == 'mean':
            freq = torch.bincount(t, minlength = self.num_ent)
            div_coeff = torch.where(freq == 0, 1, freq)
        elif self.aggr_method == 'sum':
            div_coeff = torch.ones(self.num_ent).cuda()
        else:
            raise NotImplementedError
        div_coeff = torch.unsqueeze(div_coeff[t], dim = 1)
        upd = self.res(emb_ent).index_add(dim = 0, index = t, source = self.rho(torch.div(triplet_msg, div_coeff)))
        
        return self.phi(upd), self.proj_rel(emb_rel)

class ReED(nn.Module):
    def __init__(self, hid_ent, hid_rel, dim_final, num_RAMPLayer, num_ent, num_rel,\
                 phi = 'LeakyReLU', rho = 'Identity', psi = 'Identity', \
                 aggr_method = 'sum', decoder = 'Semantic_Matching'):
        super(ReED, self).__init__()

        self.num_RAMPLayer = num_RAMPLayer
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.hid_ent = hid_ent
        self.hid_rel = hid_rel
        self.dim_final = dim_final
        layers = []
        projs_rel = []
        dim_in_ent = num_ent
        dim_in_rel = num_rel
        dim_out_ent = num_ent
        dim_out_rel = num_rel
        for l in range(num_RAMPLayer):
            dim_out_ent = hid_ent
            dim_out_rel = hid_rel
            layers.append(RAMPLayer(dim_in_ent = dim_in_ent, dim_out_ent = dim_out_ent, dim_in_rel = dim_in_rel, dim_out_rel = dim_out_rel, \
                                    num_ent = num_ent, num_rel = num_rel, phi = phi, rho = rho, psi = psi, aggr_method = aggr_method))
            dim_in_ent = dim_out_ent
            dim_in_rel = dim_out_rel
        self.layers = nn.ModuleList(layers)

        self.decoder_name = decoder

        if decoder == 'Semantic_Matching':
            self.decoder_pos = nn.Linear(dim_out_ent, num_rel * dim_out_ent, bias = False)
            self.decoder_neg = nn.Linear(dim_out_ent, num_rel * dim_out_ent, bias = False)

        elif decoder == 'Translational_Distance':
            self.decoder_head_pos = nn.Linear(dim_out_ent, num_rel * dim_final, bias = False)
            self.decoder_rel_pos = nn.Linear(dim_out_rel, num_rel * dim_final, bias = False)
            self.decoder_tail_pos = nn.Linear(dim_out_ent, num_rel * dim_final, bias = False)
            self.decoder_head_neg = nn.Linear(dim_out_ent, num_rel * dim_final, bias = False)
            self.decoder_rel_neg = nn.Linear(dim_out_rel, num_rel * dim_final, bias = False)
            self.decoder_tail_neg = nn.Linear(dim_out_ent, num_rel * dim_final, bias = False)
            
        else:
            raise NotImplementedError
        self.param_init()
    
    def param_init(self):
        if self.decoder_name == 'Semantic_Matching':
            dim_first = self.decoder_pos.weight.shape[0]
            dim_second = self.decoder_pos.weight.shape[1]//self.num_rel
            nn.init.normal_(self.decoder_pos.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_first + dim_second)))
            nn.init.normal_(self.decoder_neg.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_first + dim_second)))

        elif self.decoder_name == 'Translational_Distance':
            dim_ent = self.decoder_head_pos.weight.shape[0]
            dim_rel = self.decoder_rel_pos.weight.shape[0]
            nn.init.normal_(self.decoder_head_pos.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_ent + self.dim_final)))
            nn.init.normal_(self.decoder_rel_pos.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_rel + self.dim_final)))
            nn.init.normal_(self.decoder_tail_pos.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_ent + self.dim_final)))
            nn.init.normal_(self.decoder_head_neg.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_ent + self.dim_final)))
            nn.init.normal_(self.decoder_rel_neg.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_rel + self.dim_final)))
            nn.init.normal_(self.decoder_tail_neg.weight, std = nn.init.calculate_gain('relu')*math.sqrt(2/(dim_ent + self.dim_final)))

        else:
            raise NotImplementedError
        
    def forward(self, triplets):
        emb_ent = torch.diag(torch.ones(self.num_ent, requires_grad = False)).cuda()
        emb_rel = torch.diag(torch.ones(self.num_rel, requires_grad = False)).cuda()
        for layer in self.layers:
            emb_ent, emb_rel = layer(emb_ent, emb_rel, triplets)
        return emb_ent, emb_rel

    def score(self, emb_ent, emb_rel, triplets):
        if self.decoder_name == 'Semantic_Matching':
            return self.score_sm(emb_ent, triplets)
        elif self.decoder_name == 'Translational_Distance':
            return self.score_td(emb_ent, emb_rel, triplets)
        else:
            raise NotImplementedError

    def score_sm(self, emb_ent, triplets):
        h = triplets[:,0]
        r = triplets[:,1]
        t = triplets[:,2]

        W_pos = self.decoder_pos(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        W_neg = self.decoder_neg(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        headrel_pos = W_pos[h, r]
        headrel_neg = W_neg[h, r]

        tail_embs = emb_ent[t]
        pos_score = torch.sum(headrel_pos*tail_embs, dim = 1)
        neg_score = torch.sum(headrel_neg*tail_embs, dim = 1)
        return pos_score, neg_score
    
    def score_td(self, emb_ent, emb_rel, triplets):
        h = triplets[:,0]
        r = triplets[:,1]
        t = triplets[:,2]

        head_pos = self.decoder_head_pos(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        rel_pos = self.decoder_rel_pos(emb_rel).reshape(self.num_rel, self.num_rel, -1)
        tail_pos = self.decoder_tail_pos(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        head_neg = self.decoder_head_neg(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        rel_neg = self.decoder_rel_neg(emb_rel).reshape(self.num_rel, self.num_rel, -1)
        tail_neg = self.decoder_tail_neg(emb_ent).reshape(self.num_ent, self.num_rel, -1)
        pos_score = -torch.linalg.vector_norm(head_pos[h,r] + rel_pos[r,r] - tail_pos[t,r], dim = 1)
        neg_score = -torch.linalg.vector_norm(head_neg[h,r] + rel_neg[r,r] - tail_neg[t,r], dim = 1)

        return pos_score, neg_score

    