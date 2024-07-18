import torch
from tqdm import tqdm

def evaluate(my_model, emb_ent, emb_rel, train, target_pos, target_neg, margin = 0):
    with torch.no_grad():
        my_model.eval()

        emb_ent, emb_rel = my_model(train, emb_ent, emb_rel)

        score_true, score_false = my_model.score(emb_ent, emb_rel, target_pos)

        loss = (score_true <= score_false + margin).sum()

        total_cnt = len(target_pos) + len(target_neg)

        score_true, score_false = my_model.score(emb_ent, emb_rel, target_neg)
        loss += (score_true + margin >= score_false).sum()

        loss = loss.item()/total_cnt
            
        my_model.train()
    
    return loss