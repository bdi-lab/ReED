import torch
import numpy as np
from sklearn.decomposition import PCA

ent_pca = PCA(n_components = 32, random_state = 0)
rel_pca = PCA(n_components = 32, random_state = 0)

ent_feat = torch.load("ent_feats.pt").cpu().numpy()
rel_feat = torch.load("rel_feats.pt").cpu().numpy()

ent_feat32 = ent_pca.fit_transform(ent_feat)
rel_feat32 = rel_pca.fit_transform(rel_feat)

ent_feat32 = torch.from_numpy(ent_feat32).cuda()
rel_feat32 = torch.from_numpy(rel_feat32).cuda()

print(ent_feat32.size())
print(rel_feat32.size())

torch.save(ent_feat32, "ent_feats32.pt")
torch.save(rel_feat32, "rel_feats32.pt")