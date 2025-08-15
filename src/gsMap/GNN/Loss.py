## Define the loss function
import torch.nn.functional as F
from gsMap.GNN.Distribution import NegativeBinomial, ZeroInflatedNegativeBinomial

# import sys
# sys.path.append("/storage/yangjianLab/songliyang/SpatialData/gsMap_software/gsMap_V2/GNN")
# from Distribution import NegativeBinomial, ZeroInflatedNegativeBinomial

def rec_loss(x_hat,x,logtheta,zi_logit,distribution):
    if distribution == 'nb':
        loss  = -NegativeBinomial(mu=x_hat, theta=logtheta.exp()).log_prob(x).sum(-1).mean()
    elif distribution == 'zinb':
        loss  = -ZeroInflatedNegativeBinomial(mu=x_hat, theta=logtheta.exp(),zi_logits=zi_logit).log_prob(x).sum(-1).mean()
    else:
        loss = F.mse_loss(x_hat, x)
    
    return loss

def ce_loss(pred_label, true_label):
    return F.cross_entropy(pred_label, true_label)