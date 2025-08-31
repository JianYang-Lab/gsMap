import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from .GeneFormer import GeneModuleFormer


def full_block(in_dim, out_dim, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(p=p_drop),
    )

class transform(nn.Module):
    """
    batch transform encoder
    """
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 batch_emb_size,
                 module_dim,
                 hidden_gmf,
                 n_modules,
                 nhead,
                 n_enc_layer,
                 use_tf):
        
        super().__init__()
        self.use_tf = use_tf

        if self.use_tf:
            self.input_size = hidden_gmf + batch_emb_size
            self.gmf = GeneModuleFormer(input_dim=input_size,
                                        module_dim=module_dim,
                                        hidden_dim=hidden_gmf,
                                        n_modules=n_modules,
                                        nhead=nhead,
                                        n_enc_layer=n_enc_layer
                                        )
            self.transform = full_block(self.input_size,hidden_size) 
        else:
            self.input_size = input_size + batch_emb_size
            self.transform = full_block(self.input_size,hidden_size)   
            self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x, batch):
        if self.use_tf:
            x = self.gmf(x)
            x = self.transform(torch.cat([x,batch],dim=1))
        else:
            x = self.transform(torch.cat([x,batch],dim=1)) 
        return self.norm(x)



class Encoder(nn.Module):
    """
    GCN encoder
    """
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 emb_size,
                 batch_emb_size,
                 module_dim,
                 hidden_gmf,
                 n_modules,
                 nhead,
                 n_enc_layer,
                 use_tf,
                 variational=True):
        
        super().__init__()
        self.variational = variational

        self.tf = transform(
            input_size, 
            hidden_size,
            batch_emb_size,
            module_dim,
            hidden_gmf,
            n_modules,
            nhead,
            n_enc_layer,
            use_tf
        )

        self.mlp = nn.Sequential(full_block(hidden_size, hidden_size),
                                 full_block(hidden_size,hidden_size))
        

        self.fc_mean = nn.Linear(hidden_size,emb_size)
        self.fc_var = nn.Linear(hidden_size,emb_size)
    
    def forward(self, x, batch):
        
        xtf = self.tf(x,batch)
        h = self.mlp(xtf)
        if not self.variational:
            mu = self.fc_mean(h)
            return mu
        
        mu = self.fc_mean(h)
        logvar = self.fc_var(h)
        setattr(self, "mu", mu)
        setattr(self, "sigma", logvar.exp().sqrt())
        setattr(self, "dist", Normal(self.mu, self.sigma))
        return self.dist.rsample()

    def kl_loss(self):
        if not hasattr(self, "dist"):
            return 0
        
        mean = torch.zeros_like(self.mu)
        scale = torch.ones_like(self.sigma)
        kl_loss = kl(self.dist, Normal(mean, scale))
        return kl_loss.mean()   

class Decoder(nn.Module):
    """
    Shared decoder
    """
    def __init__(self, 
                 out_put_size,
                 hidden_size, 
                 emb_size,
                 batch_emb_size,
                 class_size,
                 decoder_type,
                 distribution,
                 n_layers=3):
        super().__init__()

        self.decoder_type = decoder_type
        self.mlp = nn.ModuleList()
        
        # Set initial input size
        if decoder_type == 'reconstruction':
            input_size = emb_size + batch_emb_size
        elif decoder_type == 'classification':
            input_size = emb_size * 2 + batch_emb_size
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        # Build MLP layers with batch embedding concat at each step
        if isinstance(n_layers, int):
            n_layers = [hidden_size] * n_layers
        
        for hidden_size in n_layers:
            self.mlp.append(full_block(input_size, hidden_size))
            input_size = hidden_size + batch_emb_size  # update for next layer input

        # Final output layer
        if decoder_type == 'reconstruction':
            self.zi_logit = nn.Linear(input_size, out_put_size)
            self.fc_rec = nn.Linear(input_size, out_put_size)
        elif decoder_type == 'classification':
            self.fc_class = nn.Linear(input_size, class_size)

        if distribution in ['nb','zinb']:
            self.act = nn.Softmax(dim=-1)
        else: 
            self.act = nn.Identity()

    def forward(self, z, batch):
        x = torch.cat([z, batch], dim=1)

        for layer in self.mlp:
            x = layer(x)
            x = torch.cat([x, batch], dim=1)  # concat batch after each layer

        if self.decoder_type == 'reconstruction':
            x_hat = self.act(self.fc_rec(x))
            zi_logit = self.zi_logit(x)
            return x_hat, zi_logit

        elif self.decoder_type == 'classification':
            x_class = self.fc_class(x)
            return x_class

        else:
            raise ValueError(f"Unknown decoder_type: {self.decoder_type}")
