import torch
import torch.nn.functional as F
from torch import nn
from einops import repeat
import numpy as np


class Linear2D(nn.Module):
    """Linear2D module consists of a linear layer with 3D weight matrix.

    Args:
        input_dim (int): The input dimension of the Linear2D module.
        hidden_dim (int): The hidden dimension of the Linear2D module.
        n_modules (int): The number of modules of the Linear2D module.
        bias (bool, optional): Whether to use bias. Defaults to False.
    """

    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 n_modules, 
                 bias=False):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modules = n_modules

        self.weights = torch.randn(input_dim, hidden_dim, n_modules)
        self.weights = nn.Parameter(
            nn.init.xavier_normal_(self.weights))
        self.bias = None
        if bias:
            self.bias = torch.randn(1, hidden_dim, n_modules)
            self.bias = nn.Parameter(
                nn.init.xavier_normal_(self.bias))

    def forward(self, x):
        affine_out = torch.einsum("bi,ijk->bjk", [x, self.weights])
        if self.bias is not None:
            affine_out = affine_out + self.bias
        return affine_out


class GeneModuler(nn.Module):
    """GeneModuler takes gene expression as input and outputs gene modules.

    Attributes:
        input_dim (int): The input dimension of the GeneModuler model.
        hidden_dim (int): The hidden dimension of the GeneModuler model.
        n_modules (int): The number of modules of the GeneModuler model.
        layernorm (nn.LayerNorm): The layer normalization layer.
        extractor (Linear2D): The Linear2D object.
    """

    def __init__(self, 
                 input_dim=2000, 
                 hidden_dim=8, 
                 n_modules=16):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modules = n_modules

        self.layernorm = nn.LayerNorm(input_dim)
        self.extractor = Linear2D(
            input_dim=input_dim, hidden_dim=hidden_dim, n_modules=n_modules
        )

    def forward(self, x, batch=None):
        if batch is not None:
            module = self.layernorm(x, batch)
        else:
            module = self.layernorm(x)
        module = self.extractor(x).transpose(2, 1)
        return F.relu(module)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    Attributes:
        d_model (int): The dimensionality of the model. This should match the dimension of the input embeddings.
        max_len (int): The maximum length of the sequence for which positional encoding is computed.
    """
    def __init__(self, 
                 d_model,
                 max_len=500):
        
        super().__init__()
        
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed)
        pe[:, 1::2] = torch.cos(position * angular_speed)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, x.size(1), :]
        return encoded
    

class GeneModuleFormer(nn.Module):
    """GeneModuleFormer is a gene expression model based on the Transformer architecture.

    Attributes:
        input_dim (int): The dimensionality of the input gene expression data.
        module_dim (int): The dimensionality of each module in the model.
        hidden_dim (int): The hidden layer dimension used within the model.
        n_modules (int): The number of modules (transformer blocks) in the model.
        nhead (int): The number of attention heads in each transformer layer.
        n_enc_layer (int): The number of encoding layers in the transformer model.
    """

    def __init__(
        self,
        input_dim=2000,
        module_dim=30,
        hidden_dim=256,
        n_modules=16,
        nhead=8,
        n_enc_layer=3,
    ):
 
        super().__init__()

        self.moduler = GeneModuler(
            input_dim=input_dim, hidden_dim=module_dim, n_modules=n_modules
        )

        self.expand = (
            nn.Linear(module_dim, hidden_dim)
            if module_dim != hidden_dim
            else nn.Identity()
        )
        
        self.module = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_dim,
                                                     nhead=nhead,
                                                     dim_feedforward=4 * hidden_dim,
                                                     batch_first=True),
            num_layers=n_enc_layer
        )
        
        self.pe = PositionalEncoding(d_model=module_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, module_dim))
        
    def forward(self, x,):
        auto_fold = self.moduler(x)
        b, _, _ = auto_fold.shape
        auto_fold = self.pe(auto_fold)
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        auto_fold = torch.cat([cls_tokens, auto_fold], dim=1)
        auto_fold = self.expand(auto_fold)
        rep = self.module(auto_fold)
        cls_rep = rep[:,0,:]
        return cls_rep