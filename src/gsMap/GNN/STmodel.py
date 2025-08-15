import torch
import torch.nn as nn
from torch.nn import functional as F

from gsMap.GNN.Encoder_Decoder import Encoder, Decoder
from gsMap.GNN.GCN import full_block

class StEmbeding(nn.Module):
    """
    Learn graph-smoothed and expression embeddings for each cell, with optional batch correction.
    
    Args:
        input_size (list): List of input feature sizes for each encoder.
        hidden_size (int): Hidden layer size in encoder/decoder.
        embedding_size (int): Latent embedding size.
        batch_embedding_size (int): Size of batch embedding vector.
        out_put_size (int): Output gene size.
        batch_size (int): Number of batches (not sample count).
        class_size (int): Number of classes for classification.
        distribution (str): Output distribution type ('nb', 'zinb', 'gaussian', etc.).
        Other GNN-related args passed to Encoder.
    """
    def __init__(self, 
                 input_size,
                 hidden_size,
                 embedding_size,
                 batch_embedding_size,
                 out_put_size,
                 batch_size,
                 class_size,
                 distribution,
                 module_dim,
                 hidden_gmf,
                 n_modules,
                 nhead,
                 n_enc_layer,
                 use_tf=True,
                 variational=True,
                 batch_representation='embedding',
                 dispersion='gene'):
        super().__init__()

        self.input_size = input_size
        self.z_num = len(self.input_size)
        self.distribution = distribution
        self.batch_representation = batch_representation
        self.num_batches = batch_size

        self.logtheta = nn.Parameter(torch.randn(batch_size, out_put_size))

        # Handle batch embedding
        if batch_representation == 'embedding':
            self.batch_embedding = nn.Embedding(batch_size, batch_embedding_size)
            self.batch_embedding_size = batch_embedding_size
        else:
            self.batch_embedding_size = batch_size  # one-hot case

        # Build encoders for each modality
        self.encoder = nn.ModuleList()
        for eid in range(self.z_num): 
            # Use indexed hidden_size if it's a list, otherwise use the same value
            h_size = hidden_size[eid] if isinstance(hidden_size, list) and len(hidden_size) > eid else hidden_size
            self.encoder.append(
                Encoder(self.input_size[eid],
                        h_size,
                        embedding_size,
                        self.batch_embedding_size,
                        module_dim,
                        hidden_gmf,
                        n_modules,
                        nhead,
                        n_enc_layer,
                        use_tf,
                        variational)
            )

        # Build decoders for reconstruction and classification
        self.decoder = nn.ModuleDict()
        # Use first hidden_size if it's a list
        h_size_decoder = hidden_size[0] if isinstance(hidden_size, list) else hidden_size
        for decoder_type in ['reconstruction', 'classification']:
            self.decoder[decoder_type] = Decoder(out_put_size,
                                                 h_size_decoder,
                                                 embedding_size,
                                                 self.batch_embedding_size,
                                                 class_size,
                                                 decoder_type,
                                                 self.distribution)

    def _process_batch(self, batch):
        if self.batch_representation == 'embedding':
            return self.batch_embedding(batch)
        else:
            return F.one_hot(batch, num_classes=self.num_batches).float()

    def forward(self, x_list, batch):
        batch = self._process_batch(batch)

        if self.distribution in ['nb', 'zinb']:
            library_size = x_list[0].sum(-1, keepdim=True)
        else:
            n = x_list[0].shape[0]
            device = x_list[0].device
            library_size = torch.ones(n, 1, device=device)

        x_rec_list, zi_logit_list, z_list = [], [], []
        for eid in range(self.z_num):  
            z = self.encoder[eid](x_list[eid], batch)
            x_rec, zi_logit = self.decoder['reconstruction'](z, batch)
            x_rec = x_rec * library_size

            x_rec_list.append(x_rec)
            zi_logit_list.append(zi_logit)
            z_list.append(z)

        x_class = self._classification(z_list, batch)
        return x_rec_list, x_class, zi_logit_list, z_list

    def _classification(self, z_list, batch):
        z = torch.cat(z_list, dim=1)
        return self.decoder['classification'](z, batch)

    def encode(self, x_list, batch):
        """Get latent representations"""
        batch = self._process_batch(batch)
        z_list = []
        for eid in range(self.z_num):  
            z = self.encoder[eid](x_list[eid], batch)
            z_list.append(z)
        
        # Return both concatenated and individual representations
        z_concat = torch.cat(z_list, dim=1)
        z_indv = z_list[0] if len(z_list) == 1 else z_list[1]  # Use expression encoder output
        
        return z_concat, z_indv