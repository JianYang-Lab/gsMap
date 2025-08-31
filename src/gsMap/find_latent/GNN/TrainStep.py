import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from .Loss import rec_loss, ce_loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7,delta=0, path=None):

        self.patience = patience
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score == -np.inf:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.early_stop = False


class ModelTrain(object):
    def __init__(self, 
                 model, 
                 optimizer,
                 distribution,
                 mode,
                 lr,
                 model_path):

        self.model = model
        self.optimizer = optimizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.distribution = distribution
        self.mode = mode
        self.lr = lr
        self.model_path = model_path
        
        self.train_loader = None
        self.val_loader = None
        self.losses = []
        self.val_losses = []
    
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

        if self.lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def compute_loss(self,x_hat,x,log_theata,zi_logit,x_class,labels):
        if self.mode == 'reconstruction':
            loss = 0
            for id in range(len(x_hat)):
                loss_rec = rec_loss(x_hat[id],x,log_theata,zi_logit[id],self.distribution)
                loss_kld = self.model.encoder[id].kl_loss()
                loss = loss + loss_rec + loss_kld
                
        elif self.mode == 'classification':
            loss = ce_loss(x_class, labels)
        
        return loss

    def _make_train_step_fn(self):
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x_gcn,ST_batches, x, labels):

            self.model.train()

            x_hat, x_class, zi_logit, _ = self.model([x,x_gcn], ST_batches)
            log_theata = self.model.logtheta[ST_batches]
            loss = self.compute_loss(x_hat,x, log_theata, zi_logit,x_class,labels)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
                
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x_gcn,ST_batches, x, labels):
            
            self.model.eval()

            x_hat, x_class, zi_logit, _ = self.model([x,x_gcn], ST_batches)
            log_theata = self.model.logtheta[ST_batches]
            loss = self.compute_loss(x_hat,x, log_theata, zi_logit,x_class,labels)

            return loss.item()

        return perform_val_step_fn
            
    def _mini_batch(self, epoch_idx, n_epochs, validation=False):
        # The mini-batch can be used with both loaders
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        # mini-batch loop
        mini_batch_losses = []
        batch_iter = len(self.train_loader)    

        for batch_idx, (x_gcn,ST_batches,x,labels) in enumerate(data_loader):
            # p = float(batch_idx + epoch_idx * batch_iter) / (n_epochs * batch_iter)
            # grl_lambda = 2. / (1. + np.exp(-10 *p)) -1

            x_gcn = x_gcn.to(self.device)
            ST_batches = ST_batches.long().to(self.device)
            x = x.to(self.device)
            labels = labels.to(self.device)

            mini_batch_loss = step_fn(x_gcn,ST_batches,x,labels)
            mini_batch_losses.append(mini_batch_loss)
            # mini_batch_lossses_rgl.append(mini_batch_loss_rgl)

        return np.mean(mini_batch_losses)


    def _set_requires_grad(self, module_group, mode):
        for name, param_group in module_group.items():
            requires_grad = (mode == name)
            for param in param_group.parameters():
                param.requires_grad = requires_grad
    
    
    def train(self, n_epochs,patience):
        loss_track = EarlyStopping(patience)
        
        self._set_requires_grad(self.model.decoder, self.mode)

        pbar = tqdm(range(n_epochs), desc=f'LGCN train ({self.mode})', total=n_epochs)
        for epoch in pbar:

            # Performs training
            train_loss = self._mini_batch(epoch,n_epochs,validation=False)

            # Performs evaluation
            with torch.no_grad():
                val_loss = self._mini_batch(epoch,n_epochs,validation=True)

            # Save the best model
            if loss_track.best_score < -val_loss:
                torch.save(self.model.state_dict(),self.model_path)

            # Update validation loss
            loss_track(val_loss)
            if loss_track.early_stop:
                print(f'Stop training, as {self.mode} validation loss has not decreased for {patience} consecutive steps.')
                break

            pbar.set_postfix({'train loss': f'{train_loss.item():.4f}',
                            'validation loss': f'{val_loss.item():.4f}'})
