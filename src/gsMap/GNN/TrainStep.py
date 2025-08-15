import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F

from gsMap.GNN.Loss import rec_loss, ce_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, delta=0, path=None):

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

    def compute_loss(self, x_hat, x, log_theata, zi_logit, x_class, labels):
        if self.mode == 'reconstruction':
            loss = 0
            for id in range(len(x_hat)):
                loss_rec = rec_loss(x_hat[id], x, log_theata, zi_logit[id], self.distribution)
                loss_kld = self.model.encoder[id].kl_loss()
                loss = loss + loss_rec + loss_kld
                
        elif self.mode == 'classification':
            loss = ce_loss(x_class, labels)
        
        return loss

    def _make_train_step_fn(self):
        # Builds function that performs a step in the train loop
        def perform_train_step_fn(x_gcn, ST_batches, x, labels):

            self.model.train()

            x_hat, x_class, zi_logit, _ = self.model([x, x_gcn], ST_batches)
            log_theata = self.model.logtheta[ST_batches]
            loss = self.compute_loss(x_hat, x, log_theata, zi_logit, x_class, labels)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        # Builds function that performs a step in the validation loop
        def perform_val_step_fn(x_gcn, ST_batches, x, labels):
            
            self.model.eval()
            
            with torch.no_grad():
                x_hat, x_class, zi_logit, _ = self.model([x, x_gcn], ST_batches)
                log_theata = self.model.logtheta[ST_batches]
                loss = self.compute_loss(x_hat, x, log_theata, zi_logit, x_class, labels)

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_gcn, ST_batches, x, labels in data_loader:
            x_gcn = x_gcn.to(self.device)
            ST_batches = ST_batches.to(self.device).long()
            x = x.to(self.device)
            labels = labels.to(self.device).long()

            mini_batch_loss = step_fn(x_gcn, ST_batches, x, labels)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(self, n_epochs, patience=10, seed=42):
        self.set_seed(seed)
        early_stopping = EarlyStopping(patience=patience, path=self.model_path)
        
        for epoch in tqdm(range(n_epochs), desc='Training'):
            self.losses.append(self._mini_batch(validation=False))

            with torch.no_grad():
                self.val_losses.append(self._mini_batch(validation=True))
                
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{n_epochs}, Loss: {self.losses[-1]:.4f}, Val Loss: {self.val_losses[-1]:.4f}')
                
            # Early stopping
            early_stopping(self.val_losses[-1])
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
            # Save best model
            if self.val_losses[-1] <= min(self.val_losses):
                torch.save(self.model.state_dict(), self.model_path)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            output = self.model(x.to(self.device))
        return output