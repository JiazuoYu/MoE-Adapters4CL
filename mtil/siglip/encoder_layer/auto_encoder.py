# --------------------------------------------------------
# Implementation of the Auto-Encoder for LEAS & DeEC
# --------------------------------------------------------
import torch.nn as nn
import torch

from ..loss import MSE_loss


class AutoEncoder(nn.Module):
    """
    The class defines the autoencoder model which takes in the latent feature embedding of 
    original CLIP layers. The default value for the input_dims is 256*13*13.
    About z-score for the Auto-Encoder, which is inspired by SEMA
    see paper: Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning
    https://arxiv.org/abs/2403.18886
    """

    def __init__(self, args, input_dims=256 * 13 * 13, hidden_dims=100,):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims, input_dims),
            nn.Sigmoid())

        self.mse_list = []
        self.softplus = nn.Softplus() 
        self.noise_AE = args.use_gate_noise
        self.noise_epsilon = args.gate_noise_epsilon
        self.is_train = args.is_train

    def forward(self, x):
        if self.is_train and self.noise_AE:  
            # Add noise when in training mode and noise is enabled
            x = self.add_noise(x)
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        mse_loss = MSE_loss(x, reconstructed_x)
        self.mse_list.append(mse_loss.item())
        # Append mse_loss to mse_list tensor
        z_score = self.get_z_score(mse_loss)
        return mse_loss, z_score

    def get_z_score(self, current_mse_loss):
        # Calculate mean and standard deviation of mse_list
        mse_tensor = torch.tensor(self.mse_list, dtype=torch.float32)
        mean_mse_loss = torch.mean(mse_tensor)
        std_mse = torch.std(mse_tensor)

        # Calculate z-score of current_mse_loss
        z_score = (current_mse_loss - mean_mse_loss) / std_mse if std_mse > 0 else torch.tensor(0.0)
        return z_score

    def get_avg(self):
        mse_tensor = torch.tensor(self.mse_list, dtype=torch.float32)
        mean_rd = torch.mean(mse_tensor)
        return mean_rd

    def get_avg_init(self, cut_off_rate=1.0):
        mse_tensor = torch.tensor(self.mse_list, dtype=torch.float32)
        length = mse_tensor.size(0)
        start_index = int(length * (1. - cut_off_rate))  # Round to make sure the index is an integer
        mean_mse_init = torch.mean(mse_tensor[start_index:])

        return mean_mse_init

    def get_std(self):
        mse_tensor = torch.tensor(self.mse_list, dtype=torch.float32)
        std_mse = torch.std(mse_tensor)
        return std_mse

    def get_std_init(self, cut_off_rate=1.0):
        mse_tensor = torch.tensor(self.mse_list, dtype=torch.float32)
        length = mse_tensor.size(0)
        start_index = int(length * (1. - cut_off_rate))  # Round to make sure the index is an integer
        std_mse_init = torch.std(mse_tensor[start_index:])

        return std_mse_init

    def add_noise(self, x):
        """
        Generate a noise of the same size as the input x and add it to x
        """
        noise = torch.normal(mean=0, std=self.noise_epsilon, size=x.size()).to(x)
        noisy_x = x + noise
        return noisy_x

