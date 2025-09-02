# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn

class Adapter(nn.Module):
    """
    The experts in MoE-Adapters(++)
    Implementation by LoRA: https://arxiv.org/abs/2106.09685
    """
    def __init__(self,
                 d_model=None,  # Model dimension
                 bottleneck=None,  # Dimension of the bottleneck layer
                 dropout=0.0,  # Dropout rate
                 init_option="lora",  # Initialization option, default is "lora"
                 adapter_scalar="1.0",  # Scaling factor for the adapter, default is 1.0
                 adapter_layernorm_option="in"):  # LayerNorm option, default is "in"
        super().__init__()  # Call the parent class constructor
        self.n_embd = d_model if d_model is None else d_model  # Set the model dimension
        self.down_size = bottleneck  # Set the bottleneck layer dimension

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option  # Set the LayerNorm option

        self.adapter_layer_norm_before = None  # Initialize LayerNorm
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)  # Create the LayerNorm layer

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))  # Create a parameter if the scaling factor is learnable
        else:
            self.scale = float(adapter_scalar)  # Otherwise, use a fixed scaling factor

        self.down_proj = nn.Linear(self.n_embd, 64)  # Create the down-projection linear layer
        self.non_linear_func = nn.ReLU()  # Non-linear activation function
        self.up_proj = nn.Linear(self.down_size, self.n_embd)  # Create the up-projection linear layer

        self.dropout = dropout  # Set the dropout rate
        if init_option == "bert":
            raise NotImplementedError  # Raise an error if the initialization option is "bert"
        elif init_option == "lora":
            with torch.no_grad():  # Initialize weights and biases
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))  # Initialize down-projection weights with Kaiming
                nn.init.zeros_(self.up_proj.weight)  # Initialize up-projection weights to 0
                nn.init.zeros_(self.down_proj.bias)  # Initialize down-projection biases to 0
                nn.init.zeros_(self.up_proj.bias)  # Initialize up-projection biases to 0

    def forward(self, x, add_residual=True, residual=None):
        # Forward propagation function
        residual = x if residual is None else residual  # Use the input as residual if no residual is provided
        if self.adapter_layernorm_option == 'in':  # Apply LayerNorm to the input if option is "in"
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)  # Apply the down-projection layer
        down = self.non_linear_func(down)  # Apply the non-linear activation function
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)  # Apply dropout
        up = self.up_proj(down)  # Apply the up-projection layer

        up = up * self.scale  # Apply the scaling factor

        if self.adapter_layernorm_option == 'out':  # Apply LayerNorm to the output if option is "out"
            up = self.adapter_layer_norm_before(up)

        if add_residual:  # Add the residual if specified
            output = up + residual
        else:
            output = up  # Otherwise, output only the result of the up-projection layer
        return output  # Return the output
