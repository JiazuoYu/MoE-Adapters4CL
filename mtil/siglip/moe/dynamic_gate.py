# --------------------------------------------------------
# Implementation of the Router for Dynamic MoE-Adapters
# --------------------------------------------------------
import torch

from torch import nn
from torch.distributions import Normal


class DynamicGate(torch.nn.Module):
    """
    This is a dynamic MOE router 
    The router's output dimensions are dynamically matched to 
    the number of experts
    """

    def __init__(self,
                 num_global_experts,  # Number of current experts
                 args,
                 fp32_gate=False,  # Whether to use fp32 precision
                 ):
        super().__init__()
        torch.manual_seed(args.gate_seed)  # random seed
        
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.noisy_gating = args.use_gate_noise  # Whether gate plus noise
        self.noise_epsilon = args.gate_noise_epsilon

        self.fp32_gate = fp32_gate
        self.global_expert_num = num_global_experts

        self.top_k = args.topk 

        self.capacity_factor = 0.0  # always use the adaptive capacity to avoid drop tokens
        self.gate_noise = 0.0  # always do not allow gate noise
        self.enable_softmax_logits = False
        self.is_train = args.is_train

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
            gates: a `Tensor` of shape [batch_size, n]
        Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
          clean_values: a `Tensor` of shape [batch, n].
          noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus normally distributed noise with standard deviation noise_stddev.
          noise_stddev: a `Tensor` of shape [batch, n], or None
          noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
          a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, current_expert_num, noise_epsilon=1e-2):
        """
        Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
        Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        # logits = x * w_gate  [100,512] * [512,22] = [100,22]
        clean_logits = x @ w_gate.to(x)

        # If noise gating is enabled and you are in training mode, add the noise
        if self.noisy_gating and train:
            # raw_noise_stddev = x * w_noise
            raw_noise_stddev = x @ w_noise.to(x)
            # Adjust the noise standard deviation 
            # by the softplus activation function and add a small constant noise term
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        if train is False:
            expert_num = current_expert_num
        else:
            expert_num = self.global_expert_num
        # get Top-k
        top_logits, top_indices = logits.topk(min(self.top_k + 1, expert_num), dim=1)
        # Select the top-k logits
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        
        # Use the scatter method to fill the top-k gates into the appropriate positions
        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # If noise gating is enabled and in training mode, 
        # and top-k is less than the number of experts, the load is calculated
        if self.noisy_gating and self.top_k < expert_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, x, w_gate, w_noise, current_expert_num):
        gate, load = self.noisy_top_k_gating(x, self.is_train, w_gate, w_noise,
                                             current_expert_num, noise_epsilon=self.noise_epsilon)
        return gate, load

    def update_expert_num(self, new_expert_num):
        self.global_expert_num = new_expert_num

