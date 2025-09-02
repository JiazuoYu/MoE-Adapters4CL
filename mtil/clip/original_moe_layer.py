# --------------------------------------------------------
# Original Implementation of Moe-Adapters, 
# see code: https://github.com/JiazuoYu/MoE-Adapters4CL
# see paper: https://arxiv.org/abs/2403.11549
# build in the transformer block of Text/Image Encoder
# --------------------------------------------------------
from collections import OrderedDict

import torch
from torch import nn
from .adapter import Adapter
from torch.distributions.normal import Normal
from collections import Counter

from .moe_compoments import SparseDispatcher, LayerNorm, QuickGELU

val_task_id = None


def get_val_task_id():
    global val_task_id
    return val_task_id


def set_val_task_id(new_value):
    global val_task_id
    val_task_id = new_value


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 adapter_flag=False, args=None, text_or_image=None, i=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.layer = i
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.task_id = args.task_id
        self.noise_epsilon = 1e-2
        self.d_model = d_model
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()
        self.apply_moe = args.apply_moe
        self.noisy_gating = True
        self.is_train = args.is_train
        self.top_k = args.topk
        self.experts_num = args.experts_num  # e = 22
        self.router_num = 1  # n = 1
        self.ffn_adapt = args.ffn_adapt

        self.text_or_image = text_or_image
        if text_or_image == 'text':
            self.choose_map_text = torch.zeros([self.experts_num])  # experts使用频率记录表 12层
        else:
            self.choose_map_image = torch.zeros([self.experts_num])  # experts使用频率记录表 12层
        self.ffn_option = args.ffn_option
        self.ffn_num = args.ffn_num
        self.autorouter = args.autorouter
        self.adapter_flag = adapter_flag
        self.adaptmlp_list = nn.ModuleList()
        if self.ffn_adapt and self.adapter_flag:
            if self.apply_moe == True:
                if self.task_id > -1:  # router>1
                    self.router_list = nn.ParameterList()
                    self.w_noise_list = nn.ParameterList()
                    for i in range(11):  # Task number
                        self.router_list.append(
                            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                        self.w_noise_list.append(
                            nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True))
                    for i in range(self.experts_num):  # Expert number
                        self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                                init_option='lora',
                                                adapter_scalar=0.1,
                                                adapter_layernorm_option='none',
                                                )
                        self.adaptmlp_list.append(self.adaptmlp)  # 专家列表
                else:  # one router for all task
                    self.router1 = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
                    self.w_noise = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
                    for i in range(self.experts_num):
                        self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                                init_option='lora',
                                                adapter_scalar=0.1,
                                                adapter_layernorm_option='none',
                                                )
                        self.adaptmlp_list.append(self.adaptmlp)
            else:  # without moe
                self.adaptmlp = Adapter(d_model=d_model, dropout=0.1, bottleneck=self.ffn_num,
                                        init_option='lora',
                                        adapter_scalar=0.1,
                                        adapter_layernorm_option='none',
                                        )
        self.loss = 0.0

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
          x: a `Tensor`.
        Returns:
          a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

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

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
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

        # 计算没有噪声的logits，logits = x * w_gate  [100,512] * [512,22] = [100,22]
        clean_logits = x @ w_gate.to(x)

        # 如果启用了噪声门控且处于训练模式，则添加噪声
        if self.noisy_gating and train:
            # 计算噪声的标准差，raw_noise_stddev = x * w_noise
            raw_noise_stddev = x @ w_noise.to(x)
            # 通过softplus激活函数调整噪声标准差，并添加一个小的常数噪声项
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            # 添加噪声到logits
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            # 如果不添加噪声，则logits为干净的logits
            logits = clean_logits

        # 计算top-k + 1，这在噪声门控中是需要的
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        # 选择前top-k的logits
        top_k_logits = top_logits[:, :self.top_k]
        # 选择前top-k的索引
        top_k_indices = top_indices[:, :self.top_k]
        # 计算前top-k的gates
        top_k_gates = self.softmax(top_k_logits)

        # 创建一个与logits相同形状的全零张量
        zeros = torch.zeros_like(logits)
        # 使用scatter方法，将前top-k的gates填入相应的位置
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        # 如果启用了噪声门控且处于训练模式，并且top-k小于专家数目，则计算负载
        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            # 否则直接根据gates计算负载
            load = self._gates_to_load(gates)

        # 返回gates和负载
        return gates, load

    def forward(self, x: torch.Tensor):

        x = x + self.attention(self.ln_1(x))  # 和原始clip一样，layernorm之后，self attn,再residual

        # Original MoE-Adapter
        if self.ffn_adapt and self.ffn_option == 'parallel' and self.adapter_flag:
            # true
            if self.apply_moe == True:
                if self.task_id > -1:
                    # 不断有新数据加入，CL的情况，多个expert
                    if self.autorouter == True and val_task_id == -1:
                        x = x + self.mlp(self.ln_2(x))
                        return x  # zero-shot
                    # x_re 就是 cls_token， 相当于取moe输入tokens的第一个
                    x_re = x.permute(1, 0, 2)[:, 0, :]
                    if val_task_id is not None and self.autorouter == True:  # val情况
                        gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router_list[val_task_id],
                                                              self.w_noise_list[val_task_id])
                    else:  # train情况
                        gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router_list[self.task_id],
                                                              self.w_noise_list[self.task_id])
                    importance = gates.sum(0)
                    loss = self.cv_squared(importance) + self.cv_squared(load)
                    loss *= 1e-2  # # Todo

                    # router 选择 expert，完成分配，MOE-Adapter
                    nonzero_indices = torch.nonzero(gates)  # 找到不为0的位置
                    counter = Counter(nonzero_indices[:, 1].tolist())
                    for number, count in counter.items():  # number是expert号，count是batch内使用的expert的次数
                        if self.text_or_image == 'text':
                            self.choose_map_text[number] = self.choose_map_text[number] + count
                            total_experts_text = sum(self.choose_map_text)
                        else:
                            self.choose_map_image[number] = self.choose_map_image[number] + count
                            total_experts_image = sum(self.choose_map_image)

                    dispatcher = SparseDispatcher(self.experts_num, gates)
                    expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))
                    # 将输入tensor分配给各个专家，然后各专家进行运算
                    expert_outputs = [self.adaptmlp_list[i](expert_inputs[i]
                                                            .view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(
                        x), add_residual=False)
                                      for i in range(self.experts_num)]  # 多个experts 1个router

                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:  # 删除没有用到的expert的输出
                            expert_outputs.pop(i)  # 删除第i项
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1
                    # 将各expert的输出合并
                    y = dispatcher.combine(expert_outputs)
                    y = y.view(x.shape[1], x.shape[0], x.shape[2])  # n x 39424 -> n x 77 x 512
                    # adapter和原始clip的mlp+ln输出相叠加，和原始clip以及adapter一样
                    x = x + self.mlp(self.ln_2(x)) + y.permute(1, 0, 2)
                else:  # 单expert情况
                    x_re = x.permute(1, 0, 2)[:, 0, :]
                    gates, load = self.noisy_top_k_gating(x_re, self.is_train, self.router1,
                                                          self.w_noise)

                    dispatcher = SparseDispatcher(self.experts_num, gates)
                    expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))  #
                    expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                                  x.shape[0], x.shape[2]).to(x),
                                                            add_residual=False) for i in
                                      range(self.experts_num)]  # 11 experts 1 router
                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:
                            expert_outputs.pop(i)
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1

                    y = dispatcher.combine(expert_outputs)  # 合并各个专家的输出
                    y = y.view(x.shape[1], x.shape[0], x.shape[2])
                    x = (x + self.mlp(self.ln_2(x)) + y.permute(1, 0, 2))

            else:  # one adapter, no routers and moe
                x_re = x.permute(1, 0, 2)
                adapt_x = self.adaptmlp(x_re, add_residual=False).permute(1, 0, 2)
                x = x + self.mlp(self.ln_2(x)) + adapt_x

        else:  # original(without adapter)
            x = x + self.mlp(self.ln_2(x))
        return x
