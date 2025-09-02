# --------------------------------------------------------
# Implementation of the Dynamic Moe-Adapters, build in 
# the transformer block of Text/Image Encoder
# --------------------------------------------------------

from collections import OrderedDict
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from ..moe.adapter import Adapter
from torch.distributions.normal import Normal
from collections import Counter

from ..moe.dynamic_gate import DynamicGate
from ..moe.moe_compoments import SparseDispatcher, LayerNorm, QuickGELU
from .auto_encoder import AutoEncoder
from ..siglip.modeling_siglip import SiglipAttention, SiglipMLP

# from ..src.finetune import iter


eval_task_id_list_text = [-1] * 12
eval_task_id_list_visual = [-1] * 24
eval_zero_shot = False
eval_best_discrepancy = []


def get_val_task_id_text(idx):
    global eval_task_id_list_text
    return eval_task_id_list_text[idx]


def update_val_task_id_text(new_value, idx):
    global eval_task_id_list_text
    eval_task_id_list_text[idx:] = [new_value] * (len(eval_task_id_list_text) - idx)


def get_val_task_id_visual(idx):
    global eval_task_id_list_visual
    return eval_task_id_list_visual[idx]


def update_val_task_id_visual(new_value, idx):
    global eval_task_id_list_visual
    eval_task_id_list_visual[idx:] = [new_value] * (len(eval_task_id_list_visual) - idx)


def get_eval_best_discrepancy():
    global eval_best_discrepancy
    return eval_best_discrepancy


def update_eval_best_discrepancy(new_value):
    global eval_best_discrepancy
    eval_best_discrepancy.append(new_value)
    

def get_eval_zero_shot():
    global eval_zero_shot
    return eval_zero_shot


def update_eval_zero_shot(new_value):
    global eval_zero_shot
    eval_zero_shot = new_value


class DynResidualAttentionBlock(nn.Module):
    """
    Core algorithm of MoE-adapter++, including Dynamic MoE-Adapters, DEeC & LEAS
    initialised 1 time before training for each task
    """

    def __init__(self,
                #  d_model: int,
                #  n_head: int,
                #  attn_mask: torch.Tensor = None,
                 adapter_flag=False,
                 LEAS_flag=False,
                 args=None,
                 text_or_image=None,
                 i=None,
                 hf_config=None,  # 兼容siglip的参数
                 ):
        super().__init__()

        d_model = hf_config.hidden_size
        self.self_attn = SiglipAttention(hf_config)
        self.layer_norm1 = LayerNorm(d_model, eps=hf_config.layer_norm_eps)
        self.mlp = SiglipMLP(hf_config)
        self.layer_norm2 = LayerNorm(d_model, eps=hf_config.layer_norm_eps)

        self.layer = i

        # task_id when training
        self.task_id = args.task_id
        self.task_num = args.task_num

        self.d_model = d_model
        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()  
        self.apply_moe = args.apply_moe  # Whether to apply moe

        self.is_train = args.is_train  # Whether or not in training mode
        # Flags required for training
        self.force_zero_shot = False
        self.force_eval_task_id = None
        if self.is_train is False:
            if args.repeat_train is False:
                self.init = True
            if args.force_val_task_id is not None:
                self.force_eval_task_id = args.force_val_task_id
            if args.force_zero_shot:
                self.force_zero_shot = True
            # build a list to track the acc of val_task_id
            if args.track_val_task_id[self.layer]:
                self.eval_acc_list = []
            if args.track_val_discrepancy[self.layer]:
                self.eval_discrepancy_list = []
            # self.eval_best_sim = 1.
            self.print_eval_batches = args.print_eval_batches

        # expert num
        self.max_expert_num = args.max_expert_num
        self.text_or_image = text_or_image
        if text_or_image == 'text':
            # print('text transformer')
            self.choose_map_text = torch.zeros([self.max_expert_num]) 
            self.data_len = 64
        else:
            # print('image transformer')
            self.choose_map_image = torch.zeros([self.max_expert_num])
            if args.model == "Siglip-B-14-224":
                self.data_len = 197
            if args.model == "ViT-B/16":
                self.data_len = 197
            elif args.model == "ViT-L/14@336px":
                self.data_len = 577

        self.ffn_num = args.ffn_num
        self.autorouter = args.autorouter
        self.single_router = args.single_router 
        self.adapter_flag = adapter_flag
        self.LEAS_layer_flag = LEAS_flag
        self.reconginition_layer = False
        
        # augmented_zero_shot (not deploy)
        self.augmented_zero_shot = args.augmented_zero_shot
        self.no_tree_strategy = args.no_tree_strategy
        self.without_LEAS = args.without_LEAS

        # setting for mutil-LEAS
        if self.LEAS_layer_flag:
            # discrepancy function
            self.discrepancy_weighted_vector = args.discrepancy_weighted_vector
            # Construct AE as LEAS, use frozen latent feature embedding as input
            self.global_expert_num = None  # current experts num
            # Determine freezing, activation and training of AEs in LEAS
            if text_or_image == 'text':
                self.global_expert_num = args.text_expert_num_list[i]
                self.frozen_expert_num = args.text_expert_num_list[i]
                self.hidden_dims = args.text_AE_hidden_dims
            else:
                self.global_expert_num = args.image_expert_num_list[i]
                self.frozen_expert_num = args.image_expert_num_list[i]
                self.hidden_dims = args.visual_AE_hidden_dims

            # Record the number of experts and routers and block unused experts.
            self.register_buffer("activated_experts_num", torch.tensor([0.0]))
            self.activated_experts_num[0] = float(self.global_expert_num)
            self.register_buffer("activated_router_num", torch.tensor([float(args.task_id)]))
            self.register_parameter('experts_mask',
                                    torch.nn.Parameter(torch.zeros(size=(self.max_expert_num,)),
                                                        requires_grad=False))
            self.experts_mask[:self.global_expert_num] = 1.0

            # expansion signal, i.e. Flag_e in paper
            self.expansion_flag = False
            if args.repeat_train is False:
                self.expansion_flag = True  # Initial tasks do not require expansion
            self.force_expansion = args.force_expansion_list[self.layer]

            # Building Auto-Encoder Sequences for LEAS / DeEC
            self.auto_encoder_list = nn.ModuleList()
            self.mse_loss = torch.tensor(0.)  # # Sum of MSE_loss for the whole layer
            self.z_score = [torch.tensor(0.)] * self.max_expert_num  # z_score for each ae
            # MSE_loss list corresponding to each AE during training
            # To calculate the mean and std value to recored selection preferences
            self.mse_loss_list = [torch.tensor(0.)] * self.max_expert_num  
            # MSE_loss corresponding to each AE during evaluation
            self.eval_mse_list = [torch.tensor(0.)] * self.max_expert_num
            self.use_LEAS_to_eval = args.use_LEAS_to_eval

            if self.use_LEAS_to_eval:
                self.mse_loss_avg_list = nn.ParameterList()
                self.mse_loss_std_list = nn.ParameterList()
                # record discrepancy when training (Practically no application)
                self.discrepancy_list = nn.ParameterList()
                # best_discrepancy_list when inference
                self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
            
            for i in range(self.task_num):  # Task number
                # Record selection preferences for LEAS during training
                self.mse_loss_avg_list.append(
                    nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                self.mse_loss_std_list.append(
                    nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                self.discrepancy_list.append(
                    nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

            # build AEs for LEAS
            for idx in range(self.max_expert_num):
                auto_encoder = AutoEncoder(
                    # input_dims=self.data_len * d_model,
                    args=args,
                    input_dims=d_model,
                    hidden_dims=self.hidden_dims,
                )
                self.auto_encoder_list.append(auto_encoder)

        # build moe adapter(experts)
        if self.adapter_flag:
            self.global_expert_num = None
            self.reconginition_layer = False  
            # Confirmation of the calculation of the discrepancy
            self.discrepancy_weighted_vector = args.discrepancy_weighted_vector  

            # Thresholds required to construct expansion and zero_shot
            if text_or_image == 'text':
                self.expansion_threshold = args.expansion_threshold_text
                self.zero_shot_threshold = args.zero_shot_threshold_text
                self.global_expert_num = args.text_expert_num_list[i]
                self.frozen_expert_num = args.text_expert_num_list[i]
                # Locating the Reconginition Layer
                if i == args.use_dyn_moe_layer_list_text.index(True):
                    self.reconginition_layer = True

            else:  # visual transformer
                self.expansion_threshold = args.expansion_threshold_image
                self.zero_shot_threshold = args.zero_shot_threshold_image
                self.global_expert_num = args.image_expert_num_list[i]
                self.frozen_expert_num = args.image_expert_num_list[i]
                # Locating the Reconginition Layer
                if i == args.use_dyn_moe_layer_list_visual.index(True):
                    self.reconginition_layer = True
                
            # Record the number of experts and routers and block unused experts.
            self.register_buffer("activated_experts_num", torch.tensor([0.0]))
            self.activated_experts_num[0] = float(self.global_expert_num)
            self.register_buffer("activated_router_num", torch.tensor([float(args.task_id)]))
            self.register_parameter('experts_mask',
                                    torch.nn. Parameter(torch.zeros(size=(self.max_expert_num,)), requires_grad=False))
            self.experts_mask[:self.global_expert_num] = 1.0

            # experts & auto_encoder
            self.adaptmlp_list = nn.ModuleList()

            # Router for Dynamic MoE-Adapters
            self.noisy_gating = args.use_gate_noise  # Whether noise gate 
            self.noise_epsilon = args.gate_noise_epsilon        
            self.adaptive_moe_gate = DynamicGate(
                num_global_experts=self.global_expert_num,
                fp32_gate=False,
                args=args
            )

            # expansion signal, i.e. Flag_e in paper
            self.expansion_flag = False
            if args.repeat_train is False:
                self.expansion_flag = True  # Initial tasks do not require expansion
            self.force_expansion = args.force_expansion_list[self.layer]

            # Building Auto-Encoder Sequences for LEAS / DeEC
            self.auto_encoder_list = nn.ModuleList()
            self.mse_loss = torch.tensor(0.)  # # Sum of MSE_loss for the whole layer
            self.z_score = [torch.tensor(0.)] * self.max_expert_num  # z_score for each ae
            # MSE_loss list corresponding to each AE during training
            # To calculate the mean and std value to recored selection preferences
            self.mse_loss_list = [torch.tensor(0.)] * self.max_expert_num  
            # MSE_loss corresponding to each AE during evaluation
            self.eval_mse_list = [torch.tensor(0.)] * self.max_expert_num
            self.use_LEAS_to_eval = args.use_LEAS_to_eval

            if self.apply_moe:
                if self.single_router is False:  # use expert
                    # build routers
                    self.router_list = nn.ParameterList()
                    self.w_noise_list = nn.ParameterList()
                    self.expert_activate_freq_list = nn.ParameterList()
                    if self.use_LEAS_to_eval:
                        self.mse_loss_avg_list = nn.ParameterList()
                        self.mse_loss_std_list = nn.ParameterList()
                        if self.reconginition_layer:
                            # Recording of discrepancy in training for automated thresholding
                            self.discrepancy_list = nn.ParameterList()
                            # when eval, best_discrepancy_list
                            self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
                            self.zs_thre_list = [torch.tensor(0.)] * self.task_num
                    for i in range(self.task_num):  # Task number
                        self.router_list.append(
                            nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True))
                        self.w_noise_list.append(
                            nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True))
                        self.expert_activate_freq_list.append(
                            nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                        # Record the mean value of mse_loss for all AEs in the recognition layer
                        if self.use_LEAS_to_eval:
                            self.mse_loss_avg_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            self.mse_loss_std_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            if self.reconginition_layer:
                                self.discrepancy_list.append(
                                    nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

                    # build AE & experts
                    for idx in range(self.max_expert_num):
                        self.adaptmlp = Adapter(d_model=d_model,
                                                dropout=0.1,
                                                bottleneck=self.ffn_num,
                                                init_option='lora',
                                                adapter_scalar=0.1,
                                                adapter_layernorm_option='none',
                                                )  # default
                        self.adaptmlp_list.append(self.adaptmlp)
                        if self.text_or_image == "text":
                            hidden_dims = args.text_AE_hidden_dims
                        else:
                            hidden_dims = args.visual_AE_hidden_dims
                        auto_encoder = AutoEncoder(
                            # input_dims=self.data_len * d_model,
                            args=args,
                            input_dims=d_model,
                            hidden_dims=hidden_dims,
                        )
                        self.auto_encoder_list.append(auto_encoder)

                else:  # one router for all task
                    self.router1 = nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True)
                    self.w_noise1 = nn.Parameter(torch.zeros(d_model, self.max_expert_num), requires_grad=True)
                    self.expert_activate_freq_list = nn.ParameterList()
                    if self.use_LEAS_to_eval:
                        self.mse_loss_avg_list = nn.ParameterList()
                        self.mse_loss_std_list = nn.ParameterList()
                        if self.reconginition_layer:
                            # Recording of discrepancy in training for automated thresholding
                            self.discrepancy_list = nn.ParameterList()
                            # when eval, best_discrepancy_list
                            self.previous_discrepancy_list = [torch.tensor(0.)] * self.task_num
                            self.zs_thre_list = [torch.tensor(0.)] * self.task_num
                            
                    for i in range(self.task_num):  # Task number
                        # Record expert activation frequency
                        self.expert_activate_freq_list.append(
                            nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                        # Record the mean value of mse_loss for all AEs in the recognition layer
                        if self.use_LEAS_to_eval:
                            self.mse_loss_avg_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            self.mse_loss_std_list.append(
                                nn.Parameter(torch.zeros(self.max_expert_num), requires_grad=False))
                            if self.reconginition_layer:
                                self.discrepancy_list.append(
                                    nn.Parameter(torch.zeros(self.task_num), requires_grad=False))

                    for idx in range(self.max_expert_num):
                        self.adaptmlp = Adapter(d_model=d_model,
                                                dropout=0.1,
                                                bottleneck=self.ffn_num,
                                                init_option='lora',
                                                adapter_scalar=0.1,
                                                adapter_layernorm_option='none',
                                                )  # default
                        self.adaptmlp_list.append(self.adaptmlp)
                        if self.text_or_image == "text":
                            hidden_dims = args.text_AE_hidden_dims
                        else:
                            hidden_dims = args.visual_AE_hidden_dims
                        auto_encoder = AutoEncoder(
                            args=args,
                            input_dims=d_model,
                            hidden_dims=hidden_dims,
                        )
                        self.auto_encoder_list.append(auto_encoder)

            else:
                self.adaptmlp = Adapter(d_model=d_model,
                                        dropout=0.1,
                                        bottleneck=self.ffn_num,
                                        init_option='lora',
                                        adapter_scalar=0.1,
                                        adapter_layernorm_option='none',
                                        )  # default
                
    def set_iteration(self, iteration):
        self.current_iteration = iteration

    def attention(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,):
        attention_mask = attention_mask.to(dtype=x.dtype, device=x.device) if attention_mask is not None else None
        # return self.self_attn(x, x, x, need_weights=False, attn_mask=attention_mask)[0]
        return self.self_attn(hidden_states=x, attention_mask=attention_mask)[0]

    def forward(self, x: torch.Tensor, x_original: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,):

        x = x + self.attention(self.layer_norm1(x), attention_mask=attention_mask) 
        if self.adapter_flag is False and self.LEAS_layer_flag is False:
            x = x + self.mlp(self.layer_norm2(x))
            return x, x

        # calculate x_frozen by frozen CLIP
        with torch.no_grad():
            x_original = x_original + self.attention(self.layer_norm1(x_original), attention_mask=attention_mask)
            x_frozen = x_original.detach()
            x_frozen = x_frozen + self.mlp(self.layer_norm2(x_frozen))
        zero_shot = self.force_zero_shot

        # Only the LEAS layer is deployed, i.e. the mutil-LEAS setup
        if self.LEAS_layer_flag and zero_shot is False:
            # Confirmation of the number of experts currently active
            current_experts_num = self.global_expert_num
            if self.is_train is False and self.force_eval_task_id is None:  # eval
                self.previous_discrepancy_list = self.get_best_discrepancy_list(x)
            if self.is_train:
                self.get_block_mse_loss_and_z_score(x)
                if self.expansion_flag is False:
                    self.add_LEAS(self.global_expert_num)

        # Dynamic MoE-Adapters
        if self.adapter_flag and zero_shot is False:
            # Confirmation of the number of experts currently active
            current_experts_num = self.global_expert_num

            # When eval get task_id by LEAS
            if self.is_train is False:
                eval_task_id = -1

                # Detection process at the reconginition layer only
                if self.reconginition_layer:
                    # get eval_task_id
                    if self.is_train is False and self.force_eval_task_id is None:  # eval
                        # Use LEAS to get task_id when eval
                        zero_shot, eval_task_id, current_experts_num = self.eval_task_id_LEAS(x_original)  

                    elif self.is_train is False and self.force_eval_task_id is not None:  # eval_force
                        eval_task_id = self.force_eval_task_id
                    # Update eval task_id to all layers
                    if self.text_or_image == "text":
                        update_val_task_id_text(eval_task_id, self.layer)
                    else:
                        update_val_task_id_visual(eval_task_id, self.layer)
                        
                else:
                    # Update eval task_id to all layers
                    if self.text_or_image == "text":
                        eval_task_id = get_val_task_id_text(self.layer)
                    else:
                        eval_task_id = get_val_task_id_visual(self.layer)
                    # # Other layers get zero-shot flag
                    zero_shot = get_eval_zero_shot()
                    
                # Record the number of times the frozen CLIP (zero-shot) was activated
                if zero_shot:
                    self.eval_acc_list.append(-1)
                else:
                    self.eval_acc_list.append(eval_task_id)
            
            # LEAS is not used, and zero-shot is not used by default, 
            # but the LEAS inference results are still retained 
            # for the normal operation of the program
            if self.without_LEAS:
                zero_shot = False

            # MoE
            if self.apply_moe:
                
                # get zero-shot flag
                if self.autorouter and zero_shot:
                    if self.augmented_zero_shot is False:
                        return x_frozen, x_frozen  # zero-shot

                # New data is constantly being added, and in the case of repeat training, multiple expert
                if self.single_router is False:  
                    
                    # cls token
                    x_re = x.permute(1, 0, 2)[:, 0, :]

                    if self.is_train:  # train
                        # get mse_loss & z_score in DEeC
                        if self.no_tree_strategy:
                            self.get_block_mse_loss_and_z_score(x)
                        else:
                            self.get_block_mse_loss_and_z_score(x_original)

                        # get expansion signal
                        if self.expansion_flag is False:
                            if all(element > self.expansion_threshold for element in
                                    self.z_score[:self.global_expert_num]) or self.force_expansion:
                                self.add_expert(self.global_expert_num)
                                current_experts_num = self.global_expert_num
                        router = self.get_effective_experts(self.router_list[self.task_id], self.experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise_list[self.task_id], self.experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, self.global_expert_num)

                    else:  # eval in LEAS
                        if self.without_LEAS:
                            current_experts_num = self.global_expert_num
                        else:
                            current_experts_num = self.eval_update_expert_num(eval_task_id)
                        eval_experts_mask = self.eval_update_expert_mask(current_experts_num)
                        router = self.get_effective_experts(self.router_list[eval_task_id], eval_experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise_list[eval_task_id], eval_experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, current_experts_num)

                    # Number of statistical expert activations
                    nonzero_indices = torch.nonzero(gates) 
                    counter = Counter(nonzero_indices[:, 1].tolist())
                    for number, count in counter.items(): 
                        if self.text_or_image == 'text':
                            self.choose_map_text[number] = self.choose_map_text[number] + count
                        else:
                            self.choose_map_image[number] = self.choose_map_image[number] + count
                    
                    # Use router to distribute data according to the current number of experts
                    dispatcher = SparseDispatcher(int(current_experts_num), gates)
                    expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))
                    expert_outputs = [self.adaptmlp_list[i](expert_inputs[i]
                                                            .view(expert_inputs[i].shape[0], x.shape[0], x.shape[2])
                                                            .to(x), add_residual=False)
                                        for i in range(int(current_experts_num))]  

                    # reshape output
                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:  
                            expert_outputs.pop(i) 
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1

                else:  
                    # mutil-experts with 1 router
                    x_re = x.permute(1, 0, 2)[:, 0, :]  # cls token
                    
                    # training stage
                    if self.is_train: 
                        # get mse_loss & z_score in DEeC
                        if self.no_tree_strategy:
                            self.get_block_mse_loss_and_z_score(x)
                        else:
                            self.get_block_mse_loss_and_z_score(x_original)

                        # get expansion signal
                        if self.expansion_flag is False:
                            if all(element > self.expansion_threshold for element in
                                    self.z_score[:self.global_expert_num]) or self.force_expansion:
                                self.add_expert(self.global_expert_num)
                                current_experts_num = self.global_expert_num
                        router = self.get_effective_experts(self.router1, self.experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise1, self.experts_mask)
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, self.task_id)
                        
                        # Number of statistical expert activations
                        nonzero_indices = torch.nonzero(gates)  
                        counter = Counter(nonzero_indices[:, 1].tolist()) 
                        for number, count in counter.items(): 
                            if self.text_or_image == 'text':
                                self.choose_map_text[number] = self.choose_map_text[number] + count
                            else:
                                self.choose_map_image[number] = self.choose_map_image[number] + count
                                
                        # Use router to distribute data according to the current number of experts
                        dispatcher = SparseDispatcher(int(self.global_expert_num), gates)
                        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1)) 
                        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x), add_residual=False) for i in
                                            range(int(self.global_expert_num))] 
                    
                    # infreence stage
                    else:  
                        if self.without_LEAS:
                            current_experts_num = self.global_expert_num
                        else:
                            current_experts_num = self.eval_update_expert_num(eval_task_id)
                        eval_experts_mask = self.eval_update_expert_mask(current_experts_num)
                        router = self.get_effective_experts(self.router1, eval_experts_mask)
                        w_noise = self.get_effective_experts(self.w_noise1, eval_experts_mask)
                        
                        gates, load = self.adaptive_moe_gate(x_re, router, w_noise, current_experts_num)               
                        # Use router to distribute data according to the current number of experts
                        dispatcher = SparseDispatcher(int(current_experts_num), gates)
                        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).view(x.shape[1], -1))  #
                        expert_outputs = [self.adaptmlp_list[i](expert_inputs[i].view(expert_inputs[i].shape[0], x.shape[0], x.shape[2]).to(x), add_residual=False) for i in
                                            range(int(current_experts_num))] 
                        
                    # reshape output
                    i = 0
                    while i < len(expert_outputs):
                        if expert_outputs[i].shape[0] == 0:
                            expert_outputs.pop(i)
                        else:
                            expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                            i += 1
                            
                # Merge the output of each expert
                y = dispatcher.combine(expert_outputs) 
                y = y.view(x.shape[1], x.shape[0], x.shape[2])  # n x 39424 -> n x 77 x 512
                # Record the raw x_frozen of this layer for the input of the next layer of DEeC
                x = x + self.mlp(self.layer_norm2(x))  
                x = x + y.permute(1, 0, 2)

            else:  # one adapter, no Dynamic MoE-Adapters
                x_re = x.permute(1, 0, 2)
                adapt_x = self.adaptmlp(x_re, add_residual=False).permute(1, 0, 2)
                x = x + self.mlp(self.layer_norm2(x)) 
                x = x + adapt_x
        
        # original frozen CLIP, zero-shot
        else:  
            x = x + self.mlp(self.layer_norm2(x)) 
        return x, x_frozen

    def update_freq_activated_experts(self, new_tensor):
        """
        count the time & freq that experts are activated,
        for frozen & check the task ID
        Update the extra tensor e.g.
            new_tensor = torch.tensor([4.0, 5.0, 6.0])
            model.module.transformer.resblocks[i].update_extra_tensor(new_tensor)

        """
        self.expert_activate_freq_list[self.task_id] = new_tensor

    def add_expert(self, add_idx):
        self.experts_mask[add_idx] = 1.0
        self.global_expert_num += 1
        
        # If true, it has already been expanded once under the current task expert
        self.expansion_flag = True 
        print(
            f"In iter {self.current_iteration}, {self.text_or_image} layer {self.layer} add a new expert, now expert_num is {int(self.global_expert_num)}, trigger device: {self.activated_experts_num.device}")
        self.activated_experts_num += 1.0
        self.adaptive_moe_gate.update_expert_num(self.global_expert_num)
        
        # Refresh Expert Activation Status Record map
        if self.text_or_image == "text":
            self.choose_map_text = torch.zeros([self.max_expert_num])
        else:
            self.choose_map_image = torch.zeros([self.max_expert_num])
        if self.single_router:
            # Single router only unfreezes the column corresponding to the expert
            self.router1.register_hook(self.grad_hook)
            self.w_noise1.register_hook(self.grad_hook)
            pass

    # Define a hook to modify the gradient
    def grad_hook(self, grad):
        # Create an all-zero matrix with the same shape as grad
        mask = torch.zeros_like(grad)        
        # Keep only the gradient of the k-th column
        mask[:, self.global_expert_num - 1] = 1
        return grad * mask

    def add_LEAS(self, add_idx):
        self.experts_mask[add_idx] = 1.0
        self.global_expert_num += 1
        
        # If true, it has already been expanded once under the current task expert
        self.expansion_flag = True  
        print(
            f"{self.text_or_image} layer {self.layer} add a new LEAS, now LEAS_num is {int(self.global_expert_num)}")
        self.activated_experts_num += 1.0

    def get_block_mse_loss_and_z_score(self, x):
        # cls token
        # x_rd = x.permute(1, 0, 2)[:, 0, :]
        # x_rd = torch.concat(x[:, 0, :], x[:, 24, :], x[:, 48, :], x[:, 72, :], x[:, 96, :], x[:, 120, :], x[:, 144, :], x[:, -1, :])
        # indices = [0, 24, 48, 72, 96, 120, 144, -1]
        # x_rd = torch.cat([x[:, i, :] for i in indices], dim=1)
        # 方法2：均值池化
        x_rd = x.mean(dim=1)  # (bs, 768)

        # 方法3：最大值池化
        # x_rd = x.max(dim=1).values  # (bs, 768)
        self.eval_mse_list = [torch.zeros(size=[], device=x.device)] * self.max_expert_num 
        # update mse_loss & z_score in all AEs
        self.mse_loss = torch.zeros(size=[], device=x.device)
        self.z_score = [torch.zeros(size=[], device=x.device)] * self.max_expert_num
        for i in range(len(self.experts_mask)):
            if self.experts_mask[i] == 1.:
                # mse_loss & z-score in current layer
                mse_loss, z_score = self.auto_encoder_list[i](x_rd)
                self.mse_loss += mse_loss
                self.z_score[i] = z_score
                self.mse_loss_list[i] = mse_loss

                # inference stage 
                if self.is_train is False: 
                    if self.use_LEAS_to_eval:
                        self.eval_mse_list[i] = mse_loss.item()
                    else:
                        self.eval_mse_list[i] = (1 / mse_loss.item())

    def get_effective_experts(self, input, mask):
        """
        Depending on the value of the mask, the corresponding column of input is preserved.
        :param input: input tensor, shape [512, a].
        :param mask: mask tensor with length a only 1 or 0
        :return: tensor after selectively preserving columns.
        """
        # Selective column retention with mask
        effective_experts = input[:, mask.bool()]
        return effective_experts

    def update_mse_loss_avg_list(self, cut_off_rate_new, cut_off_rate_frozen=1.0):
        if self.adapter_flag or self.LEAS_layer_flag:
            new_idx = self.global_expert_num
            current_rd_avg_list = self.mse_loss_avg_list[self.task_id][:new_idx]
            # frozen
            for i in range(self.frozen_expert_num):
                current_mse_loss_avg = self.auto_encoder_list[i].get_avg_init(cut_off_rate_frozen)
                current_rd_avg_list[i] = current_mse_loss_avg

            new_mse_loss_avg = self.auto_encoder_list[new_idx - 1].get_avg_init(cut_off_rate_new)
            current_rd_avg_list[new_idx - 1] = new_mse_loss_avg
            # update
            self.mse_loss_avg_list[self.task_id][:new_idx] = current_rd_avg_list

    def update_mse_loss_std_list(self, cut_off_rate_new, cut_off_rate_frozen=1.0):
        if self.adapter_flag or self.LEAS_layer_flag:
            new_idx = self.global_expert_num
            current_rd_std_list = self.mse_loss_std_list[self.task_id][:new_idx]
            # frozen
            for i in range(self.frozen_expert_num):
                current_mse_loss_std = self.auto_encoder_list[i].get_std_init(cut_off_rate_frozen)
                current_rd_std_list[i] = current_mse_loss_std

            new_mse_loss_std = self.auto_encoder_list[new_idx - 1].get_std_init(cut_off_rate_new)
            current_rd_std_list[new_idx - 1] = new_mse_loss_std
            # update
            self.mse_loss_std_list[self.task_id][:new_idx] = current_rd_std_list

    def _get_ranking(self, lst):
        """
        Get the sorting information of a list
        """
        sorted_lst = sorted(enumerate(lst), key=lambda x: x[1])
        rank = [0] * len(lst)

        for rank_idx, (original_idx, value) in enumerate(sorted_lst):
            rank[original_idx] = rank_idx

        return rank

    def eval_task_id_LEAS(self, x):
        """
        Get the task_id corresponding to the eval batch by its discrepancy
        """
        zero_shot = False
        update_eval_zero_shot(False)
        eval_task_id = -1
        length = 0
        std_list = None
        # get mse_loss & z_score
        self.get_block_mse_loss_and_z_score(x)
        best_discrepancy = 1000.
        contrast_list = self.mse_loss_avg_list
        if self.discrepancy_weighted_vector == "STD" or self.discrepancy_weighted_vector == "STD_norm":
            std_list = self.mse_loss_std_list
        eval_mse_loss = torch.tensor(self.eval_mse_list, device=x.device)
        eval_mse_loss = self._trim_zeros(eval_mse_loss)
        for i, tensor in enumerate(contrast_list):
            tensor = self._trim_zeros(tensor)
            current_discrepancy = 1000.
            if tensor is not None:
                tensor_eval = eval_mse_loss[:len(tensor)]
                contrast_tensor = torch.maximum(tensor, tensor_eval)
                if self.discrepancy_weighted_vector == "CON":
                    current_discrepancy = self.discrepancy_con(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "CON_norm":
                    current_discrepancy = self.discrepancy_con_norm(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "STD":
                    current_discrepancy = self.discrepancy_std(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "STD_norm":
                    current_discrepancy = self.discrepancy_std_norm(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "Non_factor":
                    current_discrepancy = self.discrepancy_non(tensor, tensor_eval, contrast_tensor)
                else:
                    current_discrepancy = torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor), p=2) / sqrt(len(tensor))
                current_discrepancy = current_discrepancy + self.previous_discrepancy_list[i]
        
            # Update the maximum sum and the corresponding index
            if current_discrepancy < best_discrepancy:
                best_discrepancy = current_discrepancy
                eval_task_id = i

        if best_discrepancy < self.zero_shot_threshold:
            best_rd_list = contrast_list[eval_task_id]
            best_rd_list = self._trim_zeros(best_rd_list)
            length = len(best_rd_list)

        else:
            zero_shot = True
            update_eval_zero_shot(True)
            length = 0

        if self.print_eval_batches:
            print(best_discrepancy, eval_task_id, torch.argsort(eval_mse_loss))
        self.eval_discrepancy_list.append(best_discrepancy.item())
        return zero_shot, eval_task_id, length

    def discrepancy_con(self, tensor, tensor_eval, contrast_tensor):
        return torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor), p=2) / sqrt(len(tensor))
    
    def discrepancy_non(self, tensor, tensor_eval, contrast_tensor):
        return torch.norm((tensor - tensor_eval) / (contrast_tensor), p=2) / sqrt(len(tensor))

    def discrepancy_con_norm(self, tensor, tensor_eval, contrast_tensor):
        reciprocal_tensor = 1.0 / contrast_tensor
        norm_tensor = reciprocal_tensor / reciprocal_tensor.sum()
        return torch.norm((tensor - tensor_eval) * norm_tensor / contrast_tensor, p=2) / sqrt(len(tensor))

    def discrepancy_std(self, tensor, tensor_eval, contrast_tensor, std):
        return torch.norm((tensor - tensor_eval) / (std * contrast_tensor), p=2) / sqrt(len(tensor)) / 1000

    def discrepancy_std_norm(self, tensor, tensor_eval, contrast_tensor, std):
        reciprocal_tensor = 1.0 / std
        norm_tensor = reciprocal_tensor / reciprocal_tensor.sum()
        return torch.norm((tensor - tensor_eval) * norm_tensor / contrast_tensor, p=2) / sqrt(len(tensor))

    def get_best_discrepancy_list(self, x):
        std_list = None
        # clear list
        discrepancy_list = [torch.tensor(0.).to(x)] * self.task_num
        # get mse_loss & z_score
        self.get_block_mse_loss_and_z_score(x)
        contrast_list = self.mse_loss_avg_list
        eval_mse_loss = torch.tensor(self.eval_mse_list, device=x.device)
        eval_mse_loss = self._trim_zeros(eval_mse_loss)
        if self.discrepancy_weighted_vector == "STD" or self.discrepancy_weighted_vector == "STD_norm":
            std_list = self.mse_loss_std_list
        for i, tensor in enumerate(contrast_list):
            tensor = self._trim_zeros(tensor)
            if tensor is not None:
                tensor_eval = eval_mse_loss[:len(tensor)]
                contrast_tensor = torch.maximum(tensor, tensor_eval)
                if self.discrepancy_weighted_vector == "CON":
                    current_discrepancy = self.discrepancy_con(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "CON_norm":
                    current_discrepancy = self.discrepancy_con_norm(tensor, tensor_eval, contrast_tensor)
                elif self.discrepancy_weighted_vector == "STD":
                    current_discrepancy = self.discrepancy_std(tensor, tensor_eval, contrast_tensor, std_list[i][:len(tensor)])
                elif self.discrepancy_weighted_vector == "STD_norm":
                    current_discrepancy = self.discrepancy_std_norm(tensor, tensor_eval, contrast_tensor,
                                                                            std_list[i][:len(tensor)])
                else:
                    current_discrepancy = torch.norm((tensor - tensor_eval) / (contrast_tensor * contrast_tensor),
                                                    p=2) / sqrt(len(tensor))
                discrepancy_list[i] = current_discrepancy
        return discrepancy_list

    def _trim_zeros(self, tensor):
        """
        Intercepts the portion of the tensor that is not preceded by 0.

        Params:
        tensor (torch.Tensor): the input one-dimensional tensor

        Returns:
        torch.Tensor: the intercepted tensor
        """
        # Find the index of all non-zero elements
        non_zero_indices = torch.nonzero(tensor).squeeze()

        if len(non_zero_indices) > 0:
            first_zero_index = non_zero_indices[-1].item() + 1
            return tensor[:first_zero_index]
        else:
            return None

    def eval_update_expert_mask(self, current_experts_num):
        mask = self.experts_mask.clone()
        mask[current_experts_num:] = 0.
        return mask

    def eval_update_expert_num(self, eval_task_id):
        contrast_list = self.expert_activate_freq_list[eval_task_id]
        contrast_list = self._trim_zeros(contrast_list)
        return len(contrast_list)

    def get_convergence_value(self, tensor, method='slide_window', kernel_size=3, alpha=0.2):
        """
        Smooths the input 1D tensor, and returns the last value after smoothing.

        Parameters.
        tensor (torch.Tensor): the input one-dimensional tensor
        method (str): smoothing method, supports 'slide_window' and 'ewma'.
        kernel_size (int): window size of the moving average (only works with ‘moving_average’ method)
        alpha (float): smoothing factor (only available in 'ewma' method)

        Returns: torch.
        torch.Tensor: last value of the smoothed tensor
        """
        if method == 'slide_window':
            # Moving Average (using convolution)
            if kernel_size > len(tensor):
                raise ValueError("kernel_size must be smaller than or equal to the length of the tensor.")

            kernel = torch.ones(kernel_size) / kernel_size
            kernel = kernel.view(1, 1, -1)

            tensor_unsqueezed = tensor.view(1, 1, -1)  
            smoothed_tensor = F.conv1d(tensor_unsqueezed, kernel, padding=kernel_size // 2)
            smoothed_tensor = smoothed_tensor.view(-1)  

            return smoothed_tensor[-1]  

        elif method == 'ewma':
            smoothed_tensor = torch.zeros_like(tensor)
            smoothed_tensor[0] = tensor[0]  #

            for i in range(1, len(tensor)):
                smoothed_tensor[i] = alpha * tensor[i] + (1 - alpha) * smoothed_tensor[i - 1]

            return smoothed_tensor[-1] 
        
        

