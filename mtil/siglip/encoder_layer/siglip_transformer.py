# --------------------------------------------------------
# Implementation of the transformer for Text/Image Encoder
# change the output of the transformer to two outputs
# --------------------------------------------------------

import torch
from torch import nn

from .moe_layer import DynResidualAttentionBlock, update_val_task_id_text, update_val_task_id_visual
from .original_moe_layer import ResidualAttentionBlock

from typing import Optional, Union, Tuple
from transformers.modeling_outputs import BaseModelOutput


class Transformer(nn.Module):
    """
    To match the structure of MoE-Adapter++,
    change the output of the transformer to two outputs
    Encoder in Siglip
    """
    def __init__(self, 
                #  width: int, 
                #  layers: int, 
                #  heads: int, 
                #  attn_mask: torch.Tensor = None, 
                #  adapter_flag=True,
                 args=None, 
                 text_or_image=None, 
                 hf_config=None):
        super().__init__()
        self.hf_config = hf_config
        self.width = hf_config.hidden_size
        self.layer_num = hf_config.num_hidden_layers
        # self.adapter_flag = adapter_flag
        self.dyn_moe = args.dyn_moe
        self.layer_lock = args.force_layer_lock
        
        # if true, only use the discrepancy for last recognition layer 
        self.mutil_LEAS_lock = args.mutil_LEAS_lock 
        
        self.use_LEAS_to_eval = args.use_LEAS_to_eval
        if text_or_image == "text":
            self.layer_adapter_flag_list = args.use_dyn_moe_layer_list_text
            self.LEAS_flag_list = args.use_LEAS_list_text
        else:
            self.layer_adapter_flag_list = args.use_dyn_moe_layer_list_visual
            self.LEAS_flag_list = args.use_LEAS_list_visual
        if self.dyn_moe:
            self.layers = nn.Sequential(
                *[DynResidualAttentionBlock(
                    # d_model=width,
                    # n_head=heads,
                    # attn_mask=attn_mask,
                    adapter_flag=self.layer_adapter_flag_list[i],
                    LEAS_flag=self.LEAS_flag_list[i],
                    args=args,
                    text_or_image=text_or_image,
                    i=i,
                    hf_config=hf_config
                    ) for i in
                  range(self.layer_num)])
        else:
            self.layers = nn.Sequential(
                *[ResidualAttentionBlock(
                    # width, 
                    # heads, 
                    # attn_mask, 
                    # self.layer_adapter_flag_list[i], 
                    adapter_flag=True,
                    args=args, 
                    text_or_image=text_or_image, 
                    i=i, 
                    hf_config=hf_config) for i in
                  range(self.layer_num)])

    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                ) -> Union[Tuple, BaseModelOutput]:
        # 从 hf_config 获取默认参数（对齐 SiglipEncoder 的逻辑）
        
        output_attentions = self.hf_config.output_attentions if hasattr(self.hf_config, 'output_attentions') else False
        output_hidden_states = self.hf_config.output_hidden_states if hasattr(self.hf_config, 'output_hidden_states') else False
        return_dict = self.hf_config.use_return_dict if hasattr(self.hf_config, 'use_return_dict') else False

        # 初始化隐藏状态和注意力权重收集器
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # 动态 MoE 逻辑（保持原有代码）
        if self.dyn_moe:
            update_val_task_id_visual(-1, 0)
            update_val_task_id_text(-1, 0)
            x_original = x.clone()
            hidden_states = x  # 输入 x 对应 SiglipEncoder 的 inputs_embeds
            for i in range(self.layer_num):
                if self.layers[i].reconginition_layer and not self.mutil_LEAS_lock:
                    self.trans_previous_discrepancy_to_reconginition_layer()
              
                # 假设 DynResidualAttentionBlock 返回 (hidden_states, x_original, attention_probs)
                layer_outputs = self.layers[i](
                    hidden_states,
                    x_original,
                    # attention_mask=attention_mask,  # 若需要 attention_mask，需在调用时传入
                    # output_attentions=output_attentions
                )
                hidden_states = layer_outputs[0]
                x_original = layer_outputs[1]
              
                # 收集隐藏状态和注意力权重
                if output_hidden_states:
                    encoder_states += (hidden_states,)
                if output_attentions:
                    all_attentions += (layer_outputs[2],)
          
            # 层锁定逻辑（保持原有代码）
            if self.layer_lock:  
                for i in range(self.layer_num - 1):
                    if not self.layers[i].expansion_flag and self.layers[i+1].expansion_flag:
                        for j in range(i + 1):
                            self.layers[j].expansion_flag = True
                        print(f"Lock layers before {self.layers[i].text_or_image} layer {i}")
        else:
            # 非动态 MoE 逻辑
            hidden_states = x  # 输入 x 对应 SiglipEncoder 的 inputs_embeds
            for i in range(self.layer_num):
                # 假设 ResidualAttentionBlock 返回 (hidden_states, attention_probs)
                layer_outputs = self.layers[i](
                    x=hidden_states,
                    # attention_mask=attention_mask,  # 若需要 attention_mask，需在调用时传入
                    # output_attentions=output_attentions
                )
                hidden_states = layer_outputs
              
                # 收集隐藏状态和注意力权重
                if output_hidden_states:
                    encoder_states += (hidden_states,)
                if output_attentions:
                    all_attentions += (layer_outputs,)
            # hidden_states = self.layers(hidden_states)
            # encoder_states = None
            # all_attentions = None


        # 返回格式对齐 SiglipEncoder
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        else:
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=encoder_states,
                attentions=all_attentions
            )

    def trans_previous_discrepancy_to_reconginition_layer(self):
        result = []
        previous_list = []
        for i in range(self.layer_num):
            if self.layers[i].LEAS_layer_flag:
                previous_list.append(self.layers[i].previous_discrepancy_list)
            if self.layers[i].reconginition_layer and len(previous_list) != 0:
                # Iterate over each position and add together the tensor at the corresponding position
                for idx in range(len(previous_list[0])):
                    sum_tensor = sum(lst[idx] for lst in previous_list)
                    result.append(sum_tensor)
                self.layers[i].previous_discrepancy_list = result


