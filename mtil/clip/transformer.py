# --------------------------------------------------------
# Implementation of the transformer for Text/Image Encoder
# change the output of the transformer to two outputs
# --------------------------------------------------------

import torch
from torch import nn

from .moe_layer import DynResidualAttentionBlock, update_val_task_id_text, update_val_task_id_visual
from .original_moe_layer import ResidualAttentionBlock


class Transformer(nn.Module):
    """
    To match the structure of MoE-Adapter++,
    change the output of the transformer to two outputs
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, adapter_flag=True,
                 args=None, text_or_image=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.adapter_flag = adapter_flag
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
            self.resblocks = nn.Sequential(
                *[DynResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    attn_mask=attn_mask,
                    adapter_flag=self.layer_adapter_flag_list[i],
                    LEAS_flag=self.LEAS_flag_list[i],
                    args=args,
                    text_or_image=text_or_image,
                    i=i) for i in
                  range(layers)])
        else:
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock(width, heads, attn_mask, True, args, text_or_image, i) for i in
                  range(layers)])
            # self.resblocks = nn.Sequential(
                # *[ResidualAttentionBlock(width, heads, attn_mask, self.layer_adapter_flag_list[i], args, text_or_image, i) for i in
                #   range(layers)])

    def forward(self, x: torch.Tensor):

        if self.dyn_moe:
            # Refresh the task_id of each batch.
            update_val_task_id_visual(-1, 0)
            update_val_task_id_text(-1, 0)
            x_original = x.clone()
            for i in range(self.layers):

                if self.resblocks[i].reconginition_layer and self.mutil_LEAS_lock is False:
                    self.trans_previous_discrepancy_to_reconginition_layer()
                x, x_original = self.resblocks[i](x, x_original)
            output = x
            output_original = x_original

            if self.layer_lock:    
            # Lock all previous layers that have not had an expert added to them and only allow the later layers to expand
                for i in range(self.layers - 1):
                    if self.resblocks[i].expansion_flag is False and self.resblocks[i+1].expansion_flag:
                        for j in range(i + 1):
                            self.resblocks[j].expansion_flag = True  
                        print(f"lock layers before the {self.resblocks[i].text_or_image} layer {i}")
        else:
            output = self.resblocks(x)
            output_original = output
        return output, output_original

    def trans_previous_discrepancy_to_reconginition_layer(self):
        result = []
        previous_list = []
        for i in range(self.layers):
            if self.resblocks[i].LEAS_layer_flag:
                previous_list.append(self.resblocks[i].previous_discrepancy_list)
            if self.resblocks[i].reconginition_layer and len(previous_list) != 0:
                # Iterate over each position and add together the tensor at the corresponding position
                for idx in range(len(previous_list[0])):
                    sum_tensor = sum(lst[idx] for lst in previous_list)
                    result.append(sum_tensor)
                self.resblocks[i].previous_discrepancy_list = result


