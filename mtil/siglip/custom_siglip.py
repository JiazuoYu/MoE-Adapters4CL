# --------------------------------------------------------
# Implementation of Siglip with MoE-Adapters & MoE-Adapters++
# --------------------------------------------------------
from .siglip.modeling_siglip import SIGLIP_TEXT_INPUTS_DOCSTRING, SIGLIP_VISION_INPUTS_DOCSTRING, SiglipEncoder, SiglipModel, SiglipMultiheadAttentionPoolingHead, SiglipOutput, SiglipPreTrainedModel, SiglipTextEmbeddings, SiglipVisionEmbeddings
from transformers import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from .encoder_layer.siglip_transformer import Transformer
import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .encoder_layer.original_moe_layer import set_val_task_id


class CustomSiglipConfig(SiglipConfig):
    def __init__(self, 
                 *args, 
                 # 使用一个字典来存储所有额外的自定义参数
                 custom_args: dict = None, 
                 **kwargs):
        super().__init__(*args, **kwargs)
      
        # 处理自定义参数
        if custom_args is None:
            custom_args = {}
        self.custom_args = custom_args
        

class CustomSiglipModel(SiglipPreTrainedModel):
    def __init__(
        self,
        config: SiglipConfig,
        args,   # custom args
    ):
        super().__init__(config=config)  # 完成模型的自定义
        if not isinstance(config.text_config, SiglipTextConfig):
            raise ValueError(
                "config.text_config is expected to be of type SiglipTextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, SiglipVisionConfig):
            raise ValueError(
                "config.vision_config is expected to be of type SiglipVisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config
        self.args = args

        self.text_model = SiglipTextTransformer(config=text_config, args=args)  # custom
        self.vision_model = SiglipVisionTransformer(config=vision_config, args=args)  # custom

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))

        # Initialize weights and apply final processing
        self.post_init()
        
    def get_text_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        val_task_id: Optional[int] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

        >>> # important: make sure to set padding="max_length" as that's how the model was trained
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt")
        >>> with torch.no_grad():
        ...     text_features = model.get_text_features(**inputs)
        ```"""
        if val_task_id is not None:  # moe v1
            # print("model_val_task_id: ", val_task_id)
            if self.args.dyn_moe is False:
                set_val_task_id(val_task_id)
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = text_outputs[1]

        return pooled_output

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        val_task_id: Optional[int] = None,
        # normlized: Optional[bool] = None,
    ) -> torch.FloatTensor:
        r"""
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`SiglipVisionModel`].

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> with torch.no_grad():
        ...     image_features = model.get_image_features(**inputs)
        ```"""
        if val_task_id is not None:
            print("model_val_task_id: ", val_task_id)
            if self.args.dyn_moe is False:  # moe v1
                set_val_task_id(val_task_id)
        
        # Use SiglipModel's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]

        return pooled_output
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        return_ce_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        mode: Optional[str] = None,
    ) -> Union[Tuple, SiglipOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, AutoModel
        >>> import torch

        >>> model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        >>> processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> texts = ["a photo of 2 cats", "a photo of 2 dogs"]
        >>> # important: we pass `padding=max_length` since the model was trained with this
        >>> inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> logits_per_image = outputs.logits_per_image
        >>> probs = torch.sigmoid(logits_per_image) # these are the probabilities
        >>> print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
        31.9% that image 0 is 'a photo of 2 cats'
        ```"""
        if mode == "text":
            return self.get_text_features(input_ids, attention_mask)
        
        if mode == "image":
            return self.get_image_features(pixel_values)
        
        # Use SigLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs.pooler_output
        text_embeds = text_outputs.pooler_output

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale.exp() + self.logit_bias
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            # Adapted from https://github.com/google-research/big_vision/blob/01edb81a4716f93a48be43b3a4af14e29cdb3a7f/big_vision/trainers/proj/image_text/siglip.py#L287
            eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
            m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
            loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
            nll = -torch.sum(loglik, dim=-1)
            loss = nll.mean()
            
        # if return_loss:
        #     raise NotImplementedError("SigLIP loss to be implemented")

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return SiglipOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
        
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(64, 64)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

class SiglipTextTransformer(nn.Module):
    def __init__(self, config: SiglipTextConfig, args=None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipTextEmbeddings(config)
        self.encoder = Transformer(args=args, text_or_image="text", hf_config=config)
        # self.encoder = SiglipEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.head = nn.Linear(embed_dim, embed_dim)

    @add_start_docstrings_to_model_forward(SIGLIP_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipTextConfig)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # note: SigLIP's text model does not use a causal mask, unlike the original CLIP model.
        # expand attention_mask
        if attention_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            x=hidden_states,
            attention_mask=attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # Assuming "sticky" EOS tokenization, last token is always EOS.
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.head(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig, args=None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        # self.encoder = SiglipEncoder(config)
        self.encoder = Transformer(args=args, text_or_image="image", hf_config=config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.head = SiglipMultiheadAttentionPoolingHead(config)

    @add_start_docstrings_to_model_forward(SIGLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=SiglipVisionConfig)
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            x=hidden_states,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)

        pooled_output = self.head(last_hidden_state)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
