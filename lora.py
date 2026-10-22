import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Literal, Union
from safetensors.torch import save_file


class LoRALayerBase:

    def __init__(self, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0, 
                 use_rslora=True):
        
        self.rank = rank
        self.lora_alpha = lora_alpha 
        self.scaling = self.lora_alpha / self.rank**0.5 if use_rslora else self.lora_alpha / self.rank
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else lambda x: x

    def _load_pretrained_weights(self, state_dict):
        self.weight.data = state_dict["weight"]
        if "bias" in state_dict.keys():
            self.bias.data = state_dict["bias"]

class LoRALinear(nn.Linear, LoRALayerBase):

    def __init__(self, 
                 in_features, 
                 out_features,
                 bias=True, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        self.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):

        merged_weight = self.weight.data + (self.lora_A @ self.lora_B).T * self.scaling

        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias
            
        ### Load Weights to New Linear Layer ###
        merged_linear = nn.Linear(self.in_features, 
                                  self.out_features,
                                  bias=True if self.bias is not None else False)
        
        merged_linear.load_state_dict(state_dict)

        return merged_linear

    def forward(self, x):
        
        orig_layer_out = F.linear(x, self.weight, bias=self.bias)

        lora_mult = (self.lora_A @ self.lora_B) * self.scaling
        low_rank_out = self.lora_dropout(x) @ lora_mult

        output = orig_layer_out + low_rank_out
        
        return output

class LoRAEmbedding(nn.Embedding, LoRALayerBase):

    def __init__(self, 
                 num_embeddings, 
                 embedding_dim, 
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)
        
        self.weight.requires_grad = False
        
        self.lora_A = nn.Parameter(torch.zeros(self.num_embeddings, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.embedding_dim))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):
        merged_weights = self.weight.data + (self.lora_A @ self.lora_B) * self.scaling
        state_dict = {"weight": merged_weights}
        merged_emb = nn.Embedding(self.num_embeddings, self.embedding_dim)
        merged_emb.load_state_dict(state_dict)

        return merged_emb


    def forward(self, x):

        orig_layer_out = F.embedding(input=x, 
                                     weight=self.weight,
                                     padding_idx=self.padding_idx, 
                                     max_norm=self.max_norm, 
                                     norm_type=self.norm_type, 
                                     scale_grad_by_freq=self.scale_grad_by_freq, 
                                     sparse=self.sparse)
        low_rank_A_output = F.embedding(input=x, 
                                        weight=self.lora_A, 
                                        padding_idx=self.padding_idx, 
                                        max_norm=self.max_norm, 
                                        norm_type=self.norm_type, 
                                        scale_grad_by_freq=self.scale_grad_by_freq, 
                                        sparse=self.sparse)
        low_rank_output = (low_rank_A_output @ self.lora_B) * self.scaling

        output = orig_layer_out + low_rank_output

        return output

class LoRAConv2d(nn.Conv2d, LoRALayerBase):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 bias=True,
                 rank=8, 
                 lora_alpha=8, 
                 lora_dropout=0.0,
                 use_rslora=True,
                 **kwargs):
        
        nn.Conv2d.__init__(self,
                           in_channels=in_channels, 
                           out_channels=out_channels, 
                           kernel_size=kernel_size, 
                           stride=stride, 
                           padding=padding, 
                           bias=bias, 
                           **kwargs)
        LoRALayerBase.__init__(self,
                               rank=rank, 
                               lora_alpha=lora_alpha, 
                               lora_dropout=lora_dropout, 
                               use_rslora=use_rslora)

        self.weight.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_channels, *self.kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_channels))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _merge_weights(self):   
        lora_A_flatten = self.lora_A.flatten(1)
        lora_mult = (self.lora_B.T @ lora_A_flatten) * self.scaling
        lora_mult = lora_mult.reshape(self.out_channels, self.in_channels, *self.kernel_size)
        merged_weight = self.weight.data + lora_mult


        state_dict = {"weight": merged_weight}
        if self.bias is not None:
            state_dict["bias"] = self.bias

        merged_conv = nn.Conv2d(self.in_channels, 
                                self.out_channels,
                                kernel_size=self.kernel_size, 
                                stride=self.stride, 
                                padding=self.padding,
                                bias=True if self.bias is not None else False)
        
        merged_conv.load_state_dict(state_dict)

        return merged_conv

    def forward(self, x):
        orig_layer_out = F.conv2d(input=x, 
                                  weight=self.weight, 
                                  bias=self.bias,
                                  stride=self.stride, 
                                  padding=self.padding)
        
        lora_rank_A_output = F.conv2d(input=x, 
                                      weight=self.lora_A, 
                                      bias=None,
                                      stride=self.stride, 
                                      padding=self.padding)

        lora_rank_A_output = lora_rank_A_output.permute(0,2,3,1)
        low_rank_output = (self.lora_dropout(lora_rank_A_output) @ self.lora_B) * self.scaling
        low_rank_output = low_rank_output.permute(0,3,1,2)
        output = orig_layer_out + low_rank_output

        return output
  
  
@dataclass
class LoraConfig:
    
    rank: int = 8
    target_modules: Optional[Union[list[str], str]] = None
    exclude_modules: Optional[Union[list[str], str]] = None
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = True

class LoraModel(nn.Module):

    def __init__(self, model, config):
        super(LoraModel, self).__init__()

        self.lora_model = model
        self.config = config

        if self.config.target_modules is None:
            self.config.target_modules = []
        elif isinstance(self.config.target_modules, str):
            self.config.target_modules = [self.config.target_modules]
        if self.config.exclude_modules is None:
            self.config.exclude_modules = []
        elif isinstance(self.config.exclude_modules, str):
            self.config.exclude_modules = [self.config.exclude_modules]

        orig_trainable_params = self._compute_trainable_parameters()

        self._disable_all_grads()

        self._apply_lora(self.lora_model)

        self._toggle_bias_grad()

        lora_trainable_params = self._compute_trainable_parameters()

        print_string = ""
        print_string += f"Initial Parameters : {orig_trainable_params} || "
        print_string += f"LoRA Parameters : {lora_trainable_params} || "
        print_string += f"Trainable Proportion : {round(lora_trainable_params*100/orig_trainable_params, 2)}%"

        print(print_string)

    def forward(self, *inputs, **kwargs):

        return self.lora_model(*inputs, **kwargs)
    
    def _exclude_module_name_check(self, name):
        return any([ex in name for ex in self.config.exclude_modules])
    
    def _target_module_name_check(self, name):
        return any([tgt in name for tgt in self.config.target_modules])
    
    def _apply_lora(self, module):
        for name, child in module.named_children():
            if self._target_module_name_check(name):
                if isinstance(child, nn.Linear):

                    new_layer = LoRALinear(in_features=child.in_features, 
                                           out_features=child.out_features, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank,
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora)

                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Conv2d):

                    new_layer = LoRAConv2d(in_channels=child.in_channels, 
                                           out_channels=child.out_channels, 
                                           kernel_size=child.kernel_size, 
                                           stride=child.stride, 
                                           padding=child.padding, 
                                           bias=True if child.bias is not None else False,
                                           rank=self.config.rank, 
                                           lora_alpha=self.config.lora_alpha, 
                                           lora_dropout=self.config.lora_dropout, 
                                           use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

                elif isinstance(child, nn.Embedding):

                    new_layer = LoRAEmbedding(num_embeddings=child.num_embeddings, 
                                              embedding_dim=child.embedding_dim, 
                                              rank=self.config.rank, 
                                              lora_alpha=self.config.lora_alpha, 
                                              lora_dropout=self.config.lora_dropout, 
                                              use_rslora=self.config.use_rslora)
                    
                    new_layer._load_pretrained_weights(child.state_dict())
                    setattr(module, name, new_layer)

            if (len(list(child.children())) > 0) and not any([ex in name for ex in self.config.exclude_modules]):
                self._apply_lora(child)

    def _toggle_bias_grad(self):

        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                if ".bias" in name:
                    if self.config.bias == "none":
                        param.requires_grad = False
                    elif self.config.bias == "all":
                        param.requires_grad = True
                    elif (self.config.bias == "lora_only") and self._target_module_name_check(name):
                        param.requires_grad = True

    def _disable_all_grads(self):
        
        for name, param in self.lora_model.named_parameters():
            if not self._exclude_module_name_check(name):
                param.requires_grad = False

    def _compute_trainable_parameters(self):

        total_learnable_params = 0
        for param in self.lora_model.parameters():
            if param.requires_grad:
                total_learnable_params += param.numel()

        return total_learnable_params

    def _merge_weights(self, module):

        for name, child in module.named_children():

            if isinstance(child, (LoRALinear, LoRAEmbedding, LoRAConv2d)):
                 merged_layer = child._merge_weights()
                 setattr(module, name, merged_layer)

            else:

                if len(list(child.children())) > 0:
                    self._merge_weights(child)

    def save_model(self, path, merge_weights=False):

        def _detach_cpu(param):
            return param.detach().cpu()
        if merge_weights:
            self._merge_weights(self.lora_model)
            state_dict = {name.replace("lora_model.", ""): _detach_cpu(param) for (name, param) in self.named_parameters()}
        else:

            state_dict = {name: _detach_cpu(param) for (name, param) in self.named_parameters() if (param.requires_grad)}

        save_file(state_dict, path)   
       
if __name__ == "__main__":
    
    from transformers import AutoModelForSequenceClassification
    
    target_modules = ["query", "key", "value","dense","word_embeddings"]
    exclude_modules = "classifier"
    
    config = LoraConfig(target_modules=target_modules, exclude_modules= exclude_modules, bias="lora_only")
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
    lora_model = LoraModel(model, config)