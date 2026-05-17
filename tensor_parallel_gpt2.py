"""
Tensor Parallel GPT-2 Model
Corrected weight loading for Conv1D layers
"""
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import GPT2Config, GPT2LMHeadModel
from models.tensor_parallel_layers import TensorParallelGPT2Block, ColumnParallelLinear

class TensorParallelGPT2(nn.Module):
    def __init__(self, config, flash_allreduce=None):
        super().__init__()
        self.config = config
        self.flash_allreduce = flash_allreduce
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Pad vocab for tensor parallelism
        self.padded_vocab_size = math.ceil(config.vocab_size / self.world_size) * self.world_size
        self.vocab_per_partition = self.padded_vocab_size // self.world_size

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)

        self.h = nn.ModuleList([
            TensorParallelGPT2Block(config, flash_allreduce)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = ColumnParallelLinear(config.n_embd, self.padded_vocab_size, bias=False)

        self._tie_weights()

    def _tie_weights(self):
        """Ties weights between embedding and sharded LM head."""
        vocab_start = self.rank * self.vocab_per_partition
        vocab_end = min(vocab_start + self.vocab_per_partition, self.config.vocab_size)
        real_tokens = vocab_end - vocab_start

        tied = torch.zeros(self.vocab_per_partition, self.config.n_embd)
        tied[:real_tokens] = self.wte.weight[vocab_start:vocab_end].detach().clone()
        self.lm_head.weight = nn.Parameter(tied)

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        hidden_states = self.drop(self.wte(input_ids) + self.wpe(position_ids))

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)

        if labels is not None:
            vocab_start = self.rank * self.vocab_per_partition
            local_labels = labels.clone()
            in_partition = (labels >= vocab_start) & (labels < vocab_start + self.vocab_per_partition)
            local_labels[~in_partition] = -100
            local_labels[in_partition] -= vocab_start

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, self.vocab_per_partition), 
                local_labels.view(-1), 
                reduction='sum',
                ignore_index=-100
            )
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            # FIX: divide by valid tokens only, not all tokens including padding
            valid_tokens = (labels != -100).sum().float()
            dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM)
            return loss / valid_tokens

        return logits

    @staticmethod
    def from_pretrained(model_name, flash_allreduce=None):
        config = GPT2Config.from_pretrained(model_name)
        model = TensorParallelGPT2(config, flash_allreduce)
        
        pretrained_state = None
        if dist.get_rank() == 0:
            pretrained_state = GPT2LMHeadModel.from_pretrained(model_name).state_dict()
        
        model._load_and_split_weights(pretrained_state)
        return model

    def _load_and_split_weights(self, pretrained_state):
        """Broadcasts sharded weights to all nodes."""
        for name, param in self.named_parameters():
            if any(x in name for x in ['wte', 'wpe', 'ln_']):
                hf_key = 'transformer.' + name
                if self.rank == 0: param.data.copy_(pretrained_state[hf_key])
                dist.broadcast(param.data, src=0)

        for i, block in enumerate(self.h):
            self._load_column_parallel(block.attn.c_attn, pretrained_state, f'transformer.h.{i}.attn.c_attn')
            self._load_row_parallel(block.attn.c_proj, pretrained_state, f'transformer.h.{i}.attn.c_proj')
            self._load_column_parallel(block.mlp.c_fc, pretrained_state, f'transformer.h.{i}.mlp.c_fc')
            self._load_row_parallel(block.mlp.c_proj, pretrained_state, f'transformer.h.{i}.mlp.c_proj')

    def _load_column_parallel(self, module, state_dict, prefix):
        """Splits weights along the output dimension, handling QKV specifically."""
        weight_chunks, bias_chunks = None, None
        
        if self.rank == 0:
            # GPT-2 weights are [In, Out] (Conv1D) -> Transpose to [Out, In]
            w = state_dict[f'{prefix}.weight'].t().contiguous()
            b = state_dict[f'{prefix}.bias']
            
            if "c_attn" in prefix:
                # w is [3*hidden, hidden] -> [3, hidden, hidden]
                w = w.view(3, self.config.n_embd, self.config.n_embd)
                w_chunks = torch.chunk(w, self.world_size, dim=1)
                weight_chunks = [c.reshape(-1, self.config.n_embd).contiguous() for c in w_chunks]
                
                b = b.view(3, self.config.n_embd)
                b_chunks = torch.chunk(b, self.world_size, dim=1)
                bias_chunks = [c.reshape(-1).contiguous() for c in b_chunks]
            else:
                # Standard linear (c_fc)
                weight_chunks = list(torch.chunk(w, self.world_size, dim=0))
                bias_chunks   = list(torch.chunk(b, self.world_size, dim=0))

        dist.scatter(module.weight.data, weight_chunks, src=0)
        if module.bias is not None:
            dist.scatter(module.bias.data, bias_chunks, src=0)

    def _load_row_parallel(self, module, state_dict, prefix):
        """Splits weights along the input dimension."""
        weight_chunks = None
        if self.rank == 0:
            # GPT-2 weights are [In, Out] (Conv1D) -> Transpose to [Out, In]
            w = state_dict[f'{prefix}.weight'].t().contiguous()
            weight_chunks = list(torch.chunk(w, self.world_size, dim=1))

        dist.scatter(module.weight.data, weight_chunks, src=0)
        if module.bias is not None and self.rank == 0:
            module.bias.data.copy_(state_dict[f'{prefix}.bias'])