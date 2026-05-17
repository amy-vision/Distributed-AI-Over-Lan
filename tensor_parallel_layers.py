"""
Tensor Parallel Layers for GPT-2
Split model across multiple nodes
"""
import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    Splits weight along output dimension:
      Full:    [out, in]
      Rank k:  [out/world_size, in]

    No communication needed in forward — each node computes a partial output.
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        self.world_size = dist.get_world_size()
        self.rank       = dist.get_rank()

        assert out_features % self.world_size == 0, (
            f"out_features ({out_features}) must be divisible by world_size ({self.world_size})"
        )
        self.out_features_per_partition = out_features // self.world_size

        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        """
        Input:  [batch, seq, in_features]
        Output: [batch, seq, out_features_per_partition]   (no comm)
        """
        return torch.nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    Splits weight along input dimension:
      Full:    [out, in]
      Rank k:  [out, in/world_size]

    Requires Flash All-Reduce after local matmul to sum partial results.
    """

    def __init__(self, in_features, out_features, flash_allreduce=None, bias=True):
        super().__init__()

        self.in_features    = in_features
        self.out_features   = out_features
        self.flash_allreduce = flash_allreduce

        self.world_size = dist.get_world_size()
        self.rank       = dist.get_rank()

        assert in_features % self.world_size == 0, (
            f"in_features ({in_features}) must be divisible by world_size ({self.world_size})"
        )
        self.in_features_per_partition = in_features // self.world_size

        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        # Bias only on rank 0 — added once after the All-Reduce
        if bias and self.rank == 0:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        """
        Input:  [batch, seq, in_features_per_partition]
        Output: [batch, seq, out_features]   (after All-Reduce)
        """
        # Local matmul — partial result
        output = torch.nn.functional.linear(x, self.weight, None)

        # Flash All-Reduce: sum partial results across all nodes
        if self.flash_allreduce is not None:
            output = self.flash_allreduce.all_reduce(output)
        else:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        # Add bias on rank 0 only, then broadcast to all ranks
        # NOTE: dist.broadcast must be OUTSIDE the if-block so all ranks call
        # it unconditionally — mismatched collective calls cause deadlocks
        if self.bias is not None:
            output = output + self.bias
        dist.broadcast(output, src=0)   # all ranks call this, not just rank 0

        return output


class TensorParallelGPT2Attention(nn.Module):
    """
    GPT-2 Multi-head Attention with Tensor Parallelism.

    QKV projection:    Column-parallel — each node computes its head subset, no comm
    Output projection: Row-parallel   — Flash All-Reduce to sum partial results
    """

    def __init__(self, config, flash_allreduce=None):
        super().__init__()

        self.hidden_size = config.n_embd
        self.num_heads   = config.n_head
        self.head_dim    = self.hidden_size // self.num_heads

        self.world_size = dist.get_world_size()
        self.rank       = dist.get_rank()

        assert self.num_heads % self.world_size == 0, (
            f"num_heads ({self.num_heads}) must be divisible by world_size ({self.world_size})"
        )
        self.num_heads_per_partition = self.num_heads // self.world_size

        # QKV — column-parallel (no comm after)
        self.c_attn = ColumnParallelLinear(self.hidden_size, 3 * self.hidden_size, bias=True)

        # Output projection — row-parallel (Flash All-Reduce inside)
        self.c_proj = RowParallelLinear(
            self.hidden_size, self.hidden_size,
            flash_allreduce=flash_allreduce, bias=True
        )

        self.attn_dropout  = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # QKV projection — column-parallel, no comm
        qkv = self.c_attn(x)   # [batch, seq, 3 * hidden / world_size]

        hidden_per_partition = self.num_heads_per_partition * self.head_dim
        q, k, v = qkv.split(hidden_per_partition, dim=-1)

        # Reshape to [batch, heads_per_partition, seq, head_dim]
        def reshape(t):
            return t.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale  = self.head_dim ** 0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        mask   = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)   # [batch, heads_per_partition, seq, head_dim]

        # Merge heads -> [batch, seq, hidden / world_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_per_partition)

        # Output projection — row-parallel, Flash All-Reduce inside
        output = self.c_proj(attn_output)
        return self.resid_dropout(output)


class TensorParallelGPT2MLP(nn.Module):
    """
    GPT-2 MLP with Tensor Parallelism.

    c_fc   (up):   Column-parallel — no comm after
    c_proj (down): Row-parallel    — Flash All-Reduce inside
    """

    def __init__(self, config, flash_allreduce=None):
        super().__init__()

        hidden_size       = config.n_embd
        intermediate_size = 4 * hidden_size

        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True)
        self.c_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            flash_allreduce=flash_allreduce, bias=True
        )
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.c_fc(x)                                    # column-parallel, no comm
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)                                  # row-parallel, All-Reduce inside
        return self.dropout(x)


class TensorParallelGPT2Block(nn.Module):
    """
    One GPT-2 transformer block with tensor parallelism.

    Flash All-Reduce fires twice per block:
      1. After attention output projection
      2. After MLP down projection
    """

    def __init__(self, config, flash_allreduce=None):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = TensorParallelGPT2Attention(config, flash_allreduce)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp  = TensorParallelGPT2MLP(config, flash_allreduce)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # attention with residual
        x = x + self.mlp(self.ln_2(x))    # MLP with residual
        return x