import math 
from einops import rearrange
import torch
from functools import partial

def shard_state_dict_tp(state_dict, config, world_size, rank):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    def shard_first_dim(state_dict, key):
        if key in state_dict:
            x = state_dict[key]
            dim = x.shape[0] // world_size
            state_dict[key] = x[rank * dim:(rank + 1) * dim]

    def shard_last_dim(state_dict, key):
        if key in state_dict:
            x = state_dict[key]
            dim = x.shape[-1] // world_size
            state_dict[key] = x[..., rank * dim:(rank + 1) * dim]

    def shard_gatedmlp_fc1_dim(state_dict, key):
        if key in state_dict:
            x = state_dict[key]
            dim = x.shape[0] // world_size // 2
            state_dict[key] = rearrange(
                rearrange(x, "(two o) ... -> two o ...", two=2)[:, rank * dim:(rank + 1) * dim],
                "two o ... -> (two o) ..."
            )

    def shard_qkv_headdim(state_dict, key):
        if key in state_dict:
            n_head = config.n_head
            n_head_kv = getattr(config, 'n_head_kv', n_head)
            assert n_head % world_size == 0 and n_head_kv % world_size == 0
            if n_head_kv == n_head:
                x = rearrange(state_dict[key], '(three d) ... -> three d ...', three=3)
                dim = x.shape[1] // world_size
                state_dict[key] = rearrange(x[:, rank * dim:(rank + 1) * dim],
                                            'three d ... -> (three d) ...')
            else:
                n_head_per_rank = n_head // world_size
                n_head_kv_per_rank = n_head_kv // world_size
                x = rearrange(state_dict[key], '(nheadqkv headdim) ... -> nheadqkv headdim ...',
                              nheadqkv=n_head + 2 * n_head_kv)
                state_dict[key] = rearrange(torch.cat([
                    x[rank * n_head_per_rank:(rank + 1) * n_head_per_rank],
                    x[n_head + rank * n_head_kv_per_rank:n_head + (rank + 1) * n_head_kv_per_rank],
                    x[n_head + n_head_kv + rank * n_head_kv_per_rank:n_head + n_head_kv + (rank + 1) * n_head_kv_per_rank],
                ], dim=0), "nheadqkv headdim ... -> (nheadqkv headdim) ...")

    shard_first_dim(state_dict, 'transformer.embeddings.word_embeddings.weight')
    if 'lm_head.weight' in state_dict:
        shard_first_dim(state_dict, 'lm_head.weight')
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        shard_last_dim(state_dict, 'transformer.embeddings.position_embeddings.weight')
    for i in range(config.num_hidden_layers):
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.weight')
        shard_qkv_headdim(state_dict, f'transformer.layers.{i}.mixer.Wqkv.bias')
        # print(state_dict[f'transformer.layers.{i}.mixer.out_proj.weight'].size())
        # exit(0)
        shard_last_dim(state_dict, f'transformer.layers.{i}.mixer.out_proj.weight')
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mixer.out_proj.bias', None)
        if config.activation_function in ["glu", "swiglu", "geglu"]:
            shard_gatedmlp_fc1_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
            shard_gatedmlp_fc1_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.bias')
        else:
            shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
            shard_first_dim(state_dict, f'transformer.layers.{i}.mlp.fc1.bias')
        shard_last_dim(state_dict, f'transformer.layers.{i}.mlp.fc2.weight')
        if rank != 0:
            state_dict.pop(f'transformer.layers.{i}.mlp.fc2.bias', None)
    return state_dict


def combine_state_dicts_tp(state_dicts, config):
    """Convert the state_dict of a standard GPT model to the state_dict of a GPT model
    with tensor parallel.
    """
    world_size = len(state_dicts)
    keys = state_dicts[0].keys()
    pad_vocab_size_multiple = getattr(config, 'pad_vocab_size_multiple', 1)
    vocab_size = (math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
    assert vocab_size % world_size == 0
    assert config.hidden_size % world_size == 0
    inner_dim = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
    assert inner_dim % world_size == 0

    # Sometimes the word embeddings are sharded on the 0th dim, sometimes on the 1st dim.
    # vocab_size // world_size coordinates are nonzero.
    def combine_word_embeddings(state_dicts, state_dict, key):
        dim = 0 if state_dicts[0][key].shape[0] == vocab_size // world_size else 1
        state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_dim(state_dicts, state_dict, key, dim=-1):
        if key in state_dict:
            state_dict[key] = torch.cat([s[key] for s in state_dicts], dim=dim)

    def combine_qkv_headdim(state_dicts, state_dict, key):
        n_head = config.n_head
        n_head_kv = getattr(config, 'n_head_kv', n_head)
        assert n_head % world_size == 0 and n_head_kv % world_size == 0
        n_head_per_rank = n_head // world_size
        n_head_kv_per_rank = n_head_kv // world_size
        if key in state_dict:
            if n_head_kv == n_head:
                xs = [rearrange(s[key], '(three d) ... -> three d ...', three=3) for s in state_dicts]
                state_dict[key] = rearrange(torch.cat(xs, dim=1), 'three d ... -> (three d) ...')
            else:
                xs = [rearrange(s[key], '(nheadqkv headdim) ... -> nheadqkv headdim ...',
                                nheadqkv=n_head + 2 * n_head_kv) for s in state_dicts]
                state_dict[key] = rearrange(torch.cat([
                    torch.cat([x[:n_head_per_rank] for x in xs], dim=0),
                    torch.cat([x[n_head_per_rank:n_head_per_rank + n_head_kv_per_rank] for x in xs], dim=0),
                    torch.cat([x[-n_head_kv_per_rank:] for x in xs], dim=0),
                ], dim=0), "nheadqkv headdim ... -> (nheadqkv headdim) ...")

    def combine_gated_mlp(state_dicts, state_dict, key):
        if key in state_dict:
            xs = [rearrange(s[key], '(two d) ... -> two d ...', two=2) for s in state_dicts]
            state_dict[key] = rearrange(torch.cat(xs, dim=1), 'two d ... -> (two d) ...')

    state_dict = state_dicts[0].copy()  # don't modify state_dict[0] inplace
    combine_word_embeddings(state_dicts, state_dict, 'transformer.embeddings.word_embeddings.weight')
    if 'lm_head.weight' in state_dict:
        combine_word_embeddings(state_dicts, state_dict, 'lm_head.weight')
    if 'transformer.embeddings.position_embeddings.weight' in state_dict:
        combine_dim(state_dicts, state_dict, 'transformer.embeddings.position_embeddings.weight', -1)
    mlp_combine_fn = (combine_gated_mlp if config.activation_function in ['glu', 'swiglu', 'geglu']
                      else partial(combine_dim, dim=0))
    for i in range(config.num_hidden_layers):
        combine_qkv_headdim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.Wqkv.weight')
        combine_qkv_headdim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.Wqkv.bias')
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mixer.out_proj.weight', -1)
        mlp_combine_fn(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc1.weight')
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc1.bias', 0)
        combine_dim(state_dicts, state_dict, f'transformer.layers.{i}.mlp.fc2.weight', -1)
    return state_dict
