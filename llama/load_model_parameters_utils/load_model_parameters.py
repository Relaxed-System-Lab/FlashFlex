import torch
from .shard_combine_utils import shard_state_dict_tp

def load_weights(
    layer, state_dict,
    layer_idx, config
):
    """Loads all layer weights from the state dictionary.
    
    Args:
        layer: The layer object to load weights into.
        state_dict (dict): The nested dictionary containing all state data.
        layer_idx (int): The index of the layer.
        config: The configuration object.
    Returns:
        layer: The layer object with loaded weights.
    """
    
    # Construct the base string for key access
    base_str = f'transformer.layers.{layer_idx}'
    
    assert layer.mixer.Wqkv.weight.is_contiguous()
    assert layer.mixer.out_proj.weight.is_contiguous()
    assert layer.mlp.fc1.weight.is_contiguous()
    assert layer.mlp.fc2.weight.is_contiguous()
    assert layer.norm1.weight.is_contiguous()
    assert layer.norm2.weight.is_contiguous()
    
    # Load mixer weights
    layer.mixer.Wqkv.weight.data.copy_(state_dict[f'{base_str}.mixer.Wqkv.weight'])
    layer.mixer.out_proj.weight.data.copy_(state_dict[f'{base_str}.mixer.out_proj.weight'])
    
    # Load mlp weights
    layer.mlp.fc1.weight.data.copy_(state_dict[f'{base_str}.mlp.fc1.weight'])
    
    if config.activation_function in ["glu", "swiglu", "geglu"]:
        layer.mlp.fc2.weight.data.copy_(state_dict[f'{base_str}.mlp.fc2.weight'])
    
    # Load normalization layer weights and biases
    layer.norm1.weight.data.copy_(state_dict[f'{base_str}.norm1.weight'])
    layer.norm2.weight.data.copy_(state_dict[f'{base_str}.norm2.weight'])
    
    # Set dropout probabilities to 0.0
    layer.dropout1.p = 0.0
    layer.dropout2.p = 0.0
    layer.mixer.inner_attn.drop.p = 0.0
    layer.mixer.inner_cross_attn.drop.p = 0.0
    
    return layer


def load_parameters_given_tp_group(model, config, state_dicts_path, layer_tp_sizes, layer_tp_rank_group, rank):
    # layer_tp_sizes = tp_ranks_whole_model[1:-2]
    # tp_rank = [i for sub_array in layer_tp_rank_group for i in range(len(sub_array))]

    for module in model.model_cur_stage:
        if module.label()[0] == 0:
            embed_data = torch.load(f'{state_dicts_path}/separate_state_dicts/embeddings.pt')
            module.embeddings.word_embeddings.weight.data.copy_(embed_data)
        elif module.label()[0] == 1:
            layer = module.layers[0]

            layer_state_dict = torch.load(f'{state_dicts_path}/separate_state_dicts/layer_{module.label()[1]}.pt')
            # layer_state_dict = torch.load(f'{state_dicts_path}/separate_state_dicts/layer_{0}.pt')
            buffer_data = torch.load(f'{state_dicts_path}/inv_freq.pt')
            layer.mixer.rotary_emb.inv_freq.copy_(buffer_data)
            # print(f"{layer_tp_rank_group[module.label()[1]].index(rank)}-{torch.distributed.get_rank()}")
            
            # new_dict = {}
            # key_length = len('transformer.layers.0.')
            # for key, val in layer_state_dict.items():
            #     new_dict[f'transformer.layers.{module.label()[1]}.{key[key_length:]}'] = val
            # layer_state_dict = new_dict

            # for key, val in layer_state_dict.items():
            #     layer_state_dict[key] = torch.arange(val.numel(), device=val.device, dtype=val.dtype).reshape(val.shape) / 1e3
            if torch.distributed.get_rank() == 0:
                print(layer_state_dict.keys())
                print(len(model.model_cur_stage))
                
            
            layer_state_dict = shard_state_dict_tp(layer_state_dict, config, layer_tp_sizes[module.label()[1]], layer_tp_rank_group[module.label()[1]].index(rank))
            load_weights(layer, layer_state_dict, module.label()[1], config)
        elif module.label()[0] == 2:
            ln_f_data = torch.load(f'{state_dicts_path}/separate_state_dicts/ln_f.pt')
            module.ln_f.weight.data.copy_(ln_f_data)
        else:
            lm_head_data = torch.load(f'{state_dicts_path}/separate_state_dicts/lm_head.pt')
            module.lm_head.weight.data.copy_(lm_head_data)


def load_model_parameters(model, config, state_dicts_path, hetero_groups, rank, args):
    
    """
    Loads and applies specific model parameters from external files to different components of a given model.

    Parameters:
    - model (PyTorch Model): The neural network model whose parameters are to be updated.
    - config (dict): A configuration dictionary containing model-specific settings.
    - tp_ranks_whole_model (list): A list of tensor parallel (TP) ranks for the entire model. This is used to determine the layer sizes for sharding.
    - tp_group_list (list of lists): A nested list where each sublist represents a tensor parallel group, used to compute the TP rank.
    - rank (int): The rank of the current process in a distributed training setup, used to index into the TP rank list.

    This function iterates through each module of the model's current stage. Depending on the type of module (identified by its label), it loads different parameters:
    - For embedding layers, it loads embedding weights.
    - For transformer layers, it loads layer-specific weights and applies rotary position embeddings.
    - For the final layer normalization and language model head, it loads their respective weights.

    Note:
    - The function assumes the existence of specific .pt files in 'separate_state_dicts' directory for each type of parameter.
    - Functions 'shard_state_dict_tp' and 'load_weights' need to be defined and available in the scope.
    - The actual structure and labeling of the model components must align with the logic used in the conditional statements.
    """

    if torch.distributed.get_rank() == 0:
        print("...loading parameters")

    layer_related_tp_rank_groups = hetero_groups['layer_related_tp_rank_groups']
    pipeline_nums = len(args.hetero_configs)
    pipeline_layer_tp_rank_groups = [[] for _ in range(pipeline_nums)]
    pipeline_layer_tp_sizes = [[] for _ in range(pipeline_nums)]
    for i in range(len(layer_related_tp_rank_groups)):
        for pp_id in range(pipeline_nums):
            tp_size = len(layer_related_tp_rank_groups[i][pp_id])
            pipeline_layer_tp_rank_groups[pp_id].append(layer_related_tp_rank_groups[i][pp_id])
            pipeline_layer_tp_sizes[pp_id].append(tp_size)

    # if torch.distributed.get_rank() == 0:

    #     print(pipeline_layer_tp_rank_groups)
    #     print(pipeline_layer_tp_sizes)
    # exit(0)
    for i in range(pipeline_nums):
        for tp_rank_group in pipeline_layer_tp_rank_groups[i]:
            if rank in tp_rank_group:
                cur_pp_id = i
                break

    load_parameters_given_tp_group(model, config, state_dicts_path, pipeline_layer_tp_sizes[cur_pp_id][1:-2], pipeline_layer_tp_rank_groups[cur_pp_id][1:-2], rank)

