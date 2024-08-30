# hetero_config = [2,2,1]
# process_groups = [CommGroup([0,1]), CommGroup([2,3]), CommGroup([4])]
# process_groups_whole_model = [process_groups[rank//2]] * 6
# pp_ranks_whole_model = [0] + [0,1,2] + [2,2]
# pp_groups = [[0,2,4],[1,3]]
# tp_groups = [dist.new_group([0,1]), dist.new_group([2,3]), dist.new_group([4])]

from flashflex import CommGroup
from third_party import get_args
import torch.distributed as dist
from .gen_p2p_lists import generate_send_recv_lists

def show_groups(groups):
    for group in groups:
        if group is None:
            print('None', end = ' ')
        else:
            group.print()
    print()


def generate_index_mapping(original_lists):
    index_mapping = {}
    for index, group in enumerate(original_lists):
        for rank in group:
            index_mapping[rank] = index
    return index_mapping

def generate_index_mapping_dp(tp_rank_groups):
    """
        tp_rank_groups is 3d, second dim represent tp groups on a stage, third dim represent tp group on specific stage
    """

    index_mapping = {}
    for stage_index, groups in enumerate(tp_rank_groups):
        for index, group in enumerate(groups):
            for rank in group:
                index_mapping[rank] = stage_index
    return index_mapping

def generate_index_mapping_tp(tp_rank_groups):
    """
        Note that if a stage has many tp_groups, then it will need a list to represent which stage is this tp_group on and which tp group it is
    """
    index_mapping = {}
    for index, group in enumerate(tp_rank_groups):
        for sub_id, sub_group in enumerate(group):
            for rank in sub_group:
                index_mapping[rank] = [index, sub_id]
    return index_mapping

def get_group_for_rank(rank, index_mapping):
    return index_mapping.get(rank)

def get_pp_rank_groups(tp_groups):    
    """
        Generate pipeline parallel rank groups
    """

    max_len = max(len(tp_group) for tp_group in tp_groups)
    pp_groups = []
    for i in range(max_len):
        pp_group = []
        for tp_group in tp_groups:
            if i < len(tp_group):
                pp_group.append(tp_group[i])
            else:
                pp_group.append(tp_group[-1])
        pp_groups.append(pp_group)

    return pp_groups

def gen_tp_dp_rank_groups(hetero_config, dp_degree):

    tp_rank_groups = []
    dp_rank_groups_head = []
    tp_degree = [total_deg // dp_deg for total_deg, dp_deg in zip(hetero_config, dp_degree)]
    
    current_rank = 0
    for stage in range(len(hetero_config)):

        # tp_rank = hetero_config[stage]
        dp_deg = dp_degree[stage]
        stage_tp_rank_group = []
        new_dp_group_head = []
        for _ in range(dp_deg):
            new_tp_group = []
            for _ in range(tp_degree[stage]):
                new_tp_group.append(current_rank)
                current_rank += 1
            stage_tp_rank_group.append(new_tp_group)
            new_dp_group_head.append(new_tp_group[0])

        tp_rank_groups.append(stage_tp_rank_group)
        dp_rank_groups_head.append(new_dp_group_head)

    dp_rank_groups_per_rank = []

    for r in range(sum(hetero_config)):
        for i in range(len(dp_rank_groups_head)):
            g = dp_rank_groups_head[i]

            if r in g:
                dp_rank_groups_per_rank.append(g)
                break
        count = 0
        for g in dp_rank_groups_head:
            if r not in g:
                count += 1
        if count == len(dp_rank_groups_head):    
            dp_rank_groups_per_rank.append([r])
    

    return tp_rank_groups, dp_rank_groups_per_rank, dp_rank_groups_head

def gen_tp_pp_rank_whole_model(stage_num, pp_partition, tp_rank_groups):
    pp_ranks_whole_model = [0]
    tp_ranks_whole_model = [0]

    for stage in range(stage_num):
        # If there are extra layers, add one more layer to the current stage
        # layers_in_current_stage = num_layer_per_stage + (1 if stage < extra_layers else 0)
        
        layers_in_current_stage = pp_partition[stage]

        pp_ranks_whole_model.extend([stage] * layers_in_current_stage)
        tp_ranks_whole_model.extend([len(tp_rank_groups[stage])] * layers_in_current_stage)

    pp_ranks_whole_model.extend([stage_num-1] * 2)
    tp_ranks_whole_model.extend([0] * 2)

    return pp_ranks_whole_model, tp_ranks_whole_model

def gen_dp_tp_pp_rank_whole_model(stage_num, pp_partition, total_degree):
    """
        tp and dp's total degree in each stage is known.
    """
    pp_ranks_whole_model = [0]
    dp_tp_ranks_whole_model = [0]

    for stage in range(stage_num):
        # If there are extra layers, add one more layer to the current stage
        # layers_in_current_stage = num_layer_per_stage + (1 if stage < extra_layers else 0)
        
        layers_in_current_stage = pp_partition[stage]

        pp_ranks_whole_model.extend([stage] * layers_in_current_stage)
        dp_tp_ranks_whole_model.extend([total_degree[stage]] * layers_in_current_stage)

    pp_ranks_whole_model.extend([stage_num-1] * 2)
    dp_tp_ranks_whole_model.extend([0] * 2)

    return pp_ranks_whole_model, dp_tp_ranks_whole_model

def gen_tp_comm_groups(pp_layouts):
    # Form process_groups and tp_groups, different from inference version, each stage has many tp_groups
    tp_comm_groups = []

    for pp_layout in pp_layouts:
        for tp_group in pp_layout:
            comm_group = CommGroup(tp_group)
            tp_comm_groups.append(comm_group)

    return tp_comm_groups

def gen_matched_tp_comm_group(tp_rank_group, tp_comm_groups):
    for comm_group in tp_comm_groups:
        if tp_rank_group == comm_group.ranks:
            return comm_group


def gen_existing_stage(layer_layouts, layer):
    layer_existing_stage = []
    for layer_layout in layer_layouts:
        for i in range(len(layer_layout)):
            layout = layer_layout[i]
            if layer in layout:
                layer_existing_stage.append(i)
    return layer_existing_stage

def gen_pp_rank_whole_model(layer_layout, base_rank):
    """
        tp and dp's total degree in each stage is known.
    """
    stage_num = len(layer_layout)
    pp_ranks_whole_model = [0 + base_rank]

    for stage in range(stage_num):
        # If there are extra layers, add one more layer to the current stage
        # layers_in_current_stage = num_layer_per_stage + (1 if stage < extra_layers else 0)
        
        layers_in_current_stage = layer_layout[stage]

        pp_ranks_whole_model.extend([stage + base_rank] * len(layers_in_current_stage))

    pp_ranks_whole_model.extend([stage_num - 1 + base_rank] * 2)

    return pp_ranks_whole_model

def gen_current_tp_pp_groups(tp_rank_groups, pp_rank_groups, layer_layout, rank, forward_backward=True):
    """
        Given a pipeline, define the correct process group for this rank. Layerwise groups are just duplicates of them
        Arguments:
            pp_layout: 
                e.g.                         
                [
                    [0, 1, 2, 3],
                    [4, 5],
                    [6, 7],
                ]  
    """    
    # Form process_groups_whole_model and pp_ranks_whole_model
    pp_ranks_whole_model = gen_pp_rank_whole_model(layer_layout, 0)
    

    pp_index_mapping = generate_index_mapping(pp_rank_groups)
    tp_index_mapping = generate_index_mapping(tp_rank_groups)
    pp_rank_mapping = get_group_for_rank(rank, pp_index_mapping)    # eg: 0     -> on which line
    tp_rank_mapping = get_group_for_rank(rank, tp_index_mapping)    # eg: 0     -> first tp group

    p2p_lists = generate_send_recv_lists(pp_rank_groups, pp_rank_groups[0], forward_backward=forward_backward)

    tp_pp_rank_groups = {
        'tp_rank_mapping': tp_rank_mapping, 
        'pp_rank_mapping': pp_rank_mapping,
        'tp_rank_groups': tp_rank_groups,
        'pp_rank_groups': pp_rank_groups,
        'current_tp_rank_group': tp_rank_groups[tp_rank_mapping],
        'current_pp_rank_groups':  pp_rank_groups[pp_rank_mapping],
        'pp_ranks_whole_model': pp_ranks_whole_model,
        'p2p_lists': p2p_lists,
    }
    
    return tp_pp_rank_groups

def gen_layer_dp_rank_group(layer_related_tp_groups):
    tp_sizes = [len(tp_group) for tp_group in layer_related_tp_groups]
    max_tp_size = max(tp_sizes)
    
    padded_tp_groups = []
    for i in range(len(layer_related_tp_groups)):
        group = layer_related_tp_groups[i]
        max_replicate = max_tp_size // tp_sizes[i] 
        padded_tp_group = []
        for r in group:
            for _ in range(max_replicate):
                padded_tp_group.append(r)
        padded_tp_groups.append(padded_tp_group)

    related_dp_rank_groups = []
    for i in range(max_tp_size):
        related_dp_rank_group = []
        for group in padded_tp_groups:
            related_dp_rank_group.append(group[i])
        related_dp_rank_groups.append(related_dp_rank_group)    

    layer_dp_rank_group = []
    for group in related_dp_rank_groups:
        if dist.get_rank() in group:
            layer_dp_rank_group.append(group)

    min_tp_size_ranks = layer_related_tp_groups[tp_sizes.index(min(tp_sizes))]
    layer_related_dp_rank_groups = [[] for _ in range(len(min_tp_size_ranks))]
    for group in related_dp_rank_groups:
        for i in range(len(min_tp_size_ranks)):
            if min_tp_size_ranks[i] in group:
                layer_related_dp_rank_groups[i].append(group)
    
    return layer_dp_rank_group, layer_related_dp_rank_groups

def gen_layer_dp_rank_groups(pp_layouts, layer_layouts, layer):
    # Form dp_rank_groups
    layer_existing_stage = gen_existing_stage(layer_layouts, layer)

    layer_related_tp_groups = []
    layer_dp_rank_group = []
    for pp_layout, existing_stage in zip(pp_layouts, layer_existing_stage):
        layer_related_tp_groups.append(pp_layout[existing_stage])

    layer_dp_rank_group, layer_related_dp_rank_groups = gen_layer_dp_rank_group(layer_related_tp_groups)

    return layer_dp_rank_group, layer_related_tp_groups, layer_related_dp_rank_groups

def gen_layouts(pp_layouts, hetero_configs, layer_partitions):
    """
        Arguments:
            hetero_config: pipelines, 
                e.g. 
                    [
                        [4, 2, 2], 
                        [4, 4],
                    ]
            layer_partitions:
                e.g.
                    [
                        [16, 8, 8],
                        [16, 16]
                    ]
        Return:
            pp_layouts: List[List[List]]. Multiple pipelines conduct data parallel, 
                        each pipeline has multiple stages,
                        each stage conduct tensor parallel,
            layer_layouts: for each pipeline, describe each layer on which stage, 
                e.g. (not considering embedding, prenorm, cls)
                    [
                        [[layer 0 ~ layer 15], [layer 16 ~ layer 23], [layer 24 ~ layer 32]]
                        [[layer 0 ~ layer 15], [layer 16 ~ layer 32]]
                    ]
                note this is meaning full when layers are not separated proportionally, 
                although hetero_config is [4, 2, 2], layer partition could be [8, 16, 16] or some other partition,
                This phenomenon is plausible when 4 GPUs are RTX 3090 but 2 GPUs are RTX A6000
    """
    # gen pp_layouts if using hetero_configs
    if pp_layouts is None:
        pp_layouts = []
        rank_count = 0
        for hetero_config in hetero_configs:
            pp_layout = []
            for gpus in hetero_config:
                stage_layout = []
                for _ in range(gpus):
                    stage_layout.append(rank_count)
                    rank_count += 1
                pp_layout.append(stage_layout)
            pp_layouts.append(pp_layout)
    
    layer_layouts = []
    for layer_patition in layer_partitions:
        layer_count = 0
        layer_layout = []
        for layers in layer_patition:
            layout = []
            for _ in range(layers):
                layout.append(layer_count)
                layer_count += 1
            layer_layout.append(layout)
        layer_layouts.append(layer_layout)

    return pp_layouts, layer_layouts

def gen_layer_tp_pp_groups(pp_layouts, layer_layouts, rank):
    """
        given pipeline layouts -> create dp, tp, pp rank groups and groups for each layer
        Arguments:
            e.g.:
                The following
                    [
                        [
                            [0, 1, 2, 3],
                            [4, 5],
                            [6, 7],
                        ]
                        [
                            [8, 9, 10, 11],
                            [12, 13, 14, 15],
                        ]
                    ]
                means:
                    pp1: 
                        0   4   6      
                        1   5   7
                        2   
                        3
                    
                    pp2:
                        8       12
                        9       13
                        10      14
                        11      15

                    layer_partition:
                        pp1: 16 8 8
                        pp2: 16 16
                        
                    tp_groups: 
                        pp1: [0, 1, 2, 3] * 16, [4, 5] * 8, [6, 7] * 8
                        pp2: [8, 9, 10, 11] * 16, [12, 13, 14, 15] * 16
                    
                    dp_groups:
                        [0, 8] * 16, [4, 12] * 8, [6, 12] * 8
        Return:
            process_groups: process groups for each layer  
    """
    all_pp_ranks_whole_model = []
    for layer_layout in layer_layouts:
        line_pp_ranks_whole_model = gen_pp_rank_whole_model(layer_layout, 0)
        all_pp_ranks_whole_model.append(line_pp_ranks_whole_model)

    for pp_layout, layer_layout in zip(pp_layouts, layer_layouts):
        # Form tp_rank_groups
        tp_rank_groups = pp_layout
        appear = sum([rank in tp_rank_group for tp_rank_group in tp_rank_groups])
        if not appear:
            continue
        
        # Form pp_rank_groups
        pp_rank_groups = get_pp_rank_groups(tp_rank_groups)

        layer_tp_pp_rank_group = gen_current_tp_pp_groups(tp_rank_groups, pp_rank_groups, layer_layout, rank=rank) 
        
    layer_tp_pp_rank_group['all_pp_ranks_whole_model'] = all_pp_ranks_whole_model
    return layer_tp_pp_rank_group

def gen_layer_dp_comm_groups(layer_dp_rank_groups, existed_comm_groups):
    layer_dp_comm_groups = []
    for layer_dp_rank_group in layer_dp_rank_groups:
        layer_dp_comm_group = []
        for group in layer_dp_rank_group:
            for comm_group in existed_comm_groups:
                if comm_group.ranks == group:
                    layer_dp_comm_group.append(comm_group)
        layer_dp_comm_groups.append(layer_dp_comm_group)
    
    return layer_dp_comm_groups

def gen_layer_dp_overall_comm_groups(layer_dp_overall_rank_groups):
    existed_rank_groups = []
    existed_comm_groups = []

    for layer_dp_rank_group in layer_dp_overall_rank_groups:
        if layer_dp_rank_group not in existed_rank_groups:
            comm_group = CommGroup(layer_dp_rank_group)
            existed_rank_groups.append(layer_dp_rank_group)
            existed_comm_groups.append(comm_group)
        
    layer_dp_comm_groups = []
    for layer_dp_rank_group in layer_dp_overall_rank_groups:
        layer_dp_comm_groups.append(existed_comm_groups[existed_rank_groups.index(layer_dp_rank_group)])
    

    return layer_dp_comm_groups


def gen_layer_dp_related_comm_groups(layers_dp_rank_groups):

    existed_rank_groups = []
    existed_comm_groups = []

    for layer_dp_rank_groups in layers_dp_rank_groups:
        for layer_dp_rank_group in layer_dp_rank_groups:
            for group in layer_dp_rank_group:
                if group not in existed_rank_groups:
                    comm_group = CommGroup(group)
                    existed_rank_groups.append(group)
                    existed_comm_groups.append(comm_group)
    
    layers_dp_comm_groups = []
    for layer_dp_rank_groups in layers_dp_rank_groups:
        layer_dp_comm_groups = []
        for layer_dp_rank_group in layer_dp_rank_groups:
            layer_dp_comm_group = []
            for group in layer_dp_rank_group:
                    layer_dp_comm_group.append(existed_comm_groups[existed_rank_groups.index(group)])

            layer_dp_comm_groups.append(layer_dp_comm_group)
        layers_dp_comm_groups.append(layer_dp_comm_groups)

    dist.barrier()
    return layers_dp_comm_groups, existed_comm_groups

def gen_current_stage_start_end_id(pp_layouts, layer_partitions, rank):
    line_cnt = 0
    # pipeline id and pipeline stage
    location = [0, 0]

    for pp_layout in pp_layouts:
        for tp_group in pp_layout:
            if rank in tp_group:
                location[1] = pp_layout.index(tp_group)
                location[0] = line_cnt
        else:
            line_cnt += 1
    
    import copy
    line_layer_partition = copy.deepcopy(layer_partitions[location[0]])
    # reformat and consider embedding, prenorm, cls
    line_layer_partition[0] += 1
    line_layer_partition[-1] += 2

    start = sum(line_layer_partition[:location[1]])
    end = start + line_layer_partition[location[1]]



    return start, end

def add_utils_layers(groups):
    # add embedding, prenorm, cls layers
    groups.insert(0, groups[0])
    for i in range(2):
        groups.append(groups[-1])

def add_utils_layers_for_all_pipelines(layer_layouts):
    import copy

    padded_layer_layouts = copy.deepcopy(layer_layouts)
    args = get_args()
    
    for layer_layout in padded_layer_layouts:
        layer_layout[0].insert(0, -1)
        layer_layout[-1].append(args.total_layer_num)
        layer_layout[-1].append(args.total_layer_num + 1) 

    return padded_layer_layouts

def gen_hetero_groups(pp_layouts, hetero_configs, layer_partitions):

    """
        Generate all the necessary communication groups, including:
            layer_related_tp_rank_groups: for each layer, record which tp rank groups have appeared
            layer_related_dp_rank_groups: for each layer, record which dp rank groups should involve 
            layer_dp_overall_rank_groups: for each layer, record dp groups by all the participated ranks
            layer_dp_rank_groups: for each stage, record the existed layers and its related dp rank groups
    """
    rank = dist.get_rank()

    pp_layouts, layer_layouts = gen_layouts(pp_layouts, hetero_configs, layer_partitions)
    
    hetero_groups = gen_layer_tp_pp_groups(pp_layouts, layer_layouts, rank)
    stage_idxs = gen_current_stage_start_end_id(pp_layouts, layer_partitions, rank)

    hetero_groups['stage_idxs'] = stage_idxs

    hetero_groups['pp_layouts'] = pp_layouts
    hetero_groups['layer_layouts'] = layer_layouts

    padded_layer_layouts = add_utils_layers_for_all_pipelines(layer_layouts)
    layer_dp_rank_groups = []
    layer_dp_overall_rank_groups = []
    layer_related_dp_rank_groups = []
    layer_related_tp_rank_groups = []
    for i in range(-1, sum(layer_partitions[0]) + 2):
        layer_dp_rank_group, related_tp_groups, related_dp_rank_groups = gen_layer_dp_rank_groups(pp_layouts, padded_layer_layouts, i)

        layer_related_tp_rank_groups.append(related_tp_groups)
        layer_dp_rank_groups.append(layer_dp_rank_group)
        layer_related_dp_rank_groups.append(related_dp_rank_groups)

        layer_dp_overall_group = []
        for group in related_tp_groups:
            layer_dp_overall_group.extend(group)

        layer_dp_overall_rank_groups.append(layer_dp_overall_group)
    
    # Form tp comm groups
    tp_comm_groups = gen_tp_comm_groups(pp_layouts)
    hetero_groups['current_tp_comm_group'] = gen_matched_tp_comm_group(hetero_groups['current_tp_rank_group'], tp_comm_groups)
    hetero_groups['current_tp_group'] = hetero_groups['current_tp_comm_group'].group

    # Form dp comm groups
    layer_related_dp_comm_groups, existed_comm_groups = gen_layer_dp_related_comm_groups(layer_related_dp_rank_groups)
    layer_dp_overall_comm_groups = gen_layer_dp_overall_comm_groups(layer_dp_overall_rank_groups)
    layer_dp_comm_groups = gen_layer_dp_comm_groups(layer_dp_rank_groups, existed_comm_groups)

    # Store all the information in a dict
    hetero_groups['layer_dp_comm_groups'] = layer_dp_comm_groups
    hetero_groups['layer_dp_rank_groups'] = layer_dp_rank_groups
    hetero_groups['layer_related_tp_rank_groups'] = layer_related_tp_rank_groups
    hetero_groups['layer_related_dp_rank_groups'] = layer_related_dp_rank_groups
    hetero_groups['layer_related_dp_comm_groups'] = layer_related_dp_comm_groups
    hetero_groups['layer_dp_overall_rank_groups'] = layer_dp_overall_rank_groups
    hetero_groups['layer_dp_overall_comm_groups'] = layer_dp_overall_comm_groups

    return hetero_groups