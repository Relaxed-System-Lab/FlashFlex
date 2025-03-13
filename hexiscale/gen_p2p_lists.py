
def generate_send_recv_lists(pipeline_groups, mainline, forward_backward=False):

    # initialize empty send and receive lists for each rank
    ranks = set(rank for group in pipeline_groups for rank in group)

    def initialize_lists():
        SendList = {rank: [] for rank in ranks}
        RecvList = {rank: [] for rank in ranks}
        SendBoolean = {rank: [] for rank in ranks}
        RecvBoolean = {rank: [] for rank in ranks}
        return SendList, RecvList, SendBoolean, RecvBoolean
    
    forward_lists = initialize_lists()
    backward_lists = initialize_lists()

    def send_append(idx_from, idx_to, p2p_lists):
        SendList, SendBoolean = p2p_lists[0], p2p_lists[2]
        # Avoid appending duplicates
        if group[idx_to] not in SendList[group[idx_from]]:
            SendList[group[idx_from]].append(group[idx_to])
            SendBoolean[group[idx_from]].append(not is_mainline)

    def recv_append(idx_from, idx_to, p2p_lists ):
        RecvList, RecvBoolean = p2p_lists[1], p2p_lists[3]
        # Avoid appending duplicates
        if group[idx_from] not in RecvList[group[idx_to]]:
            RecvList[group[idx_to]].append(group[idx_from])
            RecvBoolean[group[idx_to]].append(not is_mainline)
    
    # fill up send and receive lists based on pipeline groups
    for group in pipeline_groups:
        is_mainline = set(group) == set(mainline)
        for i in range(len(group) - 1):

            send_append(i, i + 1, forward_lists)
            recv_append(i, i + 1, forward_lists)

    
    if forward_backward:
        for group in pipeline_groups:
            is_mainline = set(group) == set(mainline)
            for i in range(len(group) - 1):
                send_append(i + 1, i, backward_lists)
                recv_append(i + 1, i, backward_lists)
        return forward_lists, backward_lists
    else:
        return forward_lists, None

