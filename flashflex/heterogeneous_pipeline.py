import torch
import torch.nn as nn
import numpy as np
from flashflex.utils import unwrap_model, chunk_batch
from typing import Union, Optional, Tuple, List
import operator
import copy
from flashflex import wrap_modules_data_parallel, wrap_modules_checkpoint
import torch.distributed as dist
from typing import Any
from flashflex.utils import *
from flashflex.fsdp_hooks_utils import *
from flashflex.fsdp_comm_utils import get_fsdp_comm_config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from flashflex.fsdp_comm_utils import initialize_comm_utils, hetero_comm_ddp, reduce_scatter_hook_with_ga

Shape = Union[List[int], torch.Size]

_RECV_BUFFER = None


class PipelineParallel(nn.Module):
    def __init__(
        self, 
        model, 
        model_ranks : List, 
        all_lines_model_ranks: List,
        layer_output_tensor_shapes,
        layer_output_tensor_dtypes=None,
        layer_dp_sizes=None,
        chunks = 1, 
        *,
        optim=None,
        p2p_lists=None,
        process_group=None,
        broadcast_group=None,
        dp_process_group=None,
        broadcast_group_list=None,
        info = False,
        show_process = True):

        super().__init__()

        # initialize rank variables
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.device_count = torch.cuda.device_count()
        self.local_rank = self.global_rank % self.device_count
        
        self.pp_global_ranks = [i for i in range(self.world_size)] if process_group is None else sorted(list(set(list(process_group))))
        self.pp_on = (len(self.pp_global_ranks) > 1)
        assert(self.global_rank in self.pp_global_ranks)
        self.group = dist.new_group(process_group)
        self.group_size = dist.get_world_size(self.group)
        self.group_rank = dist.get_rank(self.group)

        # initialize communication variables
        self.fwd_send_list, self.fwd_recv_list, self.fwd_send_empty_list, self.fwd_recv_empty_list = p2p_lists[0]
        self.bwd_send_list, self.bwd_recv_list, self.bwd_send_empty_list, self.bwd_recv_empty_list = p2p_lists[1]

        self.dp_process_group = dp_process_group
        self.broadcast_group_list = broadcast_group_list

        self.broadcast_group = broadcast_group

        # initialize model info
        self.microbatches = []
        self.chunks = int(chunks)
        self.total_model_len = len(model)
        assert(self.chunks >= 1)
        assert(len(model) == len(model_ranks))
        assert(len(model) == len(layer_output_tensor_shapes))
        self.stage_start_idx, cnt = model_ranks.index(self.group_rank), model_ranks.count(self.group_rank)
        self.stage_end_idx = self.stage_start_idx + cnt
        self.model_cur_stage = model[self.stage_start_idx:self.stage_end_idx] # cuda(self.local_rank)
        self.current_line_model_ranks = model_ranks[self.stage_start_idx : self.stage_end_idx]
        self.all_lines_model_ranks = [stage_lines_model_ranks[self.stage_start_idx : self.stage_end_idx] for stage_lines_model_ranks in all_lines_model_ranks]
        # print(global_bsz_size, self.local_rank)
        # exit(0)
        self.optim = optim

        # cache input and output tensors for backward
        self.input_tensors, self.output_tensors = [], []
        self.losses_reduced = []

        # check tensor dtype and initialize shape variables
        layer_output_tensor_dtypes = self.get_default_tensor_dtype(layer_output_tensor_shapes) if layer_output_tensor_dtypes is None else layer_output_tensor_dtypes
        self.check_tensor_dtype(layer_output_tensor_shapes, layer_output_tensor_dtypes)
        self.stage_input_tensor_shape = [None] if self.is_pipeline_first_stage() else layer_output_tensor_shapes[self.stage_start_idx - 1]
        self.stage_output_tensor_shape = [None] if self.is_pipeline_last_stage() else layer_output_tensor_shapes[self.stage_end_idx - 1]
        self.stage_input_tensor_dtype = [None] if self.is_pipeline_first_stage() else layer_output_tensor_dtypes[self.stage_start_idx - 1]
        self.stage_output_tensor_dtype = [None] if self.is_pipeline_last_stage() else layer_output_tensor_dtypes[self.stage_end_idx - 1]
        self.dp_size_prev_stage = None if self.is_pipeline_first_stage() else layer_dp_sizes[self.stage_start_idx - 1]
        self.dp_size_cur_stage = None if self.is_pipeline_last_stage() else layer_dp_sizes[self.stage_end_idx - 1]

        # initialize dp sizes variables
        if layer_dp_sizes is None:
            layer_dp_sizes = [1] * len(model)
            self.dp_size_input = layer_dp_sizes[0]
        else:
            assert(len(model) == len(layer_dp_sizes))
            self.dp_size_input = layer_dp_sizes[0]

        # initialize util variables
        self.info = info
        self.show_process = show_process
        self.chunk_warning = True
        self.checkpoint_flags_stage = None


    def _1F1B_warmup(self, num_warmup_microbatches, num_microbatches, microbatches, forward_step_func, loss_func, forward_only=False, args=None):
        input_tensors = []
        output_tensors = []
        fwd_num, bwd_num = 0, 0
        if self.info:
            print('rank %d'%self.global_rank, 'start warmup')
            print('rank %d'%self.global_rank, 'num_warmup_microbatches', num_warmup_microbatches)
        # Run warmup forward passes.
        for i in range(num_warmup_microbatches):
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)

            cur_microbatch = [microbatches[0][i], microbatches[1][i]]
            
            pre_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage)

            output_tensor = self.forward_step(
                forward_step_func,
                cur_microbatch,
                self.model_cur_stage,
                input_tensor,
                loss_func, 
                forward_only=forward_only,
                args=args,
            )
            
            post_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
            
            fwd_num += 1
            self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd, dtypes=send_tensor_dtypes)

            if not forward_only:
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

        if self.info:
            print('rank %d'%self.global_rank, 'finish warmup')
        
        return fwd_num, bwd_num, input_tensors, output_tensors


    def _1F1B_cooldown(self, num_warmup_microbatches, num_microbatches, input_tensors, output_tensors, bwd_num, forward_only=False):
        if self.info:
            print('rank %d'%self.global_rank, 'start cooldown')
            print('rank %d'%self.global_rank, 'num_warmup_microbatches', num_warmup_microbatches)
        # Run cooldown backward passes.
        if not forward_only:
            for i in range(num_warmup_microbatches):
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                recv_tensor_shapes_bwd = self.stage_input_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
                send_tensor_shapes_bwd = self.stage_output_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
                recv_tensor_dtypes = self.stage_input_tensor_dtype
                send_tensor_dtypes = self.stage_output_tensor_dtype
                
                output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes_bwd, dtypes=send_tensor_dtypes)
                
                pre_pipeline_backward(num_microbatches, bwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
                
                input_tensor_grad = self.backward_step(
                    input_tensor,
                    output_tensor,
                    output_tensor_grad,
                )
                bwd_num += 1

                self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd, dtypes=recv_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'finish cooldown')

    def _1F1B_forward_backward(
        self,
        forward_step_func,
        batch, 
        loss_func, 
        forward_only=False,
        args=None,
        ):
        """Run non-interleaved 1F1B schedule, with communication between pipeline
        stages.

        Returns dictionary with losses if the last stage, empty dict otherwise."""
        
        fsdp_comm_config = get_fsdp_comm_config()

        fsdp_comm_config.current_iter = args.current_iter

        # Chunk input batch into microbatches
        microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        self.real_chunks = len(microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0][0].shape[0] for m in microbatches])
                print()
                self.chunk_warning = False

        # Compute number of warmup microbatches.
        num_microbatches = self.real_chunks
        num_warmup_microbatches = self.group_size - self.group_rank - 1
        # print(f"{num_warmup_microbatches}-{dist.get_rank()}")
        # exit(0)
        num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
        num_microbatches_remaining = num_microbatches - num_warmup_microbatches

        # Update stage_input_tensor_shape with correct microbatch_size
        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            self.stage_input_tensor_shape, self.stage_input_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_prev_stage, self.stage_input_tensor_shape)

        # Update stage_output_tensor_shape with correct microbatch_size
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [None]
        else:
            self.stage_output_tensor_shape, self.stage_output_tensor_shape_last = \
                self.update_tensor_shape(microbatches, self.dp_size_input, self.dp_size_cur_stage, self.stage_output_tensor_shape)

        # print('rank %d'%self.global_rank, self.stage_input_tensor_shape, self.stage_input_tensor_shape_last, self.stage_output_tensor_shape, self.stage_output_tensor_shape_last, self.stage_input_tensor_dtype, self.stage_output_tensor_dtype)

        fwd_num, bwd_num, input_tensors, output_tensors = self._1F1B_warmup(num_warmup_microbatches, num_microbatches, microbatches, forward_step_func, loss_func, forward_only, args)

        # Before running 1F1B, need to receive first forward tensor.
        # If all microbatches are run in warmup / cooldown phase, then no need to
        # receive this tensor here.
        if num_microbatches_remaining > 0:
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'start 1f1b')
            print('rank %d'%self.global_rank, 'num_microbatches_remaining', num_microbatches_remaining)
        
        # Run 1F1B in steady state.
        for i in range(num_microbatches_remaining):
            recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_shapes_bwd = self.stage_input_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes_bwd = self.stage_output_tensor_shape_last if bwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            last_iteration = (i == (num_microbatches_remaining - 1))
            cur_microbatch = [microbatches[0][i + num_warmup_microbatches], microbatches[1][i + num_warmup_microbatches]]

            pre_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage)

            output_tensor = self.forward_step(
                forward_step_func,
                cur_microbatch,
                self.model_cur_stage,
                input_tensor,
                loss_func, 
                forward_only=forward_only,
                args=args,
            )
            
            post_pipeline_forward(num_microbatches, fwd_num, self.model_cur_stage, self.checkpoint_flags_stage)
            
            fwd_num += 1
            if forward_only:
                self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd, dtypes=send_tensor_dtypes)

                if not last_iteration:
                    input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes)
            else:
                output_tensor_grad = self.send_forward_recv_backward_multi(output_tensor, tensor_shapes=send_tensor_shapes_bwd, dtypes=send_tensor_dtypes, tensor_shapes_send=send_tensor_shapes_fwd)
                recv_tensor_shapes_fwd = self.stage_input_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_input_tensor_shape
                send_tensor_shapes_fwd = self.stage_output_tensor_shape_last if fwd_num == num_microbatches - 1 else self.stage_output_tensor_shape
                # # if send and recv is executed sequentially, dead lock will be caused!
                # self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes_fwd)
                # output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes_bwd)

                # Add input_tensor and output_tensor to end of list, then pop from the
                # start of the list for backward pass.
                input_tensors.append(input_tensor)
                output_tensors.append(output_tensor)

                # Pop input_tensor and output_tensor from the start of the list for the backward pass.
                input_tensor = input_tensors.pop(0)
                output_tensor = output_tensors.pop(0)

                pre_pipeline_backward(num_microbatches, bwd_num, self.model_cur_stage, self.checkpoint_flags_stage)

                input_tensor_grad = self.backward_step(
                    input_tensor,
                    output_tensor,
                    output_tensor_grad,
                )
                bwd_num += 1

                if last_iteration:
                    input_tensor = None
                    self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd, dtypes=recv_tensor_dtypes)
                else:
                    input_tensor = self.send_backward_recv_forward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_fwd, dtypes=recv_tensor_dtypes, tensor_shapes_send=recv_tensor_shapes_bwd)
                    # # if send and recv is executed sequentially, dead lock will be caused!
                    # self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes_bwd)
                    # input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes_fwd)

        if self.info:
            print('rank %d'%self.global_rank, 'finish 1f1b')

        self._1F1B_cooldown(num_warmup_microbatches, num_microbatches, input_tensors, output_tensors, bwd_num, forward_only)
        
        fsdp_comm_config.layer_count = fsdp_comm_config.layer_end - 1
            
        return 0


    def gpipe_forward_backward(
        self,
        forward_step_func, 
        batch,
        stage_tensor_shape,
        loss_func,
        forward_only=False,
        args=None,
        ):
        """Run gpipe schedule, with communication between pipeline stages.

        Returns dictionary with losses if the last stage, empty dict otherwise."""

        fsdp_comm_config = get_fsdp_comm_config()

        fsdp_comm_config.current_iter = args.current_iter

        output_tensor = self.gpipe_forward(forward_step_func, batch, stage_tensor_shape, loss_func, forward_only, args=args,)
        
        # exit(0)   
            # print(f"Before backward, memory allocated on rank 0: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024}")
        if not forward_only:
            self.gpipe_backward(forward_step_func, loss_func, args=args)

        
        fsdp_comm_config.layer_count = fsdp_comm_config.layer_end - 1
            
        return output_tensor
    
    def gpipe_forward(
        self,
        forward_step_func, 
        batch,
        stage_tensor_shape,
        loss_func,
        forward_only=False,
        args=None,
        ):
        
        # Chunk input batch into microbatches
        self.microbatches = [chunk_batch(batch[0], self.chunks), chunk_batch(batch[1], self.chunks)]
        # print(self.microbatches)
        # print(self.microbatches[0][0][0])
        # exit(0)

        self.real_chunks = len(self.microbatches[0])
        if self.chunks != self.real_chunks and self.chunk_warning:
            if self.global_rank == 0:
                print('\nWarning from PipelineParallel Module: Real chunks is %d !'%self.real_chunks, 'Microbatch sizes is', [m[0].shape[0] for m in self.microbatches[0]], '\n')
            self.chunk_warning = False
        self.num_microbatches = self.real_chunks
        
        # Compute tensor shapes for all microbatches, note that the last microbatch may have different microbatch_size, thus different shape!
        batch_size = batch[0][0].shape[0] * self.dp_size_input

        """
        self.stage_input_tensor_shape       : input tensor shape for previous microbatches
        self.stage_input_tensor_shape_last  : input tnesor shape for the last microbatches
        self.stage_output_tensor_shape      : output tensor shape for previous microbatches
        self.stage_output_tensor_shape_last : output tensor shape for the last microbatches

        """
        # Update stage_input_tensor_shape with correct microbatch_size
        if self.is_pipeline_first_stage():
            self.stage_input_tensor_shape = self.stage_input_tensor_shape_last = [None]
        else:
            self.stage_input_tensor_shape, self.stage_input_tensor_shape_last = \
                self.update_tensor_shape(self.microbatches, self.dp_size_input, self.dp_size_prev_stage, self.stage_input_tensor_shape)

        # Update stage_output_tensor_shape with correct microbatch_size
        if self.is_pipeline_last_stage():
            self.stage_output_tensor_shape = self.stage_output_tensor_shape_last = [None]
        else:
            self.stage_output_tensor_shape, self.stage_output_tensor_shape_last = \
                self.update_tensor_shape(self.microbatches, self.dp_size_input, self.dp_size_cur_stage, self.stage_output_tensor_shape)
        
        if self.info:
            print('rank %d'%self.global_rank, 'start forward')

        # Run forward passes.
        for i in range(self.num_microbatches):
            recv_tensor_shapes = self.stage_input_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes = self.stage_output_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_output_tensor_shape
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype

            input_tensor = self.recv_forward_multi(tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes)

            cur_microbatch = [self.microbatches[0][i], self.microbatches[1][i]]

            pre_pipeline_forward(self.num_microbatches, i, self.model_cur_stage)
            
            # this function uses either input_tensor or cur_microbatch
            input_tensor, output_tensor = self.forward_step(
                forward_step_func,
                cur_microbatch,
                self.model_cur_stage,
                input_tensor,
                loss_func, 
                forward_only=forward_only,
                args=args,
            )

            if not forward_only:
                self.input_tensors.append(list(input_tensor))
                if not args.recompute_stage_output:
                    self.output_tensors.append(list(output_tensor))

            post_pipeline_forward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)
            self.send_forward_multi(output_tensor, tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'finish forward')

        if self.is_pipeline_last_stage():
            return output_tensor
        else:
            return None

    def gpipe_backward(self, forward_step_func, loss_func, args):
        if self.info:
            print('rank %d'%self.global_rank, 'start backward')

        
        # Run backward passes.
        for i in range(self.num_microbatches):

            # for tensors in self.input_tensors:
            #     if tensors[0] is None:
            #         continue
            #     print(tensors[0].size(), self.local_rank)

            # print(len(self.input_tensors))
            if not args.recompute_stage_output:
                input_tensor = self.input_tensors.pop(0)
                output_tensor = self.output_tensors.pop(0)
            else:
                input_tensor = self.input_tensors.pop(0)
                # output_tensor = self.output_tensors.pop(0)
                pre_pipeline_forward(self.num_microbatches, i, self.model_cur_stage)
                cur_microbatch = [self.microbatches[0][i], self.microbatches[1][i]]
                _, output_tensor = self.forward_step(
                    forward_step_func,
                    cur_microbatch,
                    self.model_cur_stage,
                    input_tensor,
                    loss_func, 
                    forward_only=True,
                    args=args,
                )
                post_pipeline_forward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)

            # shapes
            recv_tensor_shapes = self.stage_input_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_input_tensor_shape
            send_tensor_shapes = self.stage_output_tensor_shape_last if i == self.num_microbatches - 1 else self.stage_output_tensor_shape
            
            # dtypes
            recv_tensor_dtypes = self.stage_input_tensor_dtype
            send_tensor_dtypes = self.stage_output_tensor_dtype
            output_tensor_grad = self.recv_backward_multi(tensor_shapes=send_tensor_shapes, dtypes=send_tensor_dtypes)
            
            
            pre_pipeline_backward(self.num_microbatches, i, self.model_cur_stage, self.checkpoint_flags_stage)
            
            input_tensor_grad = self.backward_step(
                input_tensor,
                output_tensor,
                output_tensor_grad,
            )

            self.send_backward_multi(input_tensor_grad, tensor_shapes=recv_tensor_shapes, dtypes=recv_tensor_dtypes)

        if self.info:
            print('rank %d'%self.global_rank, 'finish backward')

        return output_tensor

    # forward & backward step
    # ---------------------------------------
    def forward_step(self, forward_step_func, batch, model, input_tensor, loss_func, forward_only=True, args=None):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Returns output tensor."""

        unwrapped_model = unwrap_model(model)
        if not isinstance(input_tensor, (list, tuple)):
            input_tensor = [input_tensor]

        for x in input_tensor:
            if x is not None and x.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                x.requires_grad = True

        # input tensor is None on the first stage
        if input_tensor[0] is None:

            output_tensor = forward_step_func(batch[0], model)

        else:
            # Sync input hidden states within each tensor parallel group
            if len(self.broadcast_group_list) > 1:
                self.tp_broadcast(
                    input_tensor,
                    msg=["Forward", "Input Tensor"]
                )

            output_tensor = forward_step_func(input_tensor, model)

        if not isinstance(output_tensor, (list, tuple)):
            output_tensor = [output_tensor]

        if self.is_pipeline_last_stage():
            
            shift_logits = output_tensor[0][..., :-1, :].contiguous()
            shift_labels = batch[1][0][..., 1:].contiguous()
            
            loss = loss_func(shift_logits.view(-1, args.vocab_size), shift_labels.view(-1).long())
            
            output_tensor = [loss / self.real_chunks / args.accum_iter]


        # print(f"f{self.output_tensors}-{dist.get_rank()}")

        return input_tensor, list(output_tensor)

    def backward_step(self, input_tensor, output_tensor, output_tensor_grad):
        """Backward step through passed-in output tensor.

        If last stage, output_tensor_grad is None, otherwise gradient of loss
        with respect to stage's output tensor.

        Returns gradient of loss with respect to input tensor (None if first
        stage)."""

        # TP broadcasting
        if len(self.broadcast_group_list) > 1:
            self.tp_broadcast(
                output_tensor_grad,
                msg=["Backward", "Output Tensor Grad"]
            )
         
        # Retain the grad on the input_tensor.        
        unwrap_input_tensor_grad = not isinstance(input_tensor, list)
        if unwrap_input_tensor_grad:
            input_tensor = [input_tensor]

        input_tensor = [None if t is None or not t.requires_grad else t for t in input_tensor]
        for x in input_tensor:
            if x is not None:
                x.retain_grad()

        if not isinstance(output_tensor, list):
            output_tensor = [output_tensor]
        if not isinstance(output_tensor_grad, list):
            output_tensor_grad = [output_tensor_grad]

        # Backward pass.
        output_tensor_, output_tensor_grad_ = [], []
        for t, g in zip(output_tensor, output_tensor_grad):
            if t is not None and t.requires_grad:

                output_tensor_.append(t)
                output_tensor_grad_.append(g)
        

        torch.autograd.backward(output_tensor_, grad_tensors=output_tensor_grad_)

        # Collect the grad of the input_tensor.
        input_tensor_grad = [None]
        if input_tensor is not None:
            input_tensor_grad = []
            for x in input_tensor:
                input_tensor_grad.append(None if x is None else x.grad)

        return input_tensor_grad[0] if unwrap_input_tensor_grad else input_tensor_grad

    # collective operations
    # --------------------------------------- 
    def tp_broadcast(self, tensor, msg):
        # self.show_process = True
        for i in range(len(tensor)):
            if self.show_process:
                if tensor[i] is not None:
                    print(f'During {msg[0]}, Rank: {self.broadcast_group_list[0]} Start Broadcasting {msg[1]} {i} with Shape: {tensor[i].shape}, to Rank Group: {self.broadcast_group_list}')
                else:
                    print(f'During {msg[0]}, Rank: {self.broadcast_group_list[0]} Start Broadcasting {msg[1]} {i} with Shape: {None}, to Rank Group: {self.broadcast_group_list}')

            dist.broadcast(tensor=tensor[i] if tensor[i] is not None else torch.zeros([]).cuda(), src=self.broadcast_group_list[0], group=self.broadcast_group)

    # pipeline shape and wrap utils
    # ---------------------------------------
    def check_tensor_dtype(self, layer_output_tensor_shapes, layer_output_tensor_dtypes):
        assert(len(layer_output_tensor_shapes) == len(layer_output_tensor_dtypes))
        for i in range(len(layer_output_tensor_shapes)):
            if layer_output_tensor_shapes[i] is not None:
                assert(len(layer_output_tensor_shapes[i]) == len(layer_output_tensor_dtypes[i]))

    def get_default_tensor_dtype(self, layer_output_tensor_shapes):
        layer_output_tensor_dtypes = []
        for tensor_shape in layer_output_tensor_shapes:
            if tensor_shape is None:
                layer_output_tensor_dtypes.append(None)
            else:
                layer_output_tensor_dtypes.append([torch.float] * len(tensor_shape))
        return layer_output_tensor_dtypes

    def wrap_pipeline_modules_data_parallel(self, dp_types, dp_groups, module_types, mixed_precision=torch.bfloat16, wrap_block_name=None):
        assert(self.total_model_len == len(dp_types))
        assert(self.total_model_len == len(dp_groups))
        assert(self.total_model_len == len(module_types))

        dp_types_cur_stage = dp_types[self.stage_start_idx:self.stage_end_idx]
        module_types_cur_stage = module_types[self.stage_start_idx:self.stage_end_idx]
        dp_groups_cur_stage = dp_groups[self.stage_start_idx:self.stage_end_idx]
        pp_devices_cur_stage = [self.local_rank]*(self.stage_end_idx-self.stage_start_idx)
       
        
        self.model_cur_stage = wrap_modules_data_parallel(
            module_list=self.model_cur_stage,
            dp_types=dp_types_cur_stage,
            dp_groups=dp_groups_cur_stage,
            module_types=module_types_cur_stage,
            pp_devices=pp_devices_cur_stage,
            mixed_precision=mixed_precision,
            pp_on=True,
            wrap_block_name=wrap_block_name,
        )


    def wrap_modules_data_parallel_comm_hook(self, hetero_groups, args, dp_types_whole_model):
        # TODO: remove default_dp_type, use zero2 as much as possible
        if len(args.hetero_configs) == 1:
            return

        fsdp_comm_config = initialize_comm_utils(hetero_groups, self)
        
        dp_type_cur_stage = dp_types_whole_model[self.stage_start_idx: self.stage_end_idx]
        
        layer_related_tp_rank_groups_cur_stage = hetero_groups['layer_related_tp_rank_groups'][self.stage_start_idx: self.stage_end_idx]
        
        from torch.distributed.fsdp.api import ShardingStrategy

        fsdp_type_dict = {0: 'ddp', 1: 'zero2', 2: 'zero3'}
        sharding_strategies = {'ddp': ShardingStrategy.NO_SHARD,
                           'zero2': ShardingStrategy.SHARD_GRAD_OP,
                           'zero3': ShardingStrategy.FULL_SHARD}


        assert len(dp_type_cur_stage) == len(self.model_cur_stage)
        
        for dp_type, module, tp_rank_groups in zip(dp_type_cur_stage, self.model_cur_stage, layer_related_tp_rank_groups_cur_stage):

            comm_hook = None
            tp_size = len(tp_rank_groups[0])
            for groups in tp_rank_groups:
                if len(groups) != tp_size:
                    comm_hook = hetero_comm_ddp
                    break
            # sharding_strategy = sharding_strategies[fsdp_type_dict[dp_type]]
            # print(sharding_strategy, comm_hook, torch.distributed.get_rank())
            
            if comm_hook is None:
                comm_hook = reduce_scatter_hook_with_ga if dp_type == 1 else hetero_comm_ddp
            
                
            if isinstance(module, FSDP):
                module.register_comm_hook(fsdp_comm_config, comm_hook)
            else:
                module.layers[0].register_comm_hook(fsdp_comm_config, comm_hook)

    def wrap_pipeline_modules_checkpoint(self, checkpoint_flags, wrap_block_name=None):

        self.checkpoint_flags_stage = checkpoint_flags[self.stage_start_idx:self.stage_end_idx]
        if np.sum(checkpoint_flags) > 0:
            assert(self.total_model_len == len(checkpoint_flags))
            self.model_cur_stage = wrap_modules_checkpoint(self.model_cur_stage, self.checkpoint_flags_stage, wrap_block_name=wrap_block_name)
            if wrap_block_name is not None: # in this way, checkpoint will be warpped inside FSDP
                self.checkpoint_flags_stage = [0] * (self.stage_end_idx-self.stage_start_idx)

    def update_tensor_shape(self, microbatches, dp_size_input, dp_size, template_tensor_shape):
        # Update tensor_shape with correct microbatch_size
        tensor_shape, tensor_shape_last = copy.deepcopy(template_tensor_shape), copy.deepcopy(template_tensor_shape)

        microbatch_size = microbatches[0][0][0].size()[0] * dp_size_input // dp_size 
        microbatch_size_last = microbatches[0][0][0].size()[0] * dp_size_input // dp_size 
        
        for i in range(len(tensor_shape)):
            tensor_shape[i][0] = microbatch_size
            tensor_shape_last[i][0] = microbatch_size_last

        return tensor_shape, tensor_shape_last

    # pipeline rank utils
    # ---------------------------------------
    def get_pipeline_model_parallel_first_rank(self):
        return self.pp_global_ranks[0]


    def get_pipeline_model_parallel_last_rank(self):
        last_rank_local = self.group_size - 1
        return self.pp_global_ranks[last_rank_local]


    def get_pipeline_model_parallel_next_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline + 1) % world_size]


    def get_pipeline_model_parallel_prev_rank(self):
        rank_in_pipeline = self.group_rank
        world_size = self.group_size
        return self.pp_global_ranks[(rank_in_pipeline - 1) % world_size]

    def is_pipeline_first_stage(self):
        """Return True if in the first pipeline model-parallel stage, False otherwise."""
        return self.group_rank == 0


    def is_pipeline_last_stage(self):
        """Return True if in the last pipeline model-parallel stage, False otherwise."""
        return self.group_rank == (self.group_size - 1)

    # ---------------------------------------

    def gen_p2p_ops(
            self, 
            rank,
            ops,
            rank_recv_list=False,
            rank_send_list=False,
            tensor_recv=False,
            tensor_send=False,
            send=False, 
            empty_list=False,
        ):
        # self.show_process = True
        if send:
            for i in range(len(rank_send_list)):
                if empty_list[i]:
                    send_next_op = torch.distributed.P2POp(
                                        torch.distributed.isend,
                                        torch.zeros([]).cuda(),
                                        rank_send_list[i],
                                    )
                else:
                    if self.show_process:
                        print('Rank', self.global_rank, 'Start Sending Data with Shape:', tensor_send.shape, 'to Rank', rank_send_list[i])
                    send_next_op = torch.distributed.P2POp(
                                        torch.distributed.isend,
                                        tensor_send,
                                        rank_send_list[i],
                                    )
                ops.append(send_next_op)
        else:
            for i in range(len(rank_recv_list)):
                if empty_list[i]:
                    recv_prev_op = torch.distributed.P2POp(
                                        torch.distributed.irecv,
                                        torch.zeros([]).cuda(),
                                        rank_recv_list[i],
                                    )
                else:
                    if self.show_process:
                        print('Rank', self.global_rank, 'Start Recving Data with Shape:', tensor_recv.shape, 'from Rank', rank_recv_list[i])
                    recv_prev_op = torch.distributed.P2POp(
                                        torch.distributed.irecv,
                                        tensor_recv,
                                        rank_recv_list[i],
                                    )
                ops.append(recv_prev_op)
        return ops

        
    # p2p communication
    # ---------------------------------------
    def _run_p2pops(
            self,
            tensor_send_prev: Union[torch.Tensor, None],
            tensor_send_next: Union[torch.Tensor, None],
            tensor_recv_prev: Union[torch.Tensor, None],
            tensor_recv_next: Union[torch.Tensor, None],
    ):
        ops = []

        # serves for backward send
        if tensor_send_prev is not None:
            ops = self.gen_p2p_ops(
                        rank=self.global_rank, 
                        ops=ops, 
                        rank_send_list=self.bwd_send_list[self.global_rank], 
                        tensor_send=tensor_send_prev, 
                        send=True, 
                        empty_list=self.bwd_send_empty_list[self.global_rank],
                  )
            
        # serves for backward send
        if tensor_recv_next is not None:

            ops = self.gen_p2p_ops(
                        rank=self.global_rank, 
                        ops=ops, 
                        rank_recv_list=self.bwd_recv_list[self.global_rank], 
                        tensor_recv=tensor_recv_next, 
                        send=False, 
                        empty_list=self.bwd_recv_empty_list[self.global_rank],
                  )
        
        # serves for forward recv
        if tensor_recv_prev is not None:
            ops = self.gen_p2p_ops(
                        rank=self.global_rank, 
                        ops=ops, 
                        rank_recv_list=self.fwd_recv_list[self.global_rank], 
                        tensor_recv=tensor_recv_prev, 
                        send=False, 
                        empty_list=self.fwd_recv_empty_list[self.global_rank],
                  )
        
        # serves for forward send
        if tensor_send_next is not None:
            ops = self.gen_p2p_ops(
                        rank=self.global_rank, 
                        ops=ops, 
                        rank_send_list=self.fwd_send_list[self.global_rank], 
                        tensor_send=tensor_send_next, 
                        send=True, 
                        empty_list=self.fwd_send_empty_list[self.global_rank],
                  )

            
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()


    def _communicate(
        self,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Optional[Shape] = None,
        dtype_: Optional[torch.dtype] = None,
        *,
        scatter_gather_tensors_in_pipeline: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        fp32_residual_connection: bool = False,
    ) -> Tuple[Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Base function for communication of tensors between stages.

        dtype logic: If none of ``dtype_``, ``params_dtype``, ``fp32_residual_connection`` is specified,
        torch.float32 is used.

        See https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/arguments.py#L145-L159
        for the details of arguments of ``dtype_``, ``params_dtype``, ``fp32_residual_connection``.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if set to None).
            recv_prev: boolean for whether tensor should be received from previous rank.
            recv_next: boolean for whether tensor should be received from next rank.
            tensor_shape: optional, use when the input sequence contains less tokens than the default sequence length
            override_scatter_gather_tensors_in_pipeline:
                optional, this is used when tensor_shape is provided to override scatter gather tensors
            dtype_: This is used when tensor_shape is provided and what is the type of tensor_shape

        Keyword args:
            scatter_gather_tensors_in_pipeline: Optional. If :obj:`True`, use scatter/gather to optimize communication of tensors.
            params_dtype: Optional and legacy. Defaults to torch.float. If you manually call `.half()` or `.bfloat16()` on
                your model deliberately, pass this argument.
            fp32_residual_connection: Optional. If :obj:`True`, move residual connections to fp32.

        Returns:
            tuple containing

            - tensor_recv_prev: `torch.Tensor` if `recv_prev` is :obj:`True`, `None` otherwise.
            - tensor_recv_next: `torch.Tensor` if `recv_next` is :obj:`True`, `None` otherwise.
        """
        # Create placeholder tensors for receive in forward and backward directions if needed.
        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_shape is None:
            # In megatron, `tensor_shape` is set to `(args.seq_length, args.micro_batch_size, args.hidden_size)`
            raise RuntimeError(
                "`tensor_shape` must be specified. Common `tensor_shape` is `(seq_length, micro_batch_size, hidden_size)`")

        tensor_chunk_shape = tensor_shape

        # The dtype logic below is copied from NVIDIA/Megatron-LM repo:
        # https://github.com/NVIDIA/Megatron-LM/blob/d41696840ed0a7edb7e0499eb82a48ae112d9bb3/megatron/p2p_communication.py#L74-L81
        # NOTE (mkozuki): Currently NeMo is implementing APEX AMP O2 style using PyTorch. In O2 style, forcing p2p comm to
        # use FP32 will be a perf killer so that I decided to reanimate `dtype_` argument with the default value of `None`.
        # NOTE (mkozuki): In PyTorch AMP, i.e. `torch.cuda.amp.autocast` context, activation tensors can be either FP32,
        # FP16, or BF16 and there's no way to tell the dtypes of tensors on different devices in general.
        # It might be possible if we restrict model architecture.
        dtype = params_dtype or torch.float
        if fp32_residual_connection:
            dtype = torch.float
        requires_grad = True
        if dtype_ is not None:
            dtype = dtype_
            requires_grad = False

        global _RECV_BUFFER
        if recv_prev:
            tensor_recv_prev = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            ) 

            # _RECV_BUFFER = torch.empty(
            #     tensor_chunk_shape,
            #     requires_grad=requires_grad,
            #     device=torch.cuda.current_device(),
            #     dtype=dtype,
            # ) if _RECV_BUFFER is None else _RECV_BUFFER

            # tensor_recv_prev = _RECV_BUFFER


 
        if recv_next:
            tensor_recv_next = torch.empty(
                tensor_chunk_shape,
                requires_grad=requires_grad,
                device=torch.cuda.current_device(),
                dtype=dtype,
            ) 

            # _RECV_BUFFER = torch.empty(
            #     tensor_chunk_shape,
            #     requires_grad=requires_grad,
            #     device=torch.cuda.current_device(),
            #     dtype=dtype,
            # ) if _RECV_BUFFER is None else _RECV_BUFFER

            # tensor_recv_next = _RECV_BUFFER


        def p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next):
            commtype = ''
            if tensor_send_prev is not None:
                commtype += 'send_prev '
            if tensor_send_next is not None:
                commtype += 'send_next '
            if tensor_recv_prev is not None:
                commtype += 'recv_prev '
            if tensor_recv_next is not None:
                commtype += 'recv_next '
            return commtype
            
        commtype = p2p_type(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        if self.info:
            print('rank %d'%self.global_rank, 'start p2p', commtype)
        # Send tensors in both the forward and backward directions as appropriate.
        self._run_p2pops(tensor_send_prev, tensor_send_next, tensor_recv_prev, tensor_recv_next)
        # To protect against race condition when using batch_isend_irecv().
        # torch.cuda.synchronize()

        if self.info:
            print('rank %d'%self.global_rank, 'done p2p', commtype)

        return tensor_recv_prev, tensor_recv_next


    def recv_forward(
            self,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def recv_backward(
            self,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Receive tensor from next rank in pipeline (backward receive)."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward(
            self,
            output_tensor: torch.Tensor,
            tensor_shape: Shape = None,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        if self.is_pipeline_last_stage():
            return
        self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_backward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        if self.is_pipeline_first_stage():
            return
        self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )


    def send_forward_recv_backward(
            self,
            output_tensor: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with next rank in pipeline."""
        if self.is_pipeline_last_stage():
            return None
        _, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_backward_recv_forward(
            self,
            input_tensor_grad: torch.Tensor,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Union[None, torch.Tensor]:
        """Batched send and recv with previous rank in pipeline."""
        if self.is_pipeline_first_stage():
            return None
        input_tensor, _ = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_forward_recv_forward(
            self,
            output_tensor: torch.Tensor,
            recv_prev: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline."""
        input_tensor, _ = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor


    def send_backward_recv_backward(
            self,
            input_tensor_grad: torch.Tensor,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline."""
        _, output_tensor_grad = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return output_tensor_grad


    def send_forward_backward_recv_forward_backward(
            self,
            output_tensor: torch.Tensor,
            input_tensor_grad: torch.Tensor,
            recv_prev: bool,
            recv_next: bool,
            tensor_shape: Shape,
            *,
            dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched send and recv with previous and next ranks in pipeline."""
        input_tensor, output_tensor_grad = self._communicate(
            tensor_send_next=output_tensor.contiguous(),
            tensor_send_prev=input_tensor_grad.contiguous(),
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            dtype_=dtype,
        )
        return input_tensor, output_tensor_grad

    # ---------------------------------------

    # p2p communication multiple tensors
    # ---------------------------------------
    def recv_forward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            if len(dtypes) != len(tensor_shapes):
                dtypes.append(torch.float16)
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []

        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
            else:
                input_tensors.append(self.recv_forward(tensor_shape=tensor_shape, dtype=dtype))
                # print('recved!', input_tensors)

        return input_tensors


    def recv_backward_multi(
        self,
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
            else:
                output_tensor_grads.append(self.recv_backward(tensor_shape=tensor_shape, dtype=dtype))
        return output_tensor_grads


    def send_forward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
                
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            if len(dtypes) != len(tensor_shapes):
                dtypes.append(torch.float16)
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if output_tensor is None and tensor_shape is not None:
                output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_forward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)


    def send_backward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        *,
        dtypes = None,
    ) -> None:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            self.send_backward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)


    def send_forward_recv_backward_multi(
        self,
        output_tensors: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        output_tensor_grads = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            output_tensor = output_tensors[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                output_tensor_grads.append(None)
                continue
            if output_tensor is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    output_tensor = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    output_tensor = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            output_tensor_grad = self.send_forward_recv_backward(output_tensor, tensor_shape=tensor_shape, dtype=dtype)
            output_tensor_grads.append(output_tensor_grad)
        return output_tensor_grads


    def send_backward_recv_forward_multi(
        self,
        input_tensor_grads: Union[torch.Tensor, List[Union[None, torch.Tensor]]],
        tensor_shapes: List[Union[None, List[int]]],
        tensor_shapes_send = None,
        *,
        dtypes = None,
    ) -> List[Union[None, torch.Tensor]]:
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        if dtypes is not None:
            assert(len(dtypes) == len(tensor_shapes))
        input_tensors = []
        for i in range(len(tensor_shapes)):
            tensor_shape = tensor_shapes[i]
            input_tensor_grad = input_tensor_grads[i]
            dtype = None if dtypes is None else dtypes[i]
            if tensor_shape is None:
                input_tensors.append(None)
                continue
            if input_tensor_grad is None and tensor_shape is not None:
                if tensor_shapes_send is not None:
                    input_tensor_grad = torch.zeros(tensor_shapes_send[i],dtype=dtype).cuda(self.local_rank)
                else:
                    input_tensor_grad = torch.zeros(tensor_shape,dtype=dtype).cuda(self.local_rank)
            input_tensor = self.send_backward_recv_forward(input_tensor_grad, tensor_shape=tensor_shape, dtype=dtype)
            input_tensors.append(input_tensor)
        return input_tensors


