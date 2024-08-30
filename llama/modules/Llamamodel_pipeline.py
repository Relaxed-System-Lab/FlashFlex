import torch.nn as nn

try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None
    
try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None


class LlamaEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['embeddings', 'process_group', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
    
    def label(self):
        return [0,0]
    
    def forward(self, input_ids, position_ids=None):
        
        embedding_kwargs = ({'combine_batch_seqlen_dim': True}
                            if self.process_group is not None and self.sequence_parallel else {})
        hidden_states = self.embeddings(input_ids, position_ids=position_ids, **embedding_kwargs)
        
        return hidden_states

class LlamaLayers_(nn.Module):
    def __init__(self, model, layer_idx_start, layer_idx_end):
        super().__init__()
        model = model.transformer
        self.layer_idx = layer_idx_start
        self.layers = model.layers[layer_idx_start:layer_idx_end]
        attrs = ['prenorm', 'parallel_block', 'process_group']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def label(self):
        return [1,self.layer_idx]
    
    def forward(self, hidden_states, residual=None):

        mixer_kwargs = ({'seqlen': hidden_states.shape[1]}
                        if self.process_group is not None and self.sequence_parallel else {})
        
        for layer in self.layers:
            if self.prenorm:
                if not self.parallel_block:
                    
                    hidden_states, residual = layer(hidden_states, residual,
                                                    mixer_kwargs=mixer_kwargs)

            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)

        
        return hidden_states

class LlamaPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        self.drop_f = model.drop_f
        self.ln_f = model.ln_f
        attrs = ['fused_dropout_add_ln', 'drop_f', 'parallel_block', 'ln_f', 'prenorm', 'residual_in_fp32']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def label(self):
        return [2,0]

    def forward(self, hidden_states, residual=None):
        
        assert(residual is None)
        residual = None
        if self.prenorm:
            if not self.fused_dropout_add_ln:
                dropped = self.drop_f(hidden_states)
                if not self.parallel_block:
                    residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                if not self.parallel_block:
                    fused_add_norm_fn = (dropout_add_rms_norm if isinstance(self.ln_f, RMSNorm)
                                         else dropout_add_layer_norm)
                    hidden_states = fused_add_norm_fn(
                        hidden_states, residual, self.ln_f.weight, self.ln_f.bias,
                        self.drop_f.p if self.training else 0.0, self.ln_f.eps, prenorm=False,
                        residual_in_fp32=self.residual_in_fp32
                    )

        return hidden_states

class LlamaCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        attrs = ['lm_head', 'config', 'project_out']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def label(self):
        return [3,0]

    def forward(self, hidden_states):
        
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        
        lm_logits = self.lm_head(hidden_states)
        
        return lm_logits
