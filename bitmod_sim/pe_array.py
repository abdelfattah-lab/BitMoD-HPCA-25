from typing import List
import numpy as np
import math, pickle


class PE_Array:
    PR_SCALING = 1.5 # scaling factor to account for post placement and routing

    ## The class constructor
    # @param model-name:    Name of the model to be evaluated.
    # @param i_prec:        The input activation precision.
    # @param kv_prec:       The KV-cache precision.
    # @param w_prec:        The weight precision.
    # @param batch_size:    The batch size.
    # @param is_bit_serial: Whether using the bit-serial computing paradigm.
    # @param pe_dp_size:    The dot-product size of the PE.
    # @param pe_energy:     The energy cost of PE.
    # @param cxt_len:       The input context length.
    # @param is_generation: Whether the simulation mode is generation stage or prefill stage.
    def __init__(
        self,
        model_name: str,
        i_prec: int=16, 
        kv_prec: int=8,
        w_prec: int=8, 
        batch_size:int=1,
        is_bit_serial: bool=False,
        pe_dp_size: int=1,
        pe_energy: float=0, 
        pe_area: float=0, 
        pe_array_dim: List[int]=[],
        cxt_len: int=256,
        is_generation: bool=False,
    ):
        assert pe_energy != 0, "ERROR! You must provide the energy cost of a PE."
        assert len(pe_array_dim) == 2, f"ERROR! The dimension of PE array must be 2. But you gave {len(pe_array_dim)}."
        
        self.model_name    = model_name
        self.i_prec        = i_prec
        self.kv_prec       = kv_prec
        self.w_prec        = w_prec
        self.batch_size    = batch_size
        self.is_bit_serial = is_bit_serial

        self.pe_latency = {}
        if is_bit_serial:
            self.pe_latency['attn'] = math.ceil(math.floor(kv_prec) / 2)
            self.pe_latency['linear'] = math.ceil(math.floor(w_prec) / 2)
        else:
            self.pe_latency['attn'] = 1
            self.pe_latency['linear'] = 1
        self.pe_dp_size = pe_dp_size
        self.total_pe_count = np.prod(pe_array_dim)
        self.pe_energy      = pe_energy * self.PR_SCALING
        self.pe_area        = pe_area * self.PR_SCALING
        self.pe_array_area  = pe_area * self.total_pe_count
        self.pe_array_dim   = {'h': pe_array_dim[0], 'w': pe_array_dim[1]}
        
        self._init_model_profiler(model_name, cxt_len, is_generation)
    
    def _init_model_profiler(self, model_name, cxt_len: int=256, is_generation: bool=False):
        model_name_dict = {
            "facebook/opt-1.3b": "opt_1_point_3", 
            "microsoft/phi-2": "phi_2",
            "01-ai/Yi-6B": "yi_6",
            "meta-llama/Llama-2-7b-hf": "llama_2_7", 
            "meta-llama/Llama-2-13b-hf": "llama_2_13", 
            "meta-llama/Meta-Llama-3-8B": "llama_3_8", 
            "meta-llama/Llama-3.1-8B-Instruct": "llama_3p1_8_ins", 
            "meta-llama/Llama-3.2-3B-Instruct": "llama_3p2_3_ins", 
        }
        file_path = f'./model_shape_config/{model_name_dict[model_name]}.pickle'
        with open(file_path, 'rb') as f:
            model_config, layer_config = pickle.load(f)
        
        ########## FFN Dimension ##########
        batch_size = self.batch_size
        weight_dim = {}
        input_dim  = {}
        output_dim = {}
        for name, weight_shape in layer_config.items():
            weight_dim[name] = [1] + weight_shape
            if is_generation: # generation
                input_dim[name]  = [1, batch_size, weight_shape[1]]
                output_dim[name] = [1, batch_size, weight_shape[0]]
            else:
                input_dim[name]  = [batch_size, cxt_len, weight_shape[1]]
                output_dim[name] = [batch_size, cxt_len, weight_shape[0]]

        ########## Attention Dimension ##########
        num_hidden_layers   = model_config['num_hidden_layers']
        hidden_size         = model_config['hidden_size']
        num_attention_heads = model_config['num_attention_heads']
        head_size           = hidden_size / num_attention_heads
        if 'num_key_value_heads' in model_config.keys():
            num_key_value_heads = model_config['num_key_value_heads']
        else:
            num_key_value_heads = num_attention_heads
        query_share_factor = num_attention_heads / num_key_value_heads

        for l_idx in range(num_hidden_layers):
            op_name = f'model.layers.{l_idx}.self_attn.attn_qk'
            if is_generation: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, cxt_len, head_size] # key dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # query dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
            
            op_name = f'model.layers.{l_idx}.self_attn.attn_v'
            if is_generation: # generation
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor, head_size] # output dimension
            else:
                weight_dim[op_name] = [batch_size * num_key_value_heads, head_size, cxt_len] # value dimension
                input_dim[op_name]  = [batch_size * num_key_value_heads, query_share_factor * cxt_len, cxt_len] # score dimension
                output_dim[op_name] = [batch_size * num_key_value_heads, query_share_factor * cxt_len, head_size] # output dimension

        self.weight_dim = weight_dim
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.layer_name_list = list(weight_dim.keys())

    def _init_mem(self):
        raise NotImplementedError('ERROR! No implementation of function _init_mem()')

