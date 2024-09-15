from transformers.models.llama.modeling_llama import LlamaDecoderLayer

class AttentionRecorder:
    def __init__(self, attention_layer, attention_arg_pos = 0):
        self.orig_forward = attention_layer.forward
        self.attention_layer = attention_layer        
        self.attention_arg_pos = attention_arg_pos
        self.attention = []
        self.attention_arg_pos
        
    def __enter__(self):
        
        def _forward(*args, **kwargs):                
            output = self.orig_forward(*args, **kwargs)
            
            self.attention.append(output[self.attention_arg_pos])
            
            return output
            
        self.attention_layer.forward = _forward
            
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.attention_layer.forward = self.orig_forward

class AttentionFreezer:
    def __init__(self, attention_layer, frozen_tokens, frozen_attention, layer_no_attr = None):
        self.orig_forward = attention_layer.forward
        self.attention_layer = attention_layer
        self.frozen_tokens = frozen_tokens
        self.frozen_attention = frozen_attention
        self.layer_no_attr = layer_no_attr
        self.layer_no = 0        

    def __enter__(self):
        
        def _forward(*args, **kwargs):        
            if self.layer_no_attr:
                layer_no = eval(f"{self.attention_layer}.{self.layer_no_attr}")
            else:
                layer_no = self.layer_no

            output = self.orig_forward(*args, **kwargs)
                    
            if self.frozen_tokens and self.frozen_attention:
                output[0][:,self.frozen_tokens,:] = self.frozen_attention[self.layer_no][:,self.frozen_tokens,:]
                output[0][:,:,:] = 0
                self.layer_no += 1
            
            return output
            
        self.attention_layer.forward = _forward
            
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.attention_layer.forward = self.orig_forward
        
class HiddenStatesRecorder:
    def __init__(self, decoder_layer):
        self.orig_forward = decoder_layer.forward
        self.decoder_layer = decoder_layer        
        self.hidden_states = []
        
    def __enter__(self):
        
        def _forward(*args, **kwargs):                
            output = self.orig_forward(*args, **kwargs)
            
            self.hidden_states.append(output[0])
            
            return output
            
        self.decoder_layer.forward = _forward
            
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.decoder_layer.forward = self.orig_forward

class LlamaHiddenStatesRecorder(HiddenStatesRecorder):
    def __init__(self):
        super().__init__(decoder_layer=LlamaDecoderLayer)
        
class HiddenStatesFreezer:
    def __init__(self, decoder_layer, frozen_tokens, frozen_hidden_state, hidden_state_arg_pos, layer_no_attr = None):
        self.orig_forward = decoder_layer.forward
        self.decoder_layer = decoder_layer
        self.frozen_tokens = frozen_tokens
        self.frozen_hidden_state = frozen_hidden_state
        self.hidden_state_arg_pos = hidden_state_arg_pos
        self.layer_no_attr = layer_no_attr
        self.layer_no = 0        

    def __enter__(self):
        
        def _forward(*args, **kwargs):        
            if self.layer_no_attr:
                layer_no = eval(f"{self.decoder_layer}.{self.layer_no_attr}")
            else:
                layer_no = self.layer_no
    
            if self.frozen_tokens and self.frozen_hidden_state:
                args[self.hidden_state_arg_pos][:,self.frozen_tokens,:] = self.frozen_hidden_state[self.layer_no][:,self.frozen_tokens,:]
    
            self.layer_no += 1
            
            return self.orig_forward(*args, **kwargs)
            
        self.decoder_layer.forward = _forward
            
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.decoder_layer.forward = self.orig_forward

class LlamaHiddenStatesFreezer(HiddenStatesFreezer):
    def __init__(self, frozen_tokens, frozen_hidden_state):
        super().__init__(
            decoder_layer=LlamaDecoderLayer, 
            frozen_tokens=frozen_tokens, 
            frozen_hidden_state=frozen_hidden_state, 
            hidden_state_arg_pos=1)

