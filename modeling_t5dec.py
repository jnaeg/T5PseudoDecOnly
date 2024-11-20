import copy
import math
import os
import warnings
import types
from typing import List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import T5PreTrainedModel, T5Config 
from transformers import T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering#, T5ForTokenClassification #NOTE: commented out for compatability with transformers 4.35, for 4.40, uncomment.
from transformers.generation.utils import Seq2SeqLMOutput 
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5Stack, T5LayerNorm,  T5Attention, T5DenseActDense, T5DenseGatedActDense, T5ClassificationHead 
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from transformers.modeling_utils import ModuleUtilsMixin


def invert_causal_attention_mask(self, encoder_attention_mask: Tensor, device = None) -> Tensor: #accepts ready encoder_attention_mask, adapted from ModuleUtilsMixin.
        """
        turns zeros in attention mask to minus inf, to be added to attention scores via position bias:
        https://github.com/huggingface/transformers/blob/cbe58b4269457a6ca66a556224b23f9ef246f905/src/transformers/models/t5/modeling_t5.py#L552
        https://github.com/huggingface/transformers/blob/cbe58b4269457a6ca66a556224b23f9ef246f905/src/transformers/models/t5/modeling_t5.py#L561


        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
        else:
            device = encoder_attention_mask.device

        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        if encoder_attention_mask.dim() == 4: #NOTE: manual workaround; hands ready causal attention mask to cross-attention layer 
            encoder_extended_attention_mask = encoder_attention_mask 
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask 

class T5_DecPreTrainedModel(T5PreTrainedModel): #adds T5PseudoDecForConditionalGeneration to the _init_weights function.
    r"""
    The T5PretrainedModel base class to handle weights initialisation/downloading/loading pretrained models, 
    adjusted for T5 Decoder-Only model (see NOTE). Inherits from PretrainedModel -- for more documentation, read there.
    """

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(
            module,
            (T5Model, T5ForConditionalGeneration, T5PseudoDecForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering), #NOTE: T5DecForConditionalGeneration,  added T5DecForConditionalGeneration to this conditional, remaining code unchanged
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                module.qa_outputs.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.data.zero_()
        #elif isinstance(module, T5ForTokenClassification): #NOTE: commented out for compatability with transformers 4.35; for 4.40, uncomment this block.
        #    if hasattr(module, "classifier"):
        #        module.classifier.weight.data.normal_(mean=0.0, std=factor * 1.0)
        #        module.classifier.bias.data.zero_()
        elif isinstance(module, T5ClassificationHead):
            module.dense.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.data.zero_()
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model**-0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))


class T5PseudoEncoder(nn.Module): #simple embedding layer fitted with interface of T5 encoder stack
    def __init__(self, config):
        super().__init__()
        self.return_dict = config.return_dict
        self.main_input_name = "input_ids"
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon) #added 06/05/24 to avoid numerical overflow in decoder hidden states

    def set_input_embeddings(self, new_embeddings):
        self.embedding = new_embeddings

    def forward(self, input_ids, **kwargs):
        hidden_states = self.embedding(input_ids)
        hidden_states = self.final_layer_norm(hidden_states) 
        if not self.return_dict:
            return tuple(hidden_states)
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state = hidden_states
        )

class T5PseudoDecForConditionalGeneration(T5_DecPreTrainedModel): #replaces T5 encoder stack with T5Pseudoencoder. 
    _tied_weights_keys = ["encoder.embedding.weight", "decoder.embed_tokens.weight", "lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight", r"encoder"] #first inherited from T5ForCond, second from T5EncoderModel 

    def __init__(self, config: T5Config): 
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model) #token embedding layer

        #pseudo encoder 
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5PseudoEncoder(encoder_config)

        #decoder
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False #NOTE: given we have kept a 'pseudo-decoder', model.config.is_encoder_decoder is True!
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'decoder.block.0': 0, 'decoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.decoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.decoder.block))
        self.decoder.parallelize(self.device_map)
        self.encoder = self.encoder.to(self.decoder.first_device) #NOTE: Debug -- not setting the final layerNorm of the PseudoEncoder to the last device but both to first could be problem?
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.decoder.deparallelize()
        self.decoder = self.decoder.to("cpu")
        self.encoder = self.encoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()    

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)       

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embedding, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)   

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder #returns pseudo-encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune): #adapted from T5EncoderModel 
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.decoder.block[layer].layer[0].SelfAttention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None, 
        decoder_attention_mask: Optional[torch.BoolTensor] = None, 
        head_mask: Optional[torch.FloatTensor] = None, 
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None, #hidden states,  do cross-attention over this in decoder.
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # automatically generated from input_ids 
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        reduce_loss: Optional[str] = None,
        causal_LM: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5PseudoDecForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # Traditional 'non-causal decoder' SeqSeq training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        
        >>> # For causal/fully autoregressive LM training either set labels = input_ids, or set 'causal_LM = True' in model call.

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.

        ATTENTION: We are inheriting problems from T5-base, and -large: 
        when finetuning using weights from_pretrained, use bf16-able hardware, e.g NVIDIA AX40, RTX A6000, or risk numerical overflow.
        If no bf16-able hardware available, and logits/loss/probs become nan or inf, finetune in fp32.
        See: https://github.com/huggingface/transformers/pull/10956 
        https://github.com/huggingface/accelerate/issues/243#issuecomment-1236975399
        ```"""
        self.causal_LM = causal_LM

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask; decoder_head_mask is simply the copy of head_mask (see T5Stack)
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # Autodetect autoregressive LM task training setting: 
        if labels is not None and input_ids is not None: 
            if torch.equal(labels, input_ids): 
                self.causal_LM = True

        if self.causal_LM:    
                input_ids = self._shift_right(input_ids) #else decoder is fed correct token one-by-one via causal cross-attention!
        else:
                pass #'non-causal' decoder-only model, see: https://arxiv.org/abs/2204.05832  

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:            
            encoder_outputs = self.encoder(
                input_ids=input_ids
                )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state= encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] 

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) 

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        
        # Decode
        if self.causal_LM:             
            self.decoder.invert_attention_mask = types.MethodType(invert_causal_attention_mask, self.decoder) #adds capability for causal cross-attention
            #create causal cross-attention mask
            batch_size, proto_seq_length = input_ids.size() 
            proto_input_shape = tuple(input_ids.size()) 
            if attention_mask is not None: 
                causal_cross_attention = self.create_extended_attention_mask_for_decoder(attention_mask = attention_mask, input_shape = proto_input_shape, device=hidden_states.device)  #make sure attention mask always on same device as hidden states   
            else: 
                proto_attention_mask = torch.ones(batch_size, proto_seq_length)#, device=inputs_embeds.device)
                causal_cross_attention = self.create_extended_attention_mask_for_decoder(attention_mask = proto_attention_mask, input_shape = proto_input_shape, device=hidden_states.device)

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids, 
                attention_mask=decoder_attention_mask, #triangular mask for causal self-attention, fed to layer 0 of T5Block
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=causal_cross_attention,  #set manually to causal cross attention mask
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

        else:
            self.decoder.invert_attention_mask = types.MethodType(T5PreTrainedModel.invert_attention_mask, self.decoder)

            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids, #during training shifted labels, during generation input_ids via prepare_inputs_for_generation.
                attention_mask=decoder_attention_mask, #triangular mask fed to layer 0 of T5Block. causal self-attention.
                inputs_embeds=decoder_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask, #all-ones mask fed to layer 1 of T5Block. non-causal cross-attention over input_embeds handed from encoder (projected to key/value states).
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism 
        if self.model_parallel:
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
  
        lm_logits = self.lm_head(sequence_output)

        #Loss Calculation  
        loss = None
        if labels is not None: #obtain loss over labels
            #adjust loss reduction method here: 
            if reduce_loss is not None: #manually apply loss reduction 
                weight = torch.ones(self.config.vocab_size)
                weight[self.config.pad_token_id] = 0
                weight = weight.to(lm_logits.device)
                if reduce_loss == 'mean':
                    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean', weight=weight)
                if reduce_loss == 'sum':    
                    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='sum', weight=weight)
                if reduce_loss == 'none':
                    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none', weight=weight)    
            else:    
                loss_fct = CrossEntropyLoss(ignore_index=-100) #defaults to mean, no manual weight 0 on padding token. 
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state
        )  
            

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
        
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    
    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past    

if __name__ == "__main__": #test model
    #follows example from https://github.com/huggingface/transformers/blob/cbe58b4269457a6ca66a556224b23f9ef246f905/src/transformers/models/t5/modeling_t5.py#L1678

    #import os
    import copy
    os.environ['TRANSFORMERS_OFFLINE']="1" #for when hugginface is down
    os.environ['HF_DATASETS_OFFLINE']="1"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    model = T5PseudoDecForConditionalGeneration.from_pretrained("google-t5/t5-small")

    #Autoregressive Language Modelling (inputs = labels)
    ar_input_ids = tokenizer("Studies have shown that owning a dog is good for you.", return_tensors="pt").input_ids
    ar_input_ids2 = copy.deepcopy(ar_input_ids)

    ar_outputs = model(input_ids=ar_input_ids, labels=ar_input_ids)
    ar_loss = ar_outputs.loss
    ar_logits = ar_outputs.logits
    print(f"AR loss:", ar_loss, f"\nAR logits:", ar_logits)

    # Seq 2 Seq Training
    input_ids = tokenizer("Studies have shown that owning a dog", return_tensors="pt").input_ids
    labels = tokenizer("is good for you.", return_tensors="pt").input_ids

    seq2seq_outputs = model(input_ids=input_ids, labels=labels)
    seq2seq_loss = seq2seq_outputs.loss
    seq2seq_logits = seq2seq_outputs.logits
    print(f"loss:", seq2seq_loss, f"\nlogits:", seq2seq_logits)

    #Inference
    #input_ids = tokenizer("Studies have shown that owning a dog is good for you.", return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    #Evaluate on original T5 tasks:

    EN2DE_ids = tokenizer("translate English to German: 'My house is beautiful.'", return_tensors="pt").input_ids
    EN2DE_outputs = model.generate(input_ids=EN2DE_ids)
    print(tokenizer.decode(EN2DE_outputs[0], skip_special_tokens=True))
                           
    DE2EN_ids = tokenizer("translate German to English: 'Mein Haus ist schön.'", return_tensors="pt").input_ids
    DE2EN_outputs = model.generate(input_ids=DE2EN_ids)
    print(tokenizer.decode(DE2EN_outputs[0], skip_special_tokens=True))

    EN2FR_ids = tokenizer("translate English to French: 'My house is beautiful.'", return_tensors="pt").input_ids
    EN2FR_outputs = model.generate(input_ids=EN2FR_ids)
    print(tokenizer.decode(EN2FR_outputs[0], skip_special_tokens=True))

    FR2EN_ids = tokenizer("translate French to English: 'Ma maison est belle.'", return_tensors="pt").input_ids
    FR2EN_outputs = model.generate(input_ids=FR2EN_ids)
    print(tokenizer.decode(FR2EN_outputs[0], skip_special_tokens=True))

    summ_ids = tokenizer("summarise: 'Truly this splendid house of mine is without compare, of beauty exquisie and refined.'", return_tensors="pt").input_ids
    summ_outputs = model.generate(input_ids=summ_ids)
    print(tokenizer.decode(summ_outputs[0], skip_special_tokens=True))

    #Visualise 
    #the causal cross attentions and self-attentions: do they converge after fine-tuning?

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    ar_outputs = model(input_ids=ar_input_ids, labels=ar_input_ids, output_attentions=True)

    num_heads = model.config.num_heads
    layers = model.config.num_layers

    cmap = plt.cm.jet

    cross_attentions = []
    for i in range(layers):
        for j in range(num_heads):
            ca = torch.squeeze(ar_outputs.cross_attentions[i][:, j, :, :].detach())
            image = cmap(ca)
            cross_attentions.append(image)

    self_attentions = []
    for i in range(layers):
        for j in range(num_heads):
            sa = torch.squeeze(ar_outputs.decoder_attentions[i][:, j, :, :].detach())
            image = cmap(sa)
            self_attentions.append(image)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  
                 nrows_ncols=(6, 8),  
                 axes_pad=0.1,  
                 )
    for ax, im in zip(grid, cross_attentions):
        ax.imshow(im)
    plt.savefig('cross_attentions.png')

    for ax, im in zip(grid, self_attentions):
        ax.imshow(im)
    plt.savefig('self_attentions.png')
    
    plt.clf()

    #DEBUG
    #torch.set_printoptions(linewidth=200) #look at attention masks more closely.
    #cross_attentions =torch.squeeze(ar_outputs.cross_attentions[0][:, 0, :, :].detach()) #
    #self_attentions = torch.squeeze(ar_outputs.decoder_attentions[0][:, 0, :, :].detach()) #they should both be triangular
    #print(cross_attentions)
    #print(self_attentions)

