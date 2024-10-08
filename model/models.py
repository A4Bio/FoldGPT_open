import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache,  StaticCache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from data.pdb_tokenizer import PDBTokenizer
from data.datasets import generate_pos_sequence
from .llama_modules import LlamaModel, LlamaAttention


class VQAE(nn.Module):
    def __init__(self):
        super(VQAE, self).__init__()
        self.special_code = nn.Embedding(300, 128)

    def code2vec(self, code, max_code=255):
        
        special_embed = self.special_code(code)
        vec = special_embed/(special_embed.norm(dim=-1, keepdim=True)+1e-6)
        return vec


@dataclass
class CustomCausalLMOutputWithPast(ModelOutput):
    def __init__(self, 
                 loss: Optional[torch.FloatTensor] = None,
                 logits: Optional[torch.FloatTensor] = None,
                 logits_pos: Optional[torch.FloatTensor] = None,
                 logits_chain: Optional[torch.FloatTensor] = None,
                 past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                 hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
                 attentions: Optional[Tuple[torch.FloatTensor]] = None):
        self.loss = loss
        self.logits = logits
        self.logits_pos = logits_pos
        self.logits_chain = logits_chain
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions


MAX_CAHIN = 1000
class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, binary_code=1):
        super().__init__(config)
        self.binary_code = binary_code
        config.num_hidden_layers = 12
        self.vqae = VQAE()
        self.enc_model = LlamaModel(config)
        log2_num_embeddings = 8
        self.log2_num_embeddings = log2_num_embeddings
        if binary_code:
            self.vocab_size = log2_num_embeddings
            # self.vqid_encoding = nn.Linear(log2_num_embeddings, config.hidden_size)
            self.vqid_encoding = nn.Linear(128, config.hidden_size)
        else:
            self.vocab_size = 269
            self.vqid_encoding = nn.Embedding(self.vocab_size, config.hidden_size)
            

        self.lm_head = nn.Linear(config.hidden_size, 128, bias=False)
        self.chain_encoding = nn.Embedding(MAX_CAHIN, config.hidden_size)
        
        self.pos_encoding = nn.Embedding(1025, config.hidden_size)
        self.query_encoding = nn.Embedding(1, config.hidden_size)
        # self.seg_encoding = nn.Embedding(2, config.hidden_size)
        self.pred_attn = LlamaAttention(config=config, layer_idx=0)

        int_range = torch.arange(0, 2**log2_num_embeddings)
        bool_vectors = (int_range[:, None] & (1 << torch.arange(log2_num_embeddings-1, -1, -1))) > 0

        self.register_buffer('vq_embedding', bool_vectors.float())

        self.tokenizer = PDBTokenizer()


        # Initialize weights and apply final processing
        self.post_init()

    def decimal2binary(self, vqids):
        return self.vq_embedding[vqids]
    
    def binary2decimal(self, binary_vector):
        base = 2 ** torch.arange(binary_vector.size(-1)-1, -1, -1, device=binary_vector.device)
        vqids = sum((binary_vector * base).long(), dim=-1)
        return vqids

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=0)
            # causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_condition = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        pos_ids = position_ids
        position_ids = None

    
        if self.binary_code:
            # binary_input_ids = self.decimal2binary(input_ids)
            # key_embeds = self.vqid_encoding(binary_input_ids)
            key_embeds = self.vqid_encoding(self.vqae.code2vec(input_ids))
        else:
            key_embeds = self.vqid_encoding(input_ids)
        # query_embeds = self.query_encoding(torch.zeros_like(pos_ids))
        # key_embeds = key_embeds+self.seg_encoding(is_condition.long())
        
        input_ids = None

        if attention_mask is None:
            attention_mask = torch.ones_like(pos_ids).float()

        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.enc_model(
            input_ids=input_ids,
            inputs_embeds = key_embeds,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            is_condition = is_condition,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        
        
        centers = self.vqae.code2vec(torch.arange(269, device=logits.device)[None])[0]
        logits = torch.einsum('bld,kd->blk', logits, centers)

        loss = None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CustomCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # logits_pos = logits_pos,
            # logits_chain = logits_chain,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        prompt_pos_ids: List[List[int]],
        max_gen_len: int,
        is_condition_flags: torch.TensorType,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        max_batch_size=32,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        self.eval()
        bsz = len(prompt_tokens)
        assert bsz <= max_batch_size, (bsz, max_batch_size)
        max_seq_len = max_gen_len
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        # assert max_prompt_len <= max_seq_len
        total_len = max_prompt_len+max_gen_len*2
        # total_len = min(max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        pos_ids = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        is_condition = torch.full((bsz, total_len), 0, dtype=torch.long, device="cuda")
        mask_indices_list = []
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

            pos_ids[k, : len(t)] = torch.tensor(prompt_pos_ids[k], dtype=torch.long, device="cuda")

            is_condition[k, : len(t)] = torch.tensor(is_condition_flags[k], dtype=torch.long, device="cuda")
        
            mask_indices = generate_pos_sequence(max_prompt_len+max_gen_len-1, prompt_pos_ids[k][1:])
            pos_ids[k, len(t):] = torch.tensor(mask_indices).repeat_interleave(2)
            mask_indices_list.append(mask_indices)
            
            
            
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor([self.tokenizer.eos_token_id], device="cuda")

        for cur_pos in range(min_prompt_len, total_len, 2):
            output = self(tokens[:, :cur_pos+1], position_ids=pos_ids[:,:cur_pos+1], is_condition = is_condition[:, :cur_pos+1]==1)
            prob = F.softmax(output.logits[:,-1]/temperature, dim=-1)
            next_token = torch.multinomial(prob, 1)

            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos+1] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos+1]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        
        out_tokens = []
        for i in range(tokens.shape[0]):
            isdata = tokens[i]<256
            toks = tokens[i][isdata]
            pos = pos_ids[i][isdata]
            toks_sort = torch.zeros_like(toks)
            toks_sort[pos] = toks
            out_tokens.append(toks_sort)
        return out_tokens

        # if logprobs:
        #     token_logprobs = token_logprobs.tolist()
        # out_tokens, out_logprobs = [], []
        # for i, toks in enumerate(tokens.tolist()):
        #     # cut to max gen len
        #     start = 0 if echo else len(prompt_tokens[i])
        #     toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        #     probs = None
        #     if logprobs:
        #         probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
        #     # cut to after eos tok if any
        #     for token_id in stop_tokens:
        #         try:
        #             eos_idx = toks.index(token_id)
        #             toks = toks[:eos_idx]
        #             probs = probs[:eos_idx] if logprobs else None
        #         except ValueError:
        #             pass
        #     toks = torch.tensor(toks)
        #     toks_sort = torch.zeros_like(toks)
        #     toks_sort[mask_indices_list[i]] = toks
        #     out_tokens.append(toks_sort)
        #     out_logprobs.append(probs)
        

        # return (out_tokens, out_logprobs if logprobs else None)
    






    
    

