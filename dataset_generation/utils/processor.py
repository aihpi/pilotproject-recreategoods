"""
Util functions based on prompt-to-prompt.
"""
import string

import torch
from typing import Optional, List, Dict
from diffusers.models.attention import Attention
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import  abc
from typing import Optional, Union, Tuple, List, Callable, Dict


class CustomFluxAttnProcessor2_0:
    """Extended processor for SD3-like self-attention projections with external controller support."""

    def __init__(self, controller=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.controller = controller

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

         # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Use the controller to modify attention maps
        # if self.controller:
            # query, key, value = self.controller.modify_attention(query, key, value)

        # Compute scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        if self.controller:
            hidden_states = self.controller(hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        

def create_controller(prompts, num_inference_steps, joint_attention_kwargs: Dict, tokenizer):
    """
    Create a JointAttentionController instance to modify attention based on prompts.
    """
    # Initialize the controller
    joint_attn = JointAttentionController(
        words_shared=joint_attention_kwargs.get('shared_words', []),
        shared_factor=joint_attention_kwargs.get("shared_factor", 1.5),
        words_amplification=joint_attention_kwargs.get('amplify_words', []),
        amplification_factor=joint_attention_kwargs.get("amplification_factor", 1.5),
        words_suppression=joint_attention_kwargs.get('suppress_words', []),
        suppression_factor=joint_attention_kwargs.get("suppression_factor", 0.5),
        tokenizer=tokenizer,
    )

    # Set the prompt for the controller
    joint_attn.set_prompt(joint_attention_kwargs.get('resulting_caption'))
    controller = SelfAttentionControlEdit(prompts, num_inference_steps, joint_attention_kwargs.get('self_replace_steps'), joint_attn)
    return controller

class JointAttentionController:
    def __init__(self, words_shared, words_amplification, shared_factor, amplification_factor, words_suppression, suppression_factor, tokenizer, tokenized_prompt=None):
        self.words_shared = words_shared
        self.words_amplification = words_amplification
        self.words_suppression = words_suppression
        self.shared_factor = shared_factor
        self.amplification_factor = amplification_factor
        self.suppression_factor = suppression_factor
        self.tokenizer = tokenizer
        self.tokenized_prompt = tokenized_prompt

    def set_prompt(self, prompt):
        """Tokenize and cache the prompt."""
        self.tokenized_prompt = self.tokenizer(prompt, return_tensors="pt")

    def words_to_token_ids(self, words):
        """Map words to token IDs based on the cached tokenized prompt."""
        if self.tokenized_prompt is None:
            raise ValueError("Prompt has not been set. Call `set_prompt` first.")

        input_ids = self.tokenized_prompt.input_ids[0]

        word_to_token_ids = []
        for word in words:
            token_ids = self.tokenizer(word, add_special_tokens=False).input_ids
            for i in range(len(input_ids) - len(token_ids) + 1):
                if list(input_ids[i : i + len(token_ids)]) == token_ids:
                    word_to_token_ids.append(i)  # Append the start token position
                    break
        return word_to_token_ids

    def modify_attention(self, hidden_states):
        """
        Modify hidden states to influence attention.

        Args:
            hidden_states (torch.Tensor): Hidden states of shape [batch, seq_len, feature_dim].

        Returns:
            torch.Tensor: Modified hidden states.
        """

        # Amplify specific tokens in hidden states
        self.token_ids_to_copy = []
        if len(self.words_amplification) > 0:
            self.token_ids_to_copy = self.words_to_token_ids(self.words_amplification)
            for token_id in self.token_ids_to_copy:
                hidden_states[:, token_id, :] *= self.amplification_factor

        # Suppress specific tokens in hidden states
        if len(self.words_suppression) > 0:
            token_ids = self.words_to_token_ids(self.words_suppression)
            for token_id in token_ids:
                hidden_states[:, token_id, :] *= self.suppression_factor

        # Apply modifications for shared tokens
        if len(self.words_shared) > 0:
            token_ids = self.words_to_token_ids(self.words_shared)
            for token_id in token_ids:
                hidden_states[:, token_id, :] *= self.shared_factor

        return hidden_states



class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn):
        raise NotImplementedError

    def __call__(self, attn):
        attn = self.forward(attn)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.LOW_RESOURCE = False

        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn):
        # key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        # if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
        #     self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):

                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
       



def clean_and_tokenize(text):
    """
    Tokenizes a string into words, removing punctuation for consistency.
    """
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator).split()

def extract_key_changes(original_caption, resulting_caption):
    """
    Extract words added (for amplification) and removed (for suppression).
    """
    original_tokens = set(clean_and_tokenize(original_caption))
    resulting_tokens = set(clean_and_tokenize(resulting_caption))

    # Words added in the resulting caption
    added_words = resulting_tokens - original_tokens
    # Words removed from the original caption
    removed_words = original_tokens - resulting_tokens
    # Words shared between the original and resulting captions
    shared_words = original_tokens & resulting_tokens

    return list(added_words), list(removed_words), list(shared_words)

class SelfAttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 64 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            blend_factor = 0.95  # Adjust blend factor as needed
            merged_attention = blend_factor * attn_base + (1 - blend_factor) * att_replace
            return merged_attention
        else:
            return att_replace
    
    
    def forward(self, attn):
        super(SelfAttentionControlEdit, self).forward(attn)
        if (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_replace = attn[0], attn[1:]
            if self.joint_attn_controller:
                attn_replace = self.joint_attn_controller.modify_attention(attn_replace)
                attn[1:] = self.replace_self_attention(attn_base, attn_replace)
                for token_id in self.joint_attn_controller.token_ids_to_copy:
                    attn[1:, :, :, token_id] = attn_replace[:, :, :, token_id]
            else: attn[1:] = self.replace_self_attention(attn_base, attn_replace)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

        return attn
    
    def __init__(self, prompts, num_steps: int, self_replace_steps: Union[float, Tuple[float, float]], joint_attn_controller: JointAttentionController = None):
        super(SelfAttentionControlEdit, self).__init__()
        self.joint_attn_controller = joint_attn_controller
        self.batch_size = len(prompts)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])