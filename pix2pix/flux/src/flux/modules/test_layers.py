
import pytest
import torch
from flux.modules.layers import DoubleStreamBlock

@pytest.fixture
def setup_block():
    hidden_size = 64
    num_heads = 8
    mlp_ratio = 4.0
    qkv_bias = False
    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
    
    batch_size = 2
    seq_length = 10
    hidden_dim_per_head = hidden_size // num_heads
    img = torch.randn(batch_size, seq_length, hidden_size)
    txt = torch.randn(batch_size, seq_length, hidden_size)
    vec = torch.randn(batch_size, hidden_size)
    # Initialize positional embeddings for RoPE with shape:
    # (batch_size, num_heads, seq_length * 2, hidden_dim_per_head // 2, 2, 2).
    # The last two dimensions represent real/imaginary parts and cosine/sine components.
    pe = torch.randn(batch_size, num_heads, seq_length * 2, hidden_dim_per_head // 2, 2, 2)
    
    return block, img, txt, vec, pe, hidden_size, num_heads, seq_length

def test_output_shape(setup_block):
    block, img, txt, vec, pe, hidden_size, num_heads, seq_length = setup_block
    img_out, txt_out = block(img, txt, vec, pe)
    assert img_out.shape == (img.size(0), seq_length, hidden_size)
    assert txt_out.shape == (txt.size(0), seq_length, hidden_size)

def test_output_type(setup_block):
    block, img, txt, vec, pe, *_ = setup_block
    img_out, txt_out = block(img, txt, vec, pe)
    assert isinstance(img_out, torch.Tensor)
    assert isinstance(txt_out, torch.Tensor)

def test_forward_pass(setup_block):
    block, img, txt, vec, pe, *_ = setup_block
    img_out, txt_out = block(img, txt, vec, pe)
    assert torch.is_tensor(img_out)
    assert torch.is_tensor(txt_out)

def test_prompt_to_prompt():
    hidden_size = 64
    num_heads = 8
    mlp_ratio = 4.0
    qkv_bias = False
    block = DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias, prompt_to_prompt=True)
    
    seq_length = 10
    hidden_dim_per_head = hidden_size // num_heads
    img = torch.randn(4, seq_length, hidden_size)
    txt = torch.randn(4, seq_length, hidden_size)
    vec = torch.randn(4, hidden_size)
    pe = torch.randn(4, num_heads, seq_length * 2, hidden_dim_per_head // 2, 2, 2)
    
    img_out, txt_out = block(img, txt, vec, pe)
    assert img_out.shape == (4, seq_length, hidden_size)
    assert txt_out.shape == (4, seq_length, hidden_size)