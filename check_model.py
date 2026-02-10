import torch, json, pathlib
from safetensors.torch import safe_open   # pip install safetensors
from diffusers import FluxTransformer2DModel

model_dir = pathlib.Path("/home/felix.boelter/.cache/huggingface/hub/models--aihpi--fashion-edit-model/snapshots/65cd4e4440a813d577aeae929ffe81a43de82898/")   # folder that has /transformer
ckpt      = model_dir / "transformer" / "diffusion_pytorch_model-00003-of-00005.safetensors"

# 1) print a couple of key tensor shapes -------------------------------
with safe_open(ckpt, framework="pt", device="cpu") as f:
    q_weight = f.get_tensor("single_transformer_blocks.0.attn.to_q.weight")   # first block is enough
    print("to_q.weight :", tuple(q_weight.shape))
ckpt      = model_dir / "transformer" / "diffusion_pytorch_model-00001-of-00005.safetensors"
with safe_open(ckpt, framework="pt", device="cpu") as f:
    x_weight = f.get_tensor("x_embedder.weight")
    print("x_embedder :", tuple(x_weight.shape))

# 2) compare to the JSON ------------------------------------------------
cfg_file = model_dir / "transformer" / "config.json"
cfg = json.loads(cfg_file.read_text())

print("\nconfig says:")
print("  in_channels          :", cfg["in_channels"])
print("  attention_head_dim   :", cfg["attention_head_dim"])
print("  num_attention_heads  :", cfg["num_attention_heads"])
print("  joint_attention_dim  :", cfg["joint_attention_dim"])
print("  sum(axes_dims_rope)  :", sum(cfg["axes_dims_rope"]))