import torch as flow 

layernorm_output = flow.randn(4, 332, 512)
ltor_mask = flow.randn(4, 1, 332, 332)
position_embeddings = None
r_w_bias = None
r_r_bias = None
mem = None
layernorm_output = layernorm_output.to("cuda")
ltor_mask = ltor_mask.to("cuda")

for i in range(10000000):
    print(i)