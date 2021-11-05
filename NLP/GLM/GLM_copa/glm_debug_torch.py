import os
import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import LayerNorm
import torch.nn.functional as F
from mpu.transformer import ParallelSelfAttention, unscaled_init_method, scaled_init_method
from mpu.initialize import initialize_model_parallel
from pretrain_glm import initialize_distributed
from arguments import get_args


if __name__ == "__main__":
    # init_method = 'tcp://'
    # master_ip = os.getenv('MASTER_ADDR', 'localhost')
    # master_port = os.getenv('MASTER_PORT', '6000')
    # init_method += master_ip + ':' + master_port

    # torch.distributed.init_process_group(
    #     backend="nccl",
    #     world_size=1, 
    #     rank=1,
    #     init_method=init_method)
    # initialize_model_parallel(1)
    args = get_args()
    initialize_distributed(args)
    print("0")
    hidden_size = 512
    num_attention_heads = 8
    attention_dropout_prob = 0.1
    output_dropout_prob = 0.1
    init_method = unscaled_init_method(0.2)
    output_layer_init_method = scaled_init_method(0.2, 12)
    relative_encoding = False
    performer = False
    attention_scale = 1.0


    class AttenModel(torch.nn.Module):
        def __init__(self, ):
            super(AttenModel, self).__init__()
            self.atten = torch.nn.ModuleList(
                            [ParallelSelfAttention(
                                hidden_size,
                                num_attention_heads,
                                attention_dropout_prob,
                                output_dropout_prob,
                                init_method,
                                output_layer_init_method=output_layer_init_method,
                                relative_encoding=relative_encoding,
                                performer=performer,
                                attention_scale=attention_scale) 
                                for _ in range(12)
                            ]
                        )

        def forward(self, x, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem):
            for layer in self.atten:
                x = layer(x, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
            return x
    print("1")
    layernorm_output = torch.randn(4, 332, 512)
    ltor_mask = torch.randn(4, 1, 332, 332)
    position_embeddings = None
    r_w_bias = None
    r_r_bias = None
    mem = None
    layernorm_output = layernorm_output.to("cuda")
    ltor_mask = ltor_mask.to("cuda")
    print("2")
    model = AttenModel()
    model.train()
    model = model.to("cuda")
    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=0.01,
                        )
    for i in range(10000000):
        print("3")
        attention_output = model(layernorm_output, ltor_mask, position_embeddings, r_w_bias, r_r_bias, mem)
        optimizer.zero_grad()
        loss = attention_output.sum()
        loss.backward()
        optimizer.step()