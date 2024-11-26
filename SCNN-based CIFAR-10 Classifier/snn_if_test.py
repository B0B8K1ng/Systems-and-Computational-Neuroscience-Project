import torch
from spikingjelly.activation_based import neuron, surrogate

# 创建一个简单的IF神经元
if_node = neuron.IFNode(surrogate_function=surrogate.ATan())

# 假设有一个固定输入
input_tensor = torch.tensor([0.6])

# 模拟4个时间步
for t in range(4):
    # 输入相同，但输出可能不同
    output = if_node(input_tensor)
    print(f"Time step {t}:")
    print(f"Membrane potential: {if_node.v.item():.2f}")
    print(f"Output spike: {output.item():.0f}")
    print("---")