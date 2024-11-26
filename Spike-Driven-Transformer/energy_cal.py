import yaml
import json
import math
import logging

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

class EnergyCalculator:
    def __init__(self, config_file, log_file):
        # 基础参数保持不变
        self.EMAC = 3.7
        self.EAC = 0.9
        
        self.paper_results = {
            'total_energy': 6.09,
            'vanilla_transformer': 41.77,
        }
        
        self.config = self._load_config(config_file)
        self.firing_rates = self._load_firing_rates(log_file)
        self.init_model_params()
        
        # 输出初始化参数
        self._print_init_params()
        
    def _print_init_params(self):
        """打印初始化参数"""
        logger.info("Initialized Energy Calculator with parameters:")
        logger.info(f"Image Size: {self.img_size}x{self.img_size}")
        logger.info(f"Embedding Dimension: {self.embed_dims}")
        logger.info(f"Number of Heads: {self.num_heads}")
        logger.info(f"Time Steps: {self.time_steps}")
        logger.info(f"MLP Ratio: {self.mlp_ratio}")
        logger.info(f"Number of Layers: {self.layers}")
        logger.info(f"Number of Tokens (N): {self.N}")
        logger.info(f"Channel Dimension (D): {self.D}")
        logger.info("----------------------------------------")
        
    def calc_sps_flops(self):
        """修正SPS模块的FLOPs计算"""
        flops = {}
        H, W = self.img_size, self.img_size
        
        # Conv1: 3->D/8, stride=1, padding=1
        C_in, C_out = 3, self.D//8
        flops['conv1'] = H * W * 3 * 3 * C_in * C_out
        
        # Conv2: D/8->D/4, with maxpool
        H, W = H//2, W//2
        C_in, C_out = self.D//8, self.D//4
        flops['conv2'] = H * W * 3 * 3 * C_in * C_out
        
        # Conv3: D/4->D/2
        H, W = H//2, W//2
        C_in, C_out = self.D//4, self.D//2
        flops['conv3'] = H * W * 3 * 3 * C_in * C_out
        
        # Conv4: D/2->D
        H, W = H//2, W//2
        C_in, C_out = self.D//2, self.D
        flops['conv4'] = H * W * 3 * 3 * C_in * C_out
        
        return flops

    def calc_sps_energy(self):
        """计算SPS模块能耗"""
        flops = self.calc_sps_flops()
        rates = self.firing_rates['t0']
        
        logger.info("\nSPS Module Energy Calculation:")
        logger.info("----------------------------------------")
        
        # Conv1
        e1 = self.EMAC * self.time_steps * rates['MS_SPS_lif'] * flops['conv1']
        logger.info(f"Conv1 FLOPs: {flops['conv1']:,}")
        logger.info(f"Conv1 Firing Rate: {rates['MS_SPS_lif']:.4f}")
        logger.info(f"Conv1 Energy: {e1*1e-9:.4f} mJ")
        
        # Conv2
        e2 = self.EAC * self.time_steps * rates['MS_SPS_lif1'] * flops['conv2']
        logger.info(f"\nConv2 FLOPs: {flops['conv2']:,}")
        logger.info(f"Conv2 Firing Rate: {rates['MS_SPS_lif1']:.4f}")
        logger.info(f"Conv2 Energy: {e2*1e-9:.4f} mJ")
        
        # Conv3
        e3 = self.EAC * self.time_steps * rates['MS_SPS_lif2'] * flops['conv3']
        logger.info(f"\nConv3 FLOPs: {flops['conv3']:,}")
        logger.info(f"Conv3 Firing Rate: {rates['MS_SPS_lif2']:.4f}")
        logger.info(f"Conv3 Energy: {e3*1e-9:.4f} mJ")
        
        # Conv4
        e4 = self.EAC * self.time_steps * rates['MS_SPS_lif3'] * flops['conv4']
        logger.info(f"\nConv4 FLOPs: {flops['conv4']:,}")
        logger.info(f"Conv4 Firing Rate: {rates['MS_SPS_lif3']:.4f}")
        logger.info(f"Conv4 Energy: {e4*1e-9:.4f} mJ")
        
        total = e1 + e2 + e3 + e4
        logger.info(f"\nTotal SPS Energy: {total*1e-9:.4f} mJ")
        
        return total
        
    def calc_sdsa_energy_for_layer(self, layer_idx):
        """计算指定层的SDSA能耗"""
        rates = self.firing_rates['t0']  # 可以根据需要使用不同时间步的firing rate
        layer_prefix = f'MS_SSA_Conv{layer_idx}_'
        
        # Q,K,V生成能耗，考虑多头注意力
        qkv_flops_per_head = 3 * self.N * (self.D // self.num_heads) * (self.D // self.num_heads)
        qkv_flops = qkv_flops_per_head * self.num_heads
        
        qkv_rates = (rates[f'{layer_prefix}q_lif'] + 
                    rates[f'{layer_prefix}k_lif'] + 
                    rates[f'{layer_prefix}v_lif'])
        qkv_energy = self.EAC * self.time_steps * qkv_rates * qkv_flops
        
        # 注意力计算能耗
        attn_flops_per_head = self.N * (self.D // self.num_heads)
        attn_flops = attn_flops_per_head * self.num_heads
        attn_energy = self.EAC * self.time_steps * rates[f'{layer_prefix}kv'] * attn_flops
        
        total = qkv_energy + attn_energy
        
        if layer_idx == 0:  # 只在第一层记录详细日志
            logger.info("\nSDSA Module Energy Calculation:")
            logger.info("----------------------------------------")
            logger.info(f"QKV Generation FLOPs per head: {qkv_flops_per_head:,}")
            logger.info(f"Total QKV Generation FLOPs: {qkv_flops:,}")
            logger.info(f"Q Firing Rate: {rates[f'{layer_prefix}q_lif']:.4f}")
            logger.info(f"K Firing Rate: {rates[f'{layer_prefix}k_lif']:.4f}")
            logger.info(f"V Firing Rate: {rates[f'{layer_prefix}v_lif']:.4f}")
            logger.info(f"QKV Generation Energy: {qkv_energy*1e-9:.4f} mJ")
            logger.info(f"\nAttention FLOPs per head: {attn_flops_per_head:,}")
            logger.info(f"Total Attention FLOPs: {attn_flops:,}")
            logger.info(f"Attention Firing Rate: {rates[f'{layer_prefix}kv']:.4f}")
            logger.info(f"Attention Energy: {attn_energy*1e-9:.4f} mJ")
            logger.info(f"\nTotal SDSA Energy: {total*1e-9:.4f} mJ")
            
            # 计算与vanilla self-attention的能耗比
            vanilla_attn_flops = 2 * self.N * self.N * self.D
            vanilla_energy = self.EMAC * vanilla_attn_flops
            energy_ratio = vanilla_energy / total
            logger.info(f"\nEnergy Ratio (Vanilla/SDSA): {energy_ratio:.2f}x")
        
        return total
        
    def calc_mlp_energy_for_layer(self, layer_idx):
        """计算指定层的MLP能耗"""
        rates = self.firing_rates['t0']
        layer_prefix = f'MS_MLP_Conv{layer_idx}_'
        
        # FC1: D -> 4D
        fc1_flops = self.N * self.D * (4 * self.D)
        fc1_energy = self.EAC * self.time_steps * rates[f'{layer_prefix}fc1_lif'] * fc1_flops
        
        # FC2: 4D -> D
        fc2_flops = self.N * (4 * self.D) * self.D
        fc2_energy = self.EAC * self.time_steps * rates[f'{layer_prefix}fc2_lif'] * fc2_flops
        
        total = fc1_energy + fc2_energy
        
        if layer_idx == 0:  # 只在第一层记录详细日志
            logger.info("\nMLP Module Energy Calculation:")
            logger.info("----------------------------------------")
            logger.info(f"FC1 FLOPs: {fc1_flops:,}")
            logger.info(f"FC1 Firing Rate: {rates[f'{layer_prefix}fc1_lif']:.4f}")
            logger.info(f"FC1 Energy: {fc1_energy*1e-9:.4f} mJ")
            logger.info(f"\nFC2 FLOPs: {fc2_flops:,}")
            logger.info(f"FC2 Firing Rate: {rates[f'{layer_prefix}fc2_lif']:.4f}")
            logger.info(f"FC2 Energy: {fc2_energy*1e-9:.4f} mJ")
            logger.info(f"\nTotal MLP Energy: {total*1e-9:.4f} mJ")
            
        return total

    def calc_total_energy(self):
        """计算总能耗"""
        logger.info("\nCalculating Total Energy Consumption:")
        logger.info("========================================")
        
        # 只计算一次SPS能耗
        sps_energy = self.calc_sps_energy()
        
        # 累加每层的SDSA和MLP能耗
        total_layer_energy = 0
        for i in range(self.layers):
            layer_sdsa = self.calc_sdsa_energy_for_layer(i)
            layer_mlp = self.calc_mlp_energy_for_layer(i)
            total_layer_energy += layer_sdsa + layer_mlp
        
        # 总能耗(转换为mJ)
        total_energy = (sps_energy + total_layer_energy) * 1e-9
        
        # 记录结果
        results = {
            'sps_energy': sps_energy * 1e-9,
            'layer_energy': total_layer_energy * 1e-9 / self.layers,  # 每层平均能耗
            'total_energy': total_energy
        }
        
        logger.info("\nFinal Energy Results:")
        logger.info("----------------------------------------")
        logger.info(f"SPS Energy (once): {results['sps_energy']:.4f} mJ")
        logger.info(f"Average Layer Energy: {results['layer_energy']:.4f} mJ")
        logger.info(f"Total Energy: {results['total_energy']:.4f} mJ")
        
        self.compare_with_paper(results)
        
        return results

    def compare_with_paper(self, our_results):
        """与论文结果比较"""
        logger.info("\nComparison with Paper Results:")
        logger.info("----------------------------------------")
        logger.info(f"Our Total Energy: {our_results['total_energy']:.3f} mJ")
        logger.info(f"Paper Reported Energy: {self.paper_results['total_energy']:.3f} mJ")
        logger.info(f"Difference: {abs(our_results['total_energy'] - self.paper_results['total_energy']):.3f} mJ")
        logger.info(f"Relative Difference: {abs(our_results['total_energy'] - self.paper_results['total_energy'])/self.paper_results['total_energy']*100:.2f}%")
        
        # 与vanilla transformer比较
        energy_saving = (self.paper_results['vanilla_transformer'] - our_results['total_energy'])/self.paper_results['vanilla_transformer']*100
        logger.info(f"\nEnergy Saving vs Vanilla Transformer: {energy_saving:.2f}%")
        logger.info(f"(Vanilla Transformer: {self.paper_results['vanilla_transformer']:.2f} mJ)")

    def _load_config(self, config_file):
        """加载yaml配置文件"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
            
    def _load_firing_rates(self, log_file):
        """从日志文件中提取firing rates"""
        with open(log_file, 'r') as f:
            content = f.read()
            # 找到firing_rate部分
            start_str = 'firing_rate: \n'
            start_idx = content.find(start_str) + len(start_str)
            # 找到json数据的开始和结束
            json_start = content[start_idx:].find('{') + start_idx
            json_end = content[json_start:].find('\n}') + json_start + 2
            
            # 提取json字符串
            firing_rates_str = content[json_start:json_end]
            try:
                return json.loads(firing_rates_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析错误: {e}")
                logger.error(f"尝试解析的字符串: {firing_rates_str}")
                raise

    def init_model_params(self):
        """初始化模型参数"""
        self.img_size = self.config['img_size']  # 224
        self.embed_dims = self.config['dim']     # 768
        self.num_heads = self.config['num_heads'] # 8
        self.time_steps = self.config['time_steps'] # 4
        self.mlp_ratio = self.config['mlp_ratio']  # 4
        self.layers = self.config['layer']        # 8
        
        # 计算其他参数
        self.patch_size = 16  # 默认值
        self.N = (self.img_size // self.patch_size) ** 2  # token数
        self.D = self.embed_dims

def main():
    calculator = EnergyCalculator(
        'conf/imagenet/8_768_300E_t4.yml',
        # 'validate_imagenet_01_2024112316.log'
        'validate_imagenet_02_2024112328.log'
    )
    calculator.calc_total_energy()

if __name__ == "__main__":
    main()