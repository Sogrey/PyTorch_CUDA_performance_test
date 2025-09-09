# PyTorch GPU 性能测试工具

这个项目提供了两个用于测试和分析 NVIDIA GPU 在 PyTorch 环境下性能的工具。

## 功能特点

### 1. PyTorch GPU 性能测试 (`PyTorch_GPU_performance_test.py`)

这个脚本可以：
- 检测 GPU 硬件信息（名称、内存、CUDA版本）
- 对比 CPU 和 GPU 在不同精度下的性能差异
- 测试 FP32（单精度）和 FP16（混合精度）的计算速度
- 提供详细的性能对比报告

### 2. CUDA GPU 检测器 (`cuda_gpu_detector.py`)

这个脚本提供更全面的系统诊断：
- 检测 NVIDIA 驱动和支持的 CUDA 版本
- 验证 PyTorch 安装状态和 CUDA 支持
- 执行 GPU 性能基准测试
- 检查版本兼容性并提供建议
- 生成详细的诊断报告

## 系统要求

- Python 3.6+
- PyTorch 1.7+
- NVIDIA GPU 和驱动
- CUDA 工具包（与 PyTorch 兼容的版本）

## 安装依赖

```bash
pip install torch torchvision torchaudio
```

## 使用方法

### 运行性能测试

```bash
python PyTorch_GPU_performance_test.py
```

这将执行一系列测试并显示 CPU 与 GPU 在不同精度下的性能对比。

### 运行系统诊断

```bash
python cuda_gpu_detector.py
```

这将生成一份详细的系统诊断报告，包括硬件检测、软件兼容性和性能测试结果。

## 输出示例

### PyTorch GPU 性能测试输出

```
===== PyTorch RTX 4070 性能测试 =====
检测到GPU: NVIDIA GeForce RTX 4060 Ti
GPU内存: 8.00 GB
CUDA版本: 12.9
cuDNN版本: 91002

=== GPU预热 ===
设备: cuda, 混合精度: False, 轮次 1/1, 耗时: 0.22秒

=== CPU 基准测试 ===
设备: cpu, 混合精度: False, 轮次 1/3, 耗时: 0.21秒
设备: cpu, 混合精度: False, 轮次 2/3, 耗时: 0.19秒
设备: cpu, 混合精度: False, 轮次 3/3, 耗时: 0.18秒

=== GPU FP32 测试 ===
设备: cuda, 混合精度: False, 轮次 1/3, 耗时: 0.12秒
设备: cuda, 混合精度: False, 轮次 2/3, 耗时: 0.12秒
设备: cuda, 混合精度: False, 轮次 3/3, 耗时: 0.14秒

=== GPU FP16 (混合精度) 测试 ===
设备: cuda, 混合精度: True, 轮次 1/3, 耗时: 0.18秒
设备: cuda, 混合精度: True, 轮次 2/3, 耗时: 0.08秒
设备: cuda, 混合精度: True, 轮次 3/3, 耗时: 0.08秒

===== 性能对比 =====
CPU 平均每轮耗时: 0.19秒
GPU FP32 平均每轮耗时: 0.13秒 (1.49x CPU速度)
GPU FP16 平均每轮耗时: 0.11秒 (1.68x CPU速度, 1.13x FP32速度)
```

### CUDA GPU 检测器输出

```
============================================================
                       CUDA兼容性及性能检测报告
============================================================
诊断会话ID: 9dbdffb7-a346-40b5-a113-d0409555269a

✅ 硬件检测: 发现 2 个GPU设备
  - 驱动版本: 581.15
  - 驱动支持的CUDA版本: 13.0

✅ PyTorch已安装:
  - PyTorch版本: 2.8.0+cu129
  - PyTorch编译时CUDA版本: 12.9

✅ CUDA可用: GPU数量 1
  - 设备0名称: NVIDIA GeForce RTX 4060 Ti
  - 当前设备索引: 0

⚠ 版本兼容性问题:
  - 驱动版本: 13.0
  - PyTorch编译版本: 12.9
  建议: 安装匹配的PyTorch版本 - 参考: https://pytorch.org/get-started/locally/
  查询CUDA版本命令: nvidia-smi

🔧 GPU性能测试:
  - CUDA版本: 12.9
  - 数据传输速度: 2.00 GB/s
  - 矩阵乘法时间(?×?): 20.87 ms
  - 计算性能(GFLOPs): 11980.66
  - 总显存: 8.0 GB
  - 可用显存: 6.9 GB

✅ 性能测试成功完成
```

## 常见问题解答

### Q: 如何查看我的系统CUDA版本？
A: 在命令行中运行 `nvidia-smi` 命令可以查看当前安装的驱动和支持的CUDA版本。

### Q: 为什么PyTorch的CUDA版本与系统CUDA版本不同？
A: PyTorch通常使用它编译时的CUDA版本，而不是系统安装的CUDA版本。CUDA通常向下兼容，所以只要系统CUDA版本不低于PyTorch需要的版本，一般不会有问题。

### Q: 如何安装特定CUDA版本的PyTorch？
A: 访问 [PyTorch官方网站](https://pytorch.org/get-started/locally/) 选择适合您系统的安装命令。

### Q: 混合精度训练是什么？
A: 混合精度训练使用FP16（半精度）和FP32（单精度）混合计算，可以在保持模型精度的同时提高训练速度和减少显存使用。

## 故障排除

如果遇到问题，请检查：

1. 确保已安装最新的NVIDIA驱动
2. 确认PyTorch版本与CUDA版本兼容
3. 检查GPU是否被其他程序占用
4. 确保系统有足够的显存可用

## 许可证

MIT

## 作者

[您的名字]

## 致谢

感谢NVIDIA和PyTorch团队提供的优秀工具和文档。