import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import time
import numpy as np
import os

# 设置随机种子确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(device, use_amp=False, batch_size=256, epochs=5):
    """训练模型并返回每轮平均训练时间"""
    # 创建随机数据模拟训练过程
    train_data = torch.randn(batch_size * 10, 3, 32, 32)
    train_labels = torch.randint(0, 10, (batch_size * 10,))
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 使用新的API格式
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    scaler = GradScaler(device_type, enabled=use_amp)

    times = []
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 混合精度训练
            device_type = 'cuda' if device.type == 'cuda' else 'cpu'
            with autocast(device_type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        epoch_time = time.time() - start_time
        times.append(epoch_time)
        print(f"设备: {device}, 混合精度: {use_amp}, 轮次 {epoch + 1}/{epochs}, 耗时: {epoch_time:.2f}秒")

    return np.mean(times)


def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"检测到GPU: {gpu_name}")
        print(f"GPU内存: {gpu_memory:.2f} GB")
        
        # 使用getattr来避免类型检查错误
        cuda_version = getattr(torch.version, 'cuda', 'Unknown')
        print(f"PyTorch CUDA版本: {cuda_version}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        
        # 检查系统CUDA版本（可选）
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
            output = result.stdout
            import re
            system_cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', output)
            if system_cuda_match:
                system_cuda_version = system_cuda_match.group(1)
                print(f"系统CUDA版本: {system_cuda_version}")
                
                # 比较版本并提供建议
                try:
                    pytorch_cuda_major = float(cuda_version.split('.')[0])
                    system_cuda_major = float(system_cuda_version.split('.')[0])
                    version_diff = system_cuda_major - pytorch_cuda_major
                    
                    if version_diff >= 2:
                        print("\n⚠ 版本兼容性提示:")
                        print("  - 系统CUDA版本: " + system_cuda_version)
                        print("  - PyTorch CUDA版本: " + cuda_version)
                        print("  建议: 虽然CUDA通常向下兼容，但版本差异较大。如果遇到性能或兼容性问题，")
                        print("        可以考虑更新PyTorch到支持更新CUDA版本的版本。")
                        print("        参考: https://pytorch.org/get-started/locally/")
                        print("\n  查询CUDA版本命令: nvidia-smi")
                except:
                    pass  # 忽略版本比较错误
        except:
            pass  # 忽略nvidia-smi执行错误
    else:
        print("未检测到GPU，将只在CPU上运行测试")


def main():
    print("===== PyTorch RTX 4070 性能测试 =====")
    print_gpu_info()

    # 确保GPU预热
    if torch.cuda.is_available():
        print("\n=== GPU预热 ===")
        _ = train_model(torch.device("cuda"), use_amp=False, batch_size=32, epochs=1)
        torch.cuda.empty_cache()

    # 测试不同配置
    results = {}

    print("\n=== CPU 基准测试 ===")
    cpu_time = train_model(torch.device("cpu"), use_amp=False, batch_size=32, epochs=3)
    results["CPU"] = cpu_time

    if torch.cuda.is_available():
        print("\n=== GPU FP32 测试 ===")
        gpu_fp32_time = train_model(torch.device("cuda"), use_amp=False, epochs=3)
        results["GPU FP32"] = gpu_fp32_time

        print("\n=== GPU FP16 (混合精度) 测试 ===")
        gpu_fp16_time = train_model(torch.device("cuda"), use_amp=True, epochs=3)
        results["GPU FP16"] = gpu_fp16_time

        # 计算加速比
        fp32_speedup = cpu_time / gpu_fp32_time
        fp16_speedup = cpu_time / gpu_fp16_time
        mixed_vs_full = gpu_fp32_time / gpu_fp16_time

        print("\n===== 性能对比 =====")
        print(f"CPU 平均每轮耗时: {cpu_time:.2f}秒")
        print(f"GPU FP32 平均每轮耗时: {gpu_fp32_time:.2f}秒 ({fp32_speedup:.2f}x CPU速度)")
        print(
            f"GPU FP16 平均每轮耗时: {gpu_fp16_time:.2f}秒 ({fp16_speedup:.2f}x CPU速度, {mixed_vs_full:.2f}x FP32速度)")

        # 打印GPU利用率
        if os.name == 'nt':  # Windows系统
            print("\n提示: 您可以通过任务管理器查看GPU利用率和显存使用情况")
        elif os.name == 'posix':  # Linux系统
            print("\n提示: 您可以使用以下命令监控GPU:")
            print("  - nvidia-smi (实时GPU状态)")
            print("  - watch -n 1 nvidia-smi (每秒更新一次)")
    else:
        print("\n===== 性能结果 =====")
        print(f"CPU 平均每轮耗时: {cpu_time:.2f}秒")
        print("未检测到GPU，无法进行GPU性能对比")


if __name__ == "__main__":
    main()