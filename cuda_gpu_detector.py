import subprocess
import re
import sys
import uuid
import time

def run_command(cmd, default="未知", timeout=5):
    """执行命令并返回输出，增强错误处理和超时机制"""
    try:
        if isinstance(cmd, str) and sys.platform != "win32":
            cmd = cmd.split()

        # 支持带空格的命令字符串
        if sys.platform == "win32" and isinstance(cmd, str):
            cmd = cmd.split(" ", 1) if " " in cmd else [cmd]

        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return default
    except Exception as e:
        return f"命令执行错误: {str(e)}"

def detect_gpu_info():
    """检测GPU信息和驱动支持的CUDA版本，改进版本识别"""
    gpu_count = 0
    driver_version = "未知"
    driver_cuda_ver = "未知"

    # 尝试多种命令获取GPU信息
    cmd_variants = [
        ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query"],
        ["nvidia-smi"],
        ["lspci"] if sys.platform != "win32" else None
    ]

    for cmd in cmd_variants:
        if not cmd:
            continue

        output = run_command(cmd)
        if not output or "未知" in output:
            continue

        # 优化版本提取逻辑
        cuda_match = re.search(r"CUDA\s*Version\s*:\s*(\d+\.\d+)", output)
        if not cuda_match:
            cuda_match = re.search(r"CUDA\s*Version\s*(\d+\.\d+)", output)

        if cuda_match:
            driver_cuda_ver = cuda_match.group(1)

        # 提取驱动版本
        driver_match = re.search(r"Driver\s*Version\s*:\s*(\d+\.\d+)", output)
        if not driver_match:
            driver_match = re.search(r"NVIDIA-SMI\s+(\d+\.\d+)", output)
        if driver_match:
            driver_version = driver_match.group(1)

        # GPU设备检测
        gpu_matches = re.findall(r"GPU \d+:|Attached GPUs\s*:\s*\d+", output)
        if gpu_matches:
            gpu_count = len(gpu_matches)

        if gpu_count > 0 and driver_cuda_ver != "未知":
            return True, gpu_count, driver_version, driver_cuda_ver

    # Windows注册表查询备用方案（仅用于CUDA版本）
    if sys.platform == "win32":
        try:
            reg_cmd = r'reg query "HKLM\SOFTWARE\NVIDIA Corporation\Global\NVSMI"'
            reg_output = run_command(reg_cmd)
            if reg_output and (match := re.search(r"CUDA_VERSION\s+REG_SZ\s+(\d+\.\d+)", reg_output)):
                driver_cuda_ver = match.group(1)
        except Exception:
            pass

    return gpu_count > 0, gpu_count, driver_version, driver_cuda_ver

def check_pytorch_installation():
    """检查PyTorch安装状态及CUDA支持"""
    pytorch_info = {
        'installed': False,
        'version': "未安装",
        'cuda_version': "N/A",
        'cuda_available': False,
        'current_device': "N/A",
        'gpu_count': 0,
        'device_names': []
    }

    try:
        import torch
        pytorch_info['installed'] = True
        pytorch_info['version'] = torch.__version__

        # 检查CUDA编译版本
        pytorch_info['cuda_version'] = getattr(torch.version, 'cuda', "N/A") or "N/A"

        # 检查CUDA是否可用
        pytorch_info['cuda_available'] = torch.cuda.is_available()

        # 获取GPU设备信息
        if pytorch_info['cuda_available']:
            pytorch_info['gpu_count'] = torch.cuda.device_count()
            pytorch_info['device_names'] = [
                torch.cuda.get_device_name(i) for i in range(pytorch_info['gpu_count'])
            ]
            pytorch_info['current_device'] = torch.cuda.current_device()

    except ImportError:
        pass
    except Exception as e:
        pytorch_info['error'] = f"PyTorch检测出错: {str(e)}"

    return pytorch_info

def run_gpu_performance_test():
    """执行GPU性能测试并返回结果"""
    result = {"status": "未执行", "metrics": {}}

    try:
        import torch
        # 确保CUDA可用
        if not torch.cuda.is_available():
            result["status"] = "测试失败: CUDA不可用"
            return result

        # 检查实际CUDA版本
        cuda_version = getattr(torch.version, 'cuda', "N/A")
        result["metrics"]["cuda_version"] = cuda_version

        # 获取当前设备
        device = torch.device('cuda')
        torch.cuda.synchronize()

        # 显存容量测试
        mem_info = torch.cuda.mem_get_info()
        total_mem = mem_info[1] / (1024**3)  # 转换为GB
        free_mem = mem_info[0] / (1024**3)
        result["metrics"]["total_vram_gb"] = total_mem
        result["metrics"]["free_vram_gb"] = free_mem

        # 数据传输速度测试（小数据量）
        small_data = torch.randn(1000, 1000)
        start_time = time.time()
        device_data = small_data.to(device)
        torch.cuda.synchronize()
        transfer_time = time.time() - start_time
        data_size = small_data.element_size() * small_data.nelement()
        result["metrics"]["small_transfer_speed_gb_s"] = data_size / (transfer_time * 1e9)

        # 计算性能测试（中等规模）
        size = 5000  # 减少矩阵大小以兼容不同显卡
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)

        # 预热
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # 正式测试
        start_time = time.time()
        for _ in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        compute_time = (time.time() - start_time) / 5

        flops = 2 * size**3 / compute_time * 1e-9  # GFLOP/s
        result["metrics"]["compute_performance_gflops"] = flops
        result["metrics"]["matrix_multiply_time_ms"] = compute_time * 1000

        result["status"] = "成功"

    except torch.cuda.OutOfMemoryError:
        result["status"] = "测试失败: 显存不足"
    except Exception as e:
        result["status"] = f"测试失败: {str(e)}"
        result["error"] = str(e)

    return result

def print_summary(session_id, gpu_info, pytorch_info, performance_result):
    """格式化输出检测结果"""
    print(f"\n{'='*60}")
    print(f"{'CUDA兼容性及性能检测报告':^60}")
    print(f"{'='*60}")
    print(f"诊断会话ID: {session_id}\n")

    # 硬件信息
    has_gpu, gpu_count, driver_version, driver_cuda = gpu_info
    if has_gpu:
        print(f"✅ 硬件检测: 发现 {gpu_count} 个GPU设备")
        print(f"  - 驱动版本: {driver_version}")
        print(f"  - 驱动支持的CUDA版本: {driver_cuda}\n")
    else:
        print("❌ 硬件检测: 未检测到兼容的GPU设备")
        if gpu_count > 0:
            print("⚠️ 检测到显卡但未安装NVIDIA驱动")
        print("建议: 安装NVIDIA显卡驱动: https://www.nvidia.com/Download/index.aspx")
        return

    # PyTorch信息
    if pytorch_info['installed']:
        print(f"✅ PyTorch已安装:")
        print(f"  - PyTorch版本: {pytorch_info['version']}")
        print(f"  - PyTorch编译时CUDA版本: {pytorch_info['cuda_version']}")

        if pytorch_info['cuda_available']:
            print(f"\n✅ CUDA可用: GPU数量 {pytorch_info['gpu_count']}")
            for i, name in enumerate(pytorch_info['device_names']):
                print(f"  - 设备{i}名称: {name}")
            print(f"  - 当前设备索引: {pytorch_info['current_device']}")
        else:
            print("\n❌ CUDA不可用")

        # 版本兼容性检查
        pytorch_cuda = pytorch_info['cuda_version']
        if pytorch_cuda != "N/A" and driver_cuda != "未知":
            driver_major = driver_cuda.split('.')[0] if driver_cuda else None
            pytorch_major = pytorch_cuda.split('.')[0] if pytorch_cuda and pytorch_cuda != "N/A" else "N/A"

            if driver_major and pytorch_major != "N/A" and driver_major == pytorch_major:
                print("\n✅ 版本兼容性: 驱动和PyTorch使用相同主版本CUDA")
            else:
                # print("\n⚠ 版本兼容性问题:")
                # print(f"  - 驱动版本: {driver_cuda}")
                # print(f"  - PyTorch编译版本: {pytorch_cuda}")
                # print(f"建议: 安装匹配的PyTorch版本 - 参考: https://pytorch.org/get-started/locally/")
                # print(f"查询CUDA版本命令: nvidia-smi")
                
                # 尝试获取系统CUDA版本
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, text=True)
                    output = result.stdout
                    import re
                    system_cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', output)
                    if system_cuda_match:
                        system_cuda_version = system_cuda_match.group(1)
                        
                        print("\n⚠ 版本兼容性提示:")
                        print(f"  - 系统CUDA版本: {system_cuda_version}")
                        print(f"  - PyTorch CUDA版本: {pytorch_cuda}")

                        # 比较版本并提供建议
                        try:
                            pytorch_cuda_major = float(pytorch_cuda.split('.')[0])
                            system_cuda_major = float(system_cuda_version.split('.')[0])
                            version_diff = system_cuda_major - pytorch_cuda_major
                            
                            if version_diff >= 5:
                                print("  建议: 虽然CUDA通常向下兼容，但版本差异较大。如果遇到性能或兼容性问题，")
                                print("        可以考虑更新PyTorch到支持更新CUDA版本的版本。")
                            elif version_diff <= 2:
                                print("  建议: CUDA通常向下兼容，且版本差异不大，一般可以兼容使用，如果遇到性能或兼容性问题，")
                        except:
                            pass  # 忽略版本比较错误

                        print("        参考: https://pytorch.org/get-started/locally/")
                        print("\n  查询CUDA版本命令: nvidia-smi")
                except:
                    pass  # 忽略nvidia-smi执行错误
        print("")
    else:
        print("\n❌ PyTorch未安装")
        print(f"建议安装命令: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{driver_cuda.replace('.', '')[:2]}")
        print("更多安装选项: https://pytorch.org/get-started/locally/")
        print("查询CUDA版本命令: nvidia-smi\n")
        return

    # 性能测试结果
    print("\n🔧 GPU性能测试:")
    if performance_result['status'] == "成功":
        metrics = performance_result['metrics']
        print(f"  - CUDA版本: {metrics['cuda_version']}")
        print(f"  - 数据传输速度: {metrics['small_transfer_speed_gb_s']:.2f} GB/s")
        print(f"  - 矩阵乘法时间({metrics.get('matrix_size', '?')}×{metrics.get('matrix_size', '?')}): {metrics['matrix_multiply_time_ms']:.2f} ms")
        print(f"  - 计算性能(GFLOPs): {metrics['compute_performance_gflops']:.2f}")
        print(f"  - 总显存: {metrics['total_vram_gb']:.1f} GB")
        print(f"  - 可用显存: {metrics['free_vram_gb']:.1f} GB\n")
        print("✅ 性能测试成功完成")
    else:
        print(f"❌ {performance_result['status']}")

def main():
    """主检测流程"""
    session_id = str(uuid.uuid4())

    # 1. 检测硬件信息
    gpu_info = detect_gpu_info()

    # 2. 检测PyTorch安装
    pytorch_info = check_pytorch_installation()

    # 3. 执行性能测试 (如有需要)
    performance_result = {"status": "未执行"}
    if pytorch_info.get('installed', False) and pytorch_info.get('cuda_available', False):
        print("\n运行GPU性能基准测试...")
        performance_result = run_gpu_performance_test()

    # 4. 输出结果摘要
    print_summary(session_id, gpu_info, pytorch_info, performance_result)

if __name__ == "__main__":
    print("正在检测您的系统环境...")
    main()