# RL-Stock 修复版安装指南

## 环境要求

- Python: 3.6 - 3.7 (推荐3.7)
- 操作系统: Windows / Linux / macOS
- 内存: 至少4GB
- 磁盘空间: 至少2GB

## 快速安装

### 1. 克隆或下载项目

```bash
cd D:\PycharmProjects\RL-Stock
```

### 2. 创建虚拟环境 (推荐)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装CPU版本 (基础版)

```bash
pip install -r requirements.txt
```

### 4. (可选) 安装GPU版本

**前置条件:**
- NVIDIA GPU (支持CUDA)
- CUDA 10.0
- cuDNN 7.6

**安装步骤:**

#### 4.1 安装CUDA 10.0

**Windows:**
1. 下载: https://developer.nvidia.com/cuda-10.0-download-archive
2. 选择: Windows -> x86_64 -> 10 -> exe (local)
3. 运行安装程序,选择"精简"安装
4. 验证安装:
```bash
nvcc --version
```

**Linux:**
```bash
# Ubuntu 18.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-0-local-10.0.130-410.48/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-10-0
```

#### 4.2 安装cuDNN 7.6

1. 注册NVIDIA开发者账号: https://developer.nvidia.com/
2. 下载cuDNN 7.6 for CUDA 10.0: https://developer.nvidia.com/cudnn
3. 解压并复制文件:

**Windows:**
```bash
# 将解压的文件复制到CUDA安装目录
# 例如: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\
copy <cudnn_path>\bin\cudnn64_7.dll <cuda_path>\bin\
copy <cudnn_path>\include\cudnn.h <cuda_path>\include\
copy <cudnn_path>\lib\x64\cudnn.lib <cuda_path>\lib\x64\
```

**Linux:**
```bash
tar -xzvf cudnn-10.0-linux-x64-v7.6.5.32.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-10.0/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64
sudo chmod a+r /usr/local/cuda-10.0/include/cudnn*.h /usr/local/cuda-10.0/lib64/libcudnn*
```

#### 4.3 安装TensorFlow GPU版本

```bash
pip install tensorflow-gpu==1.14.0
```

#### 4.4 验证GPU安装

```bash
python -c "import tensorflow as tf; print('GPU Available:', tf.test.is_gpu_available())"
```

应该输出: `GPU Available: True`

## 验证安装

运行测试脚本验证所有修复:

```bash
python test_fixes.py
```

期望输出:
```
RL-STOCK FIX VERIFICATION TEST SUITE
============================================================
Test 1: Division by Zero Fix
✓ All tests passed

Test 2: Transaction Cost Calculation
✓ Transaction costs working correctly

Test 3: Risk Control Mechanisms
✓ Risk controls working

Test 4: GPU Support Check
✓ GPU support available (或 No GPU detected - will use CPU)

TEST SUMMARY
============================================================
Division By Zero: ✅ PASSED
Transaction Cost: ✅ PASSED
Risk Control: ✅ PASSED
Gpu Support: ✅ PASSED

🎉 All tests passed! All fixes verified successfully!
```

## 运行训练

### 快速测试 (1万步, 约2分钟)

```bash
python main_fixed.py
```

修改 `main_fixed.py` 底部:
```python
test_a_stock_improved(
    stock_code='sh.000001',
    total_timesteps=10000,    # 改为1万步
    use_gpu=True
)
```

### 标准训练 (10万步, CPU约20分钟, GPU约5分钟)

保持默认配置:
```python
test_a_stock_improved(
    stock_code='sh.000001',
    total_timesteps=100000,   # 10万步
    use_gpu=True
)
```

### 深度训练 (50万步, CPU约2小时, GPU约30分钟)

```python
test_a_stock_improved(
    stock_code='sh.000001',
    total_timesteps=500000,   # 50万步
    use_gpu=True
)
```

## 常见问题

### 1. ImportError: No module named 'tensorflow'

**解决:**
```bash
pip install tensorflow==1.14.0  # CPU版本
# 或
pip install tensorflow-gpu==1.14.0  # GPU版本
```

### 2. CUDA版本不匹配

**错误:** `libcublas.so.10.0: cannot open shared object file`

**解决:** 确保安装CUDA 10.0,不是其他版本

### 3. GPU内存不足

**错误:** `ResourceExhaustedError: OOM when allocating tensor`

**解决1 - 减少批次大小:**
在 `main_fixed.py` 中修改:
```python
model = PPO2(
    MlpPolicy, 
    env,
    n_steps=1024,      # 原2048,减半
    nminibatches=16,   # 原32,减半
    ...
)
```

**解决2 - 限制GPU内存:**
```python
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
```

### 4. 找不到股票数据

**错误:** `Stock file not found for code: sh.000001`

**解决:** 确保数据文件在正确位置:
```
stockdata/
├── train/
│   └── sh.000001.csv
└── test/
    └── sh.000001.csv
```

### 5. ModuleNotFoundError: No module named 'stable_baselines'

**解决:**
```bash
pip install stable-baselines==2.10.0
```

## 性能优化建议

### CPU优化

1. **使用多核:**
```python
import os
os.environ['OMP_NUM_THREADS'] = '4'  # 使用4核
```

2. **减少日志输出:**
```python
model = PPO2(MlpPolicy, env, verbose=0)  # 不输出训练日志
```

### GPU优化

1. **使用更大批次:**
```python
model = PPO2(
    MlpPolicy, env,
    n_steps=4096,      # 增加到4096
    nminibatches=64,   # 增加到64
)
```

2. **混合精度训练 (需要Volta架构或更新):**
```python
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

## 下一步

1. 阅读 `FIX_REPORT.md` 了解所有修复细节
2. 查看 `main_fixed.py` 了解训练配置
3. 运行 `test_fixes.py` 验证环境
4. 开始训练!

## 支持

如遇到问题:
1. 检查Python版本 (3.6-3.7)
2. 确认所有依赖正确安装
3. 验证数据文件存在
4. 查看详细错误信息

---

**更新时间:** 2025-01-21  
**版本:** v2.0
