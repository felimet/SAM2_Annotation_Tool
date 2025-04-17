# SAM2 標註工具安裝指南

此文件將指導您如何安裝和設置 SAM2 標註工具。

## 系統需求

- **操作系統**：Windows 10/11（64位元）
- **硬體**：
  - CPU：建議 Intel i5 或 AMD Ryzen 5 以上
  - RAM：最低 8GB，建議 16GB 以上
  - GPU：NVIDIA 顯示卡（建議 6GB 以上顯存，支援 CUDA）
  - 儲存空間：至少 10GB 可用空間

## 安裝方式

### 方式一：直接使用封裝版本

1. 下載最新的封裝版本 `SAM2_Annotation_Tool.zip`
2. 解壓縮到您喜歡的位置
3. 運行解壓縮後資料夾中的 `SAM2_Annotation_Tool.exe`

### 方式二：從源碼封裝（適合開發者）

#### 1. 安裝 Python 環境

1. 下載並安裝 [Python 3.9+](https://www.python.org/downloads/)
2. 確保安裝時勾選「Add Python to PATH」選項

#### 2. 安裝所需套件

```bash
# 安裝基本依賴
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy pillow pyinstaller tqdm

# 克隆 SAM2 儲存庫（如果需要）
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

#### 3. 下載模型權重

預訓練權重可從以下途徑獲取：
- 官方 HuggingFace 儲存庫：[https://huggingface.co/facebook/sam2](https://huggingface.co/facebook/sam2)
- 或使用提供的下載腳本：

```bash
cd checkpoints
./download_ckpts.sh
```

#### 4. 執行封裝腳本

```bash
python setup.py
```

完成後，可執行檔將位於 `dist/SAM2_Annotation_Tool` 目錄下。

## 常見問題

### 找不到 CUDA

確保已安裝支援的 NVIDIA 顯卡驅動和 CUDA Toolkit。可以通過以下命令確認 PyTorch 是否可以使用 CUDA：

```python
import torch
print(torch.cuda.is_available())
```

### 記憶體不足錯誤

可能原因：
1. 模型需要較多 GPU 記憶體
2. 處理高解析度影像

解決方式：
- 使用更小的模型（例如 sam2.1_hiera_tiny.pt 或 sam2.1_hiera_small.pt）
- 減小處理的影像尺寸
- 增加系統 RAM 或 GPU 記憶體

## 技術支援

如遇到任何問題，請在 GitHub 上提交 issue 或聯絡開發團隊。