# AI视频翻译服务

基于Whisper + NLLB的高质量视频翻译解决方案，支持批量处理和可选的DeepSeek AI润色。

## 特性

✅ **高速翻译** - 30分钟视频仅需2分钟  
✅ **高质量** - 翻译质量接近/超越Google Translate  
✅ **批量处理** - 支持目录批量翻译，自动断点续传  
✅ **GPU加速** - 支持CUDA GPU加速  
✅ **可选润色** - 集成DeepSeek API进行专业级翻译润色  
✅ **灵活配置** - 支持多种模型和参数配置  

## 性能数据

| 配置 | 30分钟视频耗时 | 质量 | 成本 |
|------|--------------|------|------|
| medium + nllb-1.3B | 115秒 | ⭐⭐⭐⭐ | 免费 |
| + DeepSeek润色 | 140秒 | ⭐⭐⭐⭐⭐ | 0.001元/视频 |

## 快速开始

### 1. 环境要求

- Python 3.11+
- CUDA 12.1+ (可选，GPU加速)
- ffmpeg

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

首次运行会自动下载模型（约2-3GB），或手动下载：

```bash
# 下载ASR模型（Whisper）
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', download_root='./models/whisper')"

# 下载翻译模型（NLLB）
python -c "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer; AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-1.3B', cache_dir='./models/nllb')"
```

### 4. 配置

复制配置模板并修改：

```bash
cp config.template.ini config.ini
```

编辑 `config.ini` 填入你的配置（如DeepSeek API密钥）。

### 5. 启动服务

```bash
# Windows
start_service.bat

# 或手动启动
python server.py
```

### 6. 翻译视频

```bash
# 翻译单个视频
python batch_translate.py video.mp4 -t zh

# 批量翻译目录
python batch_translate.py videos/ -t zh --recursive -o subtitles/

# 使用DeepSeek润色
python batch_translate.py videos/ -t zh --polish
```

## 项目结构

```
local-python-service/
├── server.py                    # 翻译服务（标准版）
├── server_with_deepseek.py     # 翻译服务（集成DeepSeek）
├── batch_translate.py          # 批量翻译工具
├── config.template.ini         # 配置模板
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
├── .gitignore                  # Git忽略文件
└── start_service.bat           # Windows启动脚本
```

## 使用说明

### 基本翻译

```bash
# 翻译单个视频
python batch_translate.py movie.mp4 -t zh

# 翻译目录中所有视频
python batch_translate.py D:\Videos\ -t zh --recursive

# 指定输出目录
python batch_translate.py videos/ -t zh -o subtitles/
```

### DeepSeek润色（可选）

需要申请DeepSeek API密钥：

1. 访问 https://platform.deepseek.com 注册
2. 充值（建议10元）
3. 获取API密钥
4. 在 `config.ini` 中填入密钥
5. 使用 `--polish` 参数

```bash
python batch_translate.py videos/ -t zh --polish
```

### 命令行参数

```
usage: batch_translate.py [-h] [-t TARGET] [-s SOURCE] [-o OUTPUT] 
                          [--translation-only] [-r] [--polish]
                          input

参数:
  input                视频文件或目录路径
  -t, --target         目标语言（默认: zh）
  -s, --source         源语言（默认: auto）
  -o, --output         输出目录
  -r, --recursive      递归处理子目录
  --translation-only   仅生成译文字幕
  --polish             使用DeepSeek润色
```

## 配置说明

编辑 `config.ini` 修改配置：

```ini
[API]
deepseek_api_key = your_api_key_here

[Models]
asr_model_size = medium              # ASR模型: tiny, base, small, medium, large-v3
translation_model = facebook/nllb-200-distilled-1.3B
use_gpu = true
beam_size = 3

[Translation]
default_target_language = zh
use_deepseek_polish = false
```

## 模型选择

### ASR模型（Whisper）

| 模型 | 大小 | 速度 | 质量 | 推荐场景 |
|------|------|------|------|---------|
| tiny | 39M | 最快 | 一般 | 测试 |
| base | 74M | 很快 | 可用 | 快速处理 |
| small | 244M | 快 | 良好 | 一般使用 |
| medium | 769M | 中等 | 优秀 | **推荐** |
| large-v3 | 1.5B | 慢 | 最好 | 高质量需求 |

### 翻译模型

| 模型 | 大小 | 速度 | 质量 | 推荐 |
|------|------|------|------|------|
| nllb-200-distilled-600M | 600M | 快 | 良好 | 快速 |
| nllb-200-distilled-1.3B | 1.3B | 中等 | 高 | **推荐** |
| m2m100_418M | 418M | 很快 | 可用 | 极速 |

## 支持的语言

中文(zh), 英语(en), 日语(ja), 韩语(ko), 德语(de), 法语(fr), 
西班牙语(es), 俄语(ru), 阿拉伯语(ar) 等200+语言

## 常见问题

### Q: 找不到ffmpeg
A: 下载ffmpeg并添加到PATH，或将ffmpeg.exe放在项目目录

### Q: CUDA错误
A: 确保安装了CUDA 12.1+和对应的PyTorch CUDA版本

### Q: 翻译很慢
A: 确认GPU加速是否启用，或使用更小的模型

### Q: DeepSeek API错误
A: 检查API密钥、网络连接和账户余额

## 性能优化

1. **使用GPU** - 速度提升5-10倍
2. **选择合适的模型** - medium ASR + nllb-1.3B 是最佳平衡
3. **调整beam_size** - 降低可提升速度（3已是优化值）
4. **批量处理** - 减少模型加载开销

## 开发

### 不集成DeepSeek
使用 `server.py`，纯本地翻译，完全免费。

### 集成DeepSeek
使用 `server_with_deepseek.py`，获得专业级翻译质量，需要API密钥。

## 许可证

MIT License

## 致谢

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - ASR引擎
- [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - 翻译模型
- [DeepSeek](https://www.deepseek.com/) - AI润色服务

## 更新日志

### v1.0.0 (2025-11)
- 初始版本
- 支持批量视频翻译
- 集成DeepSeek润色
- GPU加速支持