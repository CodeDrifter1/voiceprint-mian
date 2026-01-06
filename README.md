# 🏥 声纹识别医疗服务

基于声纹识别的智能医疗服务系统，支持医患对话识别和电子病历自动生成。

> 本项目基于 [xinnan-tech/voiceprint-api](https://github.com/xinnan-tech/voiceprint-api) 进行二次开发

## ✨ 新增功能

### 🩺 电子病历生成
- 基于 Ollama + Qwen2.5:14b 大模型自动分析医患对话
- 生成结构化电子病历（主诉、现病史、诊断、治疗方案等）
- **严格提取模式**：只提取对话中明确提到的内容，不进行AI推断

### 📚 病历管理
- 病历历史记录查看
- 在线编辑修改病历内容
- 一键删除病历

### 📄 PDF导出
- 专业医疗风格排版
- 中文字体支持
- 文件名格式：`姓名拼音_jiuzhen_时间.pdf`

### 🎙️ 对话记录优化
- 医生显示声纹注册的ID（如"王医生"）
- 患者显示填写的姓名（如"张三"）

### 🎨 前端界面重构
- 医疗专业蓝白配色
- 响应式两栏布局
- 顶部状态栏显示服务连接状态
- 弹窗式病历查看和编辑

### ⚙️ 其他改进
- 启动脚本自动杀死端口占用
- 音频文件保存（姓名拼音_时间.wav）
- VAD语音活动检测，防止空音频匹配
- 隐藏"未提及"的空字段

---

## 🏗️ 技术架构

| 组件 | 技术 | 说明 |
|------|------|------|
| 说话人分离 | Pyannote 3.1 | 区分不同说话人 |
| 声纹识别 | 3DSpeaker CAM++ | 提取声纹特征 |
| 语音转文字 | SenseVoice | ASR转写 |
| 病历生成 | Ollama + Qwen2.5 | 大模型分析 |
| PDF导出 | ReportLab | 生成PDF文档 |
| 后端框架 | FastAPI | REST API |
| 数据库 | MySQL | 声纹存储 |

## 📁 项目结构

```
voiceprint-main/
├── app/
│   ├── api/v1/
│   │   ├── medical.py          # 电子病历接口（新增）
│   │   └── voiceprint.py       # 声纹识别接口
│   ├── services/
│   │   └── voiceprint_service.py
│   └── core/
├── data/
│   ├── .voiceprint.yaml.example  # 配置示例
│   ├── recordings/               # 录音存储（新增）
│   └── medical_records/          # 病历存储（新增）
├── voiceprint_test.html          # 前端页面（重构）
├── start.sh                      # 启动脚本
├── start_server.py               # Python启动入口
└── README.md
```

## 🔌 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| 声纹识别API | 8520 | 主服务 |
| SenseVoice ASR | 8001 | 语音转文字服务 |
| Ollama | 11434 | 大模型服务 |

---

# 📦 部署指南

## 一、环境准备

### 1.1 创建 Conda 环境
```bash
conda create -n voiceprint python=3.10 -y
conda activate voiceprint
```

### 1.2 克隆项目
```bash
git clone https://github.com/CodeDrifter1/voiceprint-main.git
cd voiceprint-main
```

### 1.3 安装依赖
```bash
pip install -r requirements.txt
```

---

## 二、Pyannote 3.1 说话人分离模型下载（重要）

Pyannote 模型托管在 Hugging Face，需要登录并同意使用条款才能下载。

### 2.1 注册 Hugging Face 账号
1. 访问 https://huggingface.co/join 注册账号
2. 访问 https://huggingface.co/settings/tokens 创建 Access Token
3. 选择 `Read` 权限，复制生成的 Token

### 2.2 同意模型使用条款（必须）
访问以下页面，点击 **"Agree and access repository"** 同意条款：
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM

**注意**：必须用同一个账号同意所有三个模型的条款，否则下载会报 401 错误。

### 2.3 配置代理（国内需要）
```bash
# 设置代理，替换为你的代理地址
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
```

### 2.4 登录 Hugging Face
```bash
pip install huggingface_hub
huggingface-cli login
```
输入你创建的 Token。

### 2.5 下载 Pyannote 模型
```bash
python -c "
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
print('下载成功！')
"
```

如果报错，尝试指定 token：
```bash
python -c "
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='你的HuggingFace_Token'
)
print('下载成功！')
"
```

### 2.6 验证模型下载
```bash
ls ~/.cache/huggingface/hub/ | grep pyannote
```
应该看到：
```
models--pyannote--segmentation-3.0
models--pyannote--speaker-diarization-3.1
models--pyannote--wespeaker-voxceleb-resnet34-LM
```

### 2.7 下载完成后关闭代理
```bash
unset https_proxy
unset http_proxy
```

---

## 三、3DSpeaker 声纹识别模型

3DSpeaker 模型托管在 ModelScope，国内可直接下载，无需代理。

首次启动服务时会自动下载到 `~/.cache/modelscope/hub/`

手动下载（可选）：
```bash
python -c "
from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_campplus_sv_zh-cn_3dspeaker_16k'
)
print('下载成功！')
"
```

---

## 四、Ollama 大模型安装（电子病历功能）

### 4.1 安装 Ollama
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 4.2 下载 Qwen2.5:14b 模型
```bash
ollama pull qwen2.5:14b
```
模型约 9GB，下载需要一些时间。

### 4.3 启动 Ollama 服务
```bash
# 后台运行
nohup ollama serve > /dev/null 2>&1 &
```

### 4.4 验证 Ollama
```bash
curl http://localhost:11434/api/tags
```

---

## 五、数据库配置

### 5.1 创建数据库和用户
```sql
-- 登录 MySQL
mysql -u root -p

-- 创建数据库
CREATE DATABASE voiceprint CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 创建用户
CREATE USER 'voiceprint'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON voiceprint.* TO 'voiceprint'@'%';
FLUSH PRIVILEGES;

-- 创建声纹表
USE voiceprint;
CREATE TABLE voiceprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    speaker_id VARCHAR(255) UNIQUE NOT NULL,
    feature_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 六、配置文件

### 6.1 创建配置文件
```bash
cp data/.voiceprint.yaml.example data/.voiceprint.yaml
```

### 6.2 编辑配置
```yaml
mysql:
  database: voiceprint
  host: 127.0.0.1
  password: your_password    # 修改为你的数据库密码
  port: 3306
  user: voiceprint

server:
  authorization: your_api_key  # 修改为你的API密钥（可用 uuid 生成）
  host: 0.0.0.0
  port: 8520

voiceprint:
  similarity_threshold: 0.4    # 声纹相似度阈值
  target_sample_rate: 16000
  tmp_dir: tmp
```

---

## 七、安装中文字体（PDF导出）

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install fonts-wqy-zenhei
```

### CentOS/RHEL
```bash
sudo yum install wqy-zenhei-fonts
```

---

## 八、启动服务

```bash
# 方式1：使用启动脚本（推荐）
chmod +x start.sh
./start.sh

# 方式2：手动启动
conda activate voiceprint
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python start_server.py
```

**注意**：首次使用前需修改 `start.sh` 中的项目路径：
```bash
cd ~/voiceprint/voiceprint-api  # 改为你的实际路径
```

### 启动成功日志
```
INFO - 日志系统初始化完成
INFO - 声纹接口地址: http://192.168.0.207:8520/voiceprint/health
INFO - 数据库连接成功
INFO - 初始化声纹识别模型
INFO - 使用GPU设备: NVIDIA GeForce RTX 4090 D
INFO - 初始化 Pyannote 3.1 说话人分离模型
INFO - Pyannote 3.1 使用 GPU 加速
INFO - 模型预热完成
INFO - Uvicorn running on http://0.0.0.0:8520
```

---

## 九、访问前端

浏览器打开 `voiceprint_test.html`，修改页面中的 API 地址为服务器 IP：
```
http://192.168.0.207:8520
```

---

# 📚 API 接口

## 声纹管理
| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/voiceprint/register` | 注册声纹 |
| POST | `/voiceprint/identify` | 识别声纹 |
| POST | `/voiceprint/conversation` | 多人对话识别 |
| GET | `/voiceprint/list` | 获取所有声纹 |
| DELETE | `/voiceprint/{speaker_id}` | 删除声纹 |
| GET | `/voiceprint/health` | 健康检查 |

## 电子病历
| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/medical/medical-record` | 生成电子病历 |
| GET | `/medical/medical-records` | 获取病历列表 |
| GET | `/medical/medical-records/{id}` | 获取病历详情 |
| PUT | `/medical/medical-records/{id}` | 更新病历 |
| DELETE | `/medical/medical-records/{id}` | 删除病历 |
| GET | `/medical/medical-records/{id}/pdf` | 导出PDF |

---

# ❓ 常见问题及解决方案

### 问题1: Pyannote 下载报 401 错误
```
401 Client Error: Unauthorized
```
**解决**: 确认已访问三个模型页面并点击 "Agree"，重新生成 Token 并登录

### 问题2: Pyannote 下载超时
```
ConnectionError: HTTPSConnectionPool
```
**解决**: 配置代理后重试

### 问题3: pyarrow 版本不兼容
```
AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'
```
**解决**: `pip install "datasets>=2.18.0"`

### 问题4: NumPy 版本冲突
```
np.NaN was removed in NumPy 2.0
```
**解决**: `pip install torch==2.8.0 torchaudio==2.8.0`

### 问题5: Pyannote 缺少依赖
```
requires the hdbscan library
```
**解决**: `pip install hdbscan umap-learn`

### 问题6: 数据库权限错误
```
Access denied for user 'voiceprint'@'%'
```
**解决**: 
```sql
GRANT ALL PRIVILEGES ON voiceprint.* TO 'voiceprint'@'%';
FLUSH PRIVILEGES;
```

### 问题7: Pyannote 启动时联网失败
```
HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded
```
**解决**: 
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 问题8: PDF 中文显示为方块
**解决**: 安装 wqy-zenhei 字体
```bash
sudo apt-get install fonts-wqy-zenhei
```

### 问题9: Ollama 连接失败
**解决**: 
```bash
ollama serve  # 启动 Ollama
```

### 问题10: 端口被占用
**解决**: `start_server.py` 已自动处理，会先杀死占用端口的进程

### 问题11: use_auth_token 参数错误
```
got an unexpected keyword argument 'use_auth_token'
```
**解决**: Pyannote 4.0+ 使用 `token` 参数替代 `use_auth_token`

---

# 🔒 安全提示

- `data/.voiceprint.yaml` 包含敏感信息，已加入 `.gitignore`
- 请勿将密码和 API 密钥提交到仓库

---

# � 致License

MIT License

---

# 🙏 致谢

- 原项目：[xinnan-tech/voiceprint-api](https://github.com/xinnan-tech/voiceprint-api)
- 声纹模型：[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)
- 说话人分离：[Pyannote](https://github.com/pyannote/pyannote-audio)
