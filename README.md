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

## 📦 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/CodeDrifter1/voiceprint-main.git
cd voiceprint-main
```

### 2. 创建环境
```bash
conda create -n voiceprint python=3.10 -y
conda activate voiceprint
pip install -r requirements.txt
```

### 3. 安装额外依赖
```bash
pip install "datasets>=2.18.0"
pip install pyannote.audio hdbscan umap-learn
pip install pypinyin reportlab
```

### 4. 配置
```bash
# 复制配置文件
cp data/.voiceprint.yaml.example data/.voiceprint.yaml

# 编辑配置，填入数据库密码和API密钥
```

### 5. 初始化数据库
```sql
CREATE DATABASE voiceprint CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'voiceprint'@'%' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON voiceprint.* TO 'voiceprint'@'%';

USE voiceprint;
CREATE TABLE voiceprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    speaker_id VARCHAR(255) UNIQUE NOT NULL,
    feature_vector BLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6. 安装 Ollama（电子病历功能）
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:14b
```

### 7. 启动服务
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python start_server.py
```

### 8. 访问前端
浏览器打开 `voiceprint_test.html`，修改 API 地址为服务器 IP。

## 📚 API 接口

### 声纹管理
| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/voiceprint/register` | 注册声纹 |
| POST | `/voiceprint/identify` | 识别声纹 |
| POST | `/voiceprint/conversation` | 多人对话识别 |
| GET | `/voiceprint/list` | 获取所有声纹 |
| DELETE | `/voiceprint/{speaker_id}` | 删除声纹 |

### 电子病历（新增）
| 方法 | 接口 | 说明 |
|------|------|------|
| POST | `/medical/medical-record` | 生成电子病历 |
| GET | `/medical/medical-records` | 获取病历列表 |
| GET | `/medical/medical-records/{id}` | 获取病历详情 |
| PUT | `/medical/medical-records/{id}` | 更新病历 |
| DELETE | `/medical/medical-records/{id}` | 删除病历 |
| GET | `/medical/medical-records/{id}/pdf` | 导出PDF |

## � 项目文结构

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
├── start_server.py               # 启动脚本（优化）
├── DEPLOY_SUMMARY.md             # 详细部署文档
└── README.md
```

## ⚙️ 依赖服务

| 服务 | 端口 | 说明 |
|------|------|------|
| 声纹识别API | 8520 | 主服务 |
| SenseVoice ASR | 8001 | 语音转文字（需单独部署） |
| Ollama | 11434 | 大模型服务 |

## 📖 详细文档

完整部署指南和问题排查请参考 [DEPLOY_SUMMARY.md](./DEPLOY_SUMMARY.md)

## 🔒 安全提示

- `data/.voiceprint.yaml` 包含敏感信息，已加入 `.gitignore`
- 请勿将密码和 API 密钥提交到仓库

## 📄 License

MIT License

## 🙏 致谢

- 原项目：[xinnan-tech/voiceprint-api](https://github.com/xinnan-tech/voiceprint-api)
- 声纹模型：[3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker)
- 说话人分离：[Pyannote](https://github.com/pyannote/pyannote-audio)
