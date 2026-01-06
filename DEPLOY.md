# 声纹识别服务部署文档

## 系统要求

- Python 3.10+
- MySQL 5.7+
- CUDA 11.8+（GPU 加速，可选）
- 内存：8GB+
- 磁盘：10GB+（模型缓存）

## 依赖服务

- **SenseVoice ASR 服务**：`http://192.168.0.207:8001`（语音转文字）

---

## 一、环境准备

### 1. 安装 Python 依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# WebSocket 支持
pip install "uvicorn[standard]" websockets
```

### 2. 创建 MySQL 数据库

```sql
CREATE DATABASE voiceprint_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE voiceprint_db;

CREATE TABLE voiceprints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    speaker_id VARCHAR(100) UNIQUE NOT NULL,
    feature_vector LONGBLOB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 3. 配置文件

编辑 `data/.voiceprint.yaml`：

```yaml
mysql:
  host: 127.0.0.1
  port: 3306
  user: root
  password: your_password
  database: voiceprint_db

server:
  host: 0.0.0.0
  port: 8005
  authorization: your-api-token-here  # 留空会自动生成
```

### 4. 创建目录

```bash
mkdir -p tmp logs data
```

---

## 二、启动服务

### 开发模式

```bash
python start_server.py
```

### 生产模式（推荐）

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8005 --workers 1
```

### 使用 systemd（Linux）

创建 `/etc/systemd/system/voiceprint.service`：

```ini
[Unit]
Description=Voiceprint API Service
After=network.target mysql.service

[Service]
Type=simple
User=root
WorkingDirectory=/path/to/voiceprint-api
ExecStart=/path/to/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8005
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable voiceprint
systemctl start voiceprint
```

---

## 三、API 接口

### 基础信息

- **Base URL**: `http://your-server:8005`
- **认证方式**: Bearer Token（Header: `Authorization: Bearer <token>`）

### 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/voiceprint/health` | GET | 健康检查 |
| `/voiceprint/register` | POST | 注册声纹 |
| `/voiceprint/identify` | POST | 单人声纹识别 |
| `/voiceprint/diarize` | POST | 多说话人分离识别 |
| `/voiceprint/conversation` | POST | 多人对话识别（含转写） |
| `/voiceprint/realtime` | WebSocket | 实时对话识别 |
| `/voiceprint/{speaker_id}` | DELETE | 删除声纹 |

### 接口详情

#### 1. 健康检查
```
GET /voiceprint/health?key=<token>
```

#### 2. 注册声纹
```
POST /voiceprint/register
Content-Type: multipart/form-data
Authorization: Bearer <token>

speaker_id: 说话人ID
file: WAV音频文件
```

#### 3. 声纹识别
```
POST /voiceprint/identify
Content-Type: multipart/form-data
Authorization: Bearer <token>

speaker_ids: 候选人ID（逗号分隔）
file: WAV音频文件
```

#### 4. 多人对话识别（含转写）
```
POST /voiceprint/conversation
Content-Type: multipart/form-data
Authorization: Bearer <token>

speaker_ids: 候选人ID（留空匹配所有）
file: WAV音频文件

返回:
{
  "segments": [
    {"start": 0.0, "end": 3.5, "speaker_id": "doctor", "score": 0.92, "text": "你好"},
    ...
  ],
  "speaker_count": 2,
  "transcript": "[doctor]: 你好\n[patient]: 医生好"
}
```

#### 5. 实时对话识别（WebSocket）
```
WS /voiceprint/realtime?token=<token>&speaker_ids=<ids>

客户端发送: PCM 16bit 16kHz 单声道音频数据（二进制）
服务端返回: JSON {"speaker_id": "xxx", "score": 0.85, "text": "内容", "duration": 2.5}
```

---

## 四、配置说明

### SenseVoice ASR 服务地址

修改 `app/services/voiceprint_service.py` 中的：

```python
SENSEVOICE_URL = "http://192.168.0.207:8001"
```

### 声纹识别阈值

修改 `data/.voiceprint.yaml` 或 `app/core/config.py`：

```yaml
voiceprint:
  similarity_threshold: 0.2  # 相似度阈值，越高越严格
```

### 实时识别 VAD 参数

修改 `app/api/v1/realtime.py`：

```python
SILENCE_THRESHOLD = 0.02  # 静音阈值，越高越不敏感
MIN_SPEECH_DURATION = 0.5  # 最小语音时长（秒）
SILENCE_DURATION = 0.6  # 静音多久触发识别（秒）
```

---

## 五、测试

### 使用测试页面

浏览器打开 `voiceprint_test.html`，配置 API 地址和 Token。

### 使用 curl

```bash
# 健康检查
curl "http://localhost:8005/voiceprint/health?key=your-token"

# 注册声纹
curl -X POST "http://localhost:8005/voiceprint/register" \
  -H "Authorization: Bearer your-token" \
  -F "speaker_id=doctor_001" \
  -F "file=@audio.wav"

# 多人对话识别
curl -X POST "http://localhost:8005/voiceprint/conversation" \
  -H "Authorization: Bearer your-token" \
  -F "speaker_ids=" \
  -F "file=@conversation.wav"
```

---

## 六、常见问题

### 1. 模型下载慢
首次启动会从 ModelScope 下载模型，可设置镜像：
```bash
export MODELSCOPE_CACHE=/path/to/cache
```

### 2. CUDA 内存不足
减少并发或使用 CPU 模式。

### 3. WebSocket 连接失败
确保安装了 `websockets`：
```bash
pip install "uvicorn[standard]"
```

### 4. ASR 无返回
检查 SenseVoice 服务是否正常运行：
```bash
curl -X POST "http://192.168.0.207:8001/api/v1/asr" \
  -F "files=@test.wav" \
  -F "keys=test" \
  -F "lang=zh"
```
