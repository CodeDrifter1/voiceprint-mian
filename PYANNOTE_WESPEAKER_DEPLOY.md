# Pyannote 3.1 + WeSpeaker 部署方案

基于 Pyannote 3.1（说话人分离）+ WeSpeaker ResNet221（声纹识别）的高精度对话身份识别方案。

## 系统要求

- Python 3.10+
- CUDA 11.8+
- GPU: RTX 4090（24GB 显存充足）
- 内存: 16GB+

## 一、环境准备

### 1. 创建虚拟环境

```bash
conda create -n pyannote python=3.10
conda activate pyannote
```

### 2. 安装依赖

```bash
# PyTorch (CUDA 11.8)
pip install torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Pyannote
pip install pyannote.audio==3.1.1

# WeSpeaker
pip install wespeaker

# 其他依赖
pip install fastapi uvicorn python-multipart pyyaml pymysql
```

### 3. Hugging Face 授权

Pyannote 3.1 需要同意使用协议：

1. 注册 Hugging Face 账号：https://huggingface.co/
2. 访问模型页面，点击 "Agree"：
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. 创建 Access Token：https://huggingface.co/settings/tokens
4. 登录：

```bash
pip install huggingface_hub
huggingface-cli login
# 输入你的 token
```

## 二、代码示例

### 核心服务类

```python
# voiceprint_service_v2.py
import torch
import wespeaker
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import numpy as np
import soundfile as sf
import tempfile
import os

class VoiceprintServiceV2:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载 Pyannote 说话人分离模型
        print("加载 Pyannote 3.1 说话人分离模型...")
        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True  # 需要先 huggingface-cli login
        )
        self.diarization_pipeline.to(self.device)
        
        # 加载 WeSpeaker 声纹识别模型
        print("加载 WeSpeaker ResNet221 模型...")
        self.speaker_model = wespeaker.load_model('chinese')
        self.speaker_model.set_gpu(0)  # 使用 GPU
        
        # 声纹数据库 {speaker_id: embedding}
        self.voiceprint_db = {}
        
    def register_speaker(self, speaker_id: str, audio_path: str):
        """注册声纹"""
        embedding = self.speaker_model.extract_embedding(audio_path)
        self.voiceprint_db[speaker_id] = embedding
        return {"speaker_id": speaker_id, "status": "registered"}
    
    def identify_speaker(self, embedding: np.ndarray, threshold: float = 0.5):
        """从数据库中识别说话人"""
        best_match = None
        best_score = -1
        
        for speaker_id, db_embedding in self.voiceprint_db.items():
            # 计算余弦相似度
            score = np.dot(embedding, db_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
            )
            if score > best_score:
                best_score = score
                best_match = speaker_id
        
        if best_score >= threshold:
            return best_match, float(best_score)
        return "unknown", float(best_score)
    
    def diarize_and_identify(self, audio_path: str):
        """说话人分离 + 身份识别"""
        # 1. Pyannote 说话人分离
        diarization = self.diarization_pipeline(audio_path)
        
        # 2. 读取音频
        audio, sr = sf.read(audio_path)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        results = []
        
        # 3. 对每个片段提取声纹并识别
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            start_sample = int(turn.start * sr)
            end_sample = int(turn.end * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # 片段太短跳过
            if len(segment_audio) < sr * 0.5:  # 至少 0.5 秒
                continue
            
            # 保存临时文件提取声纹
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, segment_audio, sr)
                embedding = self.speaker_model.extract_embedding(f.name)
                os.unlink(f.name)
            
            # 识别身份
            identified_speaker, score = self.identify_speaker(embedding)
            
            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "duration": round(turn.end - turn.start, 2),
                "pyannote_label": speaker_label,  # Pyannote 的临时标签
                "speaker_id": identified_speaker,  # 识别出的真实身份
                "score": round(score, 3)
            })
        
        return {
            "segments": results,
            "speaker_count": len(set(r["speaker_id"] for r in results))
        }


# 使用示例
if __name__ == "__main__":
    service = VoiceprintServiceV2()
    
    # 注册声纹
    service.register_speaker("doctor_001", "doctor_sample.wav")
    service.register_speaker("patient_001", "patient_sample.wav")
    
    # 对话识别
    result = service.diarize_and_identify("conversation.wav")
    for seg in result["segments"]:
        print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker_id']} (置信度: {seg['score']:.2f})")
```

## 三、性能对比

| 指标 | CAM++ (当前) | Pyannote + WeSpeaker |
|------|-------------|---------------------|
| 说话人分离精度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 重叠语音处理 | 一般 | 优秀 |
| 声纹识别精度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 抗噪能力 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 推理速度 | 快 | 中等 |
| 显存占用 | ~4GB | ~8GB |
| 部署复杂度 | 简单 | 中等（需要HF授权） |

## 四、迁移建议

### 方案 A：完全替换

用 Pyannote + WeSpeaker 替换现有的 CAM++ 方案，适合追求最高精度。

### 方案 B：混合使用

- 说话人分离：升级到 Pyannote 3.1
- 声纹识别：保留 CAM++（中文效果已经很好）

### 方案 C：保持现状

如果当前 CAM++ 满足需求，不建议折腾。等遇到明显问题再升级。

## 五、常见问题

### 1. Pyannote 下载失败

确保已经：
- 在 Hugging Face 同意了模型使用协议
- 执行了 `huggingface-cli login`

### 2. 显存不足

Pyannote 3.1 + WeSpeaker 大约需要 8GB 显存，4090 完全够用。如果同时跑其他模型，注意显存分配。

### 3. 中文效果

WeSpeaker 的 `chinese` 模型针对中文优化，效果很好。如果需要更好的中文效果，可以考虑保留 CAM++ 做声纹识别。
