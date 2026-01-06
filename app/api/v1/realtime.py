"""
实时对话识别 WebSocket 接口
支持多人对话场景的实时说话人识别和语音转文字
使用 Silero VAD 进行语音活动检测
"""
import asyncio
import numpy as np
import tempfile
import soundfile as sf
import torch
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from typing import Optional, List
import json
import time

from ...services.voiceprint_service import voiceprint_service
from ...core.config import settings
from ...core.logger import get_logger
from ...database.voiceprint_db import voiceprint_db
from ...utils.audio_utils import audio_processor

logger = get_logger(__name__)

router = APIRouter()

# 音频参数
SAMPLE_RATE = 16000
SILERO_VAD_CHUNK_SIZE = 512  # Silero VAD 要求的采样点数（16kHz下约32ms）
CHUNK_DURATION = 0.1  # 每个音频块的时长（秒）- 用于缓冲计算
MIN_SPEECH_DURATION = 0.5  # 最小语音片段时长（秒）
MAX_SPEECH_DURATION = 10.0  # 最大语音片段时长（秒）
SILENCE_DURATION = 0.8  # 静音持续时间触发切分（秒）- 0.8秒避免说话停顿被误切
NOISE_TIMEOUT = 5.0  # 持续噪音超时（秒）
VAD_THRESHOLD = 0.5  # Silero VAD 阈值 (0-1)，越高越严格

# 全局 Silero VAD 模型（懒加载）
_silero_vad_model = None
_silero_vad_utils = None


def get_silero_vad():
    """获取 Silero VAD 模型（单例模式）"""
    global _silero_vad_model, _silero_vad_utils
    if _silero_vad_model is None:
        try:
            logger.info("加载 Silero VAD 模型...")
            
            # 方法1：尝试从本地目录加载
            import os
            local_vad_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'silero-vad')
            if os.path.exists(local_vad_path):
                logger.info(f"从本地加载 Silero VAD: {local_vad_path}")
                model, utils = torch.hub.load(
                    repo_or_dir=local_vad_path,
                    model='silero_vad',
                    source='local',
                    trust_repo=True
                )
            else:
                # 方法2：从 GitHub 下载（需要网络）
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True
                )
            
            _silero_vad_model = model
            _silero_vad_utils = utils
            logger.info("Silero VAD 模型加载完成")
        except Exception as e:
            logger.error(f"Silero VAD 加载失败: {e}，将使用能量检测作为备选")
    return _silero_vad_model, _silero_vad_utils


class RealtimeProcessor:
    """实时音频处理器，使用 Silero VAD"""
    
    def __init__(self, speaker_ids: Optional[List[str]] = None):
        self.speaker_ids = speaker_ids
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_start_time = 0
        self.voiceprints = {}
        self.last_process_time = time.time()
        self._debug_counter = 0
        
        # 初始化 Silero VAD
        self.vad_model, self.vad_utils = get_silero_vad()
        self.use_silero = self.vad_model is not None
        
        if self.use_silero:
            logger.info("使用 Silero VAD 进行语音检测")
        else:
            logger.info("使用能量检测进行语音检测（备选方案）")
        
        # 加载声纹特征
        self._load_voiceprints()
    
    def _load_voiceprints(self):
        """加载候选声纹特征"""
        self.voiceprints = voiceprint_db.get_voiceprints(self.speaker_ids)
        if self.voiceprints:
            logger.info(f"实时识别加载了 {len(self.voiceprints)} 个声纹: {list(self.voiceprints.keys())}")
        else:
            logger.warning("没有可用的声纹特征")
    
    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """语音活动检测 - 优先使用 Silero VAD"""
        if self.use_silero:
            try:
                # Silero VAD 要求固定大小的输入（512 samples for 16kHz）
                # 将大块音频分成小块处理，取平均概率
                speech_probs = []
                for i in range(0, len(audio_chunk), SILERO_VAD_CHUNK_SIZE):
                    chunk = audio_chunk[i:i + SILERO_VAD_CHUNK_SIZE]
                    if len(chunk) == SILERO_VAD_CHUNK_SIZE:
                        audio_tensor = torch.from_numpy(chunk).float()
                        prob = self.vad_model(audio_tensor, SAMPLE_RATE).item()
                        speech_probs.append(prob)
                
                if speech_probs:
                    # 取最大概率（只要有一段是语音就算语音）
                    speech_prob = max(speech_probs)
                    is_speech = speech_prob > VAD_THRESHOLD
                    
                    # 调试日志
                    self._debug_counter += 1
                    if self._debug_counter % 50 == 0:
                        logger.info(f"[Silero VAD] 语音概率: {speech_prob:.3f}, 阈值: {VAD_THRESHOLD}, 是否语音: {is_speech}")
                    
                    return is_speech
            except Exception as e:
                logger.warning(f"Silero VAD 检测失败: {e}，回退到能量检测")
        
        # 备选：能量检测
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = energy > 0.03
        
        self._debug_counter += 1
        if self._debug_counter % 50 == 0:
            logger.info(f"[能量VAD] 音频能量: {energy:.6f}, 是否语音: {is_speech}")
        
        return is_speech
    
    def process_chunk(self, audio_data: bytes) -> Optional[dict]:
        """处理音频块，返回识别结果（如果有）"""
        # 将字节转换为 numpy 数组
        audio_chunk = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        is_speech = self._is_speech(audio_chunk)
        
        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = time.time()
                self.audio_buffer = []
                logger.debug("检测到语音开始")
            
            self.audio_buffer.append(audio_chunk)
            self.silence_frames = 0
            
            # 检查最大时长
            current_duration = len(self.audio_buffer) * CHUNK_DURATION
            if current_duration >= MAX_SPEECH_DURATION:
                self.last_process_time = time.time()
                return self._process_speech_segment()
            
            # 噪音超时检测
            if self.audio_buffer and (time.time() - self.last_process_time) >= NOISE_TIMEOUT:
                logger.info(f"噪音超时 ({NOISE_TIMEOUT}s)，强制处理")
                self.last_process_time = time.time()
                return self._process_speech_segment()
        else:
            if self.is_speaking:
                self.silence_frames += 1
                silence_duration = self.silence_frames * CHUNK_DURATION
                
                if silence_duration >= SILENCE_DURATION:
                    logger.info(f"检测到语音结束，静音时长: {silence_duration:.2f}s")
                    self.last_process_time = time.time()
                    return self._process_speech_segment()
                else:
                    self.audio_buffer.append(audio_chunk)
        
        return None

    
    def _process_speech_segment(self) -> Optional[List[dict]]:
        """
        处理一个完整的语音片段，使用 Pyannote 进行多人分离
        返回多个识别结果（如果检测到多人）
        """
        if not self.audio_buffer:
            self.is_speaking = False
            return None
        
        audio_data = np.concatenate(self.audio_buffer)
        duration = len(audio_data) / SAMPLE_RATE
        
        # 重置状态
        self.is_speaking = False
        self.audio_buffer = []
        self.silence_frames = 0
        
        if duration < MIN_SPEECH_DURATION:
            logger.debug(f"语音片段太短 ({duration:.2f}s)，跳过")
            return None
        
        logger.info(f"处理语音片段，时长: {duration:.2f}s，使用 Pyannote 多人分离")
        
        try:
            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.tmp_dir) as tmpf:
                sf.write(tmpf.name, audio_data, SAMPLE_RATE)
                audio_path = tmpf.name
            
            try:
                # 读取 WAV 文件字节（diarize_and_transcribe 需要完整的 WAV 格式）
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                
                # 调用 diarize_and_transcribe 进行多人分离
                segments = voiceprint_service.diarize_and_transcribe(
                    self.speaker_ids, 
                    audio_bytes
                )
                
                if segments:
                    logger.info(f"Pyannote 分离出 {len(segments)} 个片段")
                    # 返回所有分离出的片段
                    results = []
                    for seg in segments:
                        speaker_id = seg.get("speaker_id", "")
                        # 有声纹匹配 = 医生，无匹配 = 患者/家属
                        role = "医生" if speaker_id else "患者/家属"
                        results.append({
                            "start": seg.get("start", 0),
                            "end": seg.get("end", 0),
                            "duration": round(seg.get("end", 0) - seg.get("start", 0), 2),
                            "speaker_id": speaker_id,
                            "role": role,
                            "score": seg.get("score", 0.0),
                            "text": seg.get("text", ""),
                            "diarization_label": seg.get("diarization_label", "")
                        })
                    return results
                else:
                    # 没有分离出片段，回退到单人识别
                    logger.info("Pyannote 未分离出片段，使用单人识别")
                    recognition = voiceprint_service.identify_and_transcribe_segment(
                        audio_path, self.voiceprints if self.voiceprints else None
                    )
                    speaker_id = recognition["speaker_id"]
                    role = "医生" if speaker_id else "患者/家属"
                    return [{
                        "duration": round(duration, 2),
                        "speaker_id": speaker_id,
                        "role": role,
                        "score": recognition["score"],
                        "text": recognition["text"]
                    }]
            finally:
                audio_processor.cleanup_temp_file(audio_path)
            
        except Exception as e:
            logger.error(f"处理语音片段失败: {e}")
            return None
    
    def flush(self) -> Optional[dict]:
        """强制处理剩余的音频缓冲"""
        if self.audio_buffer:
            return self._process_speech_segment()
        return None


class ConversationMerger:
    """
    对话合并器：将同一说话人连续的消息合并
    用于机器人等客户端，减少消息碎片化
    """
    def __init__(self):
        self.last_speaker = None
        self.pending_message = None  # 待发送的合并消息
        self.merge_timeout = 2.0  # 同一说话人消息合并的超时时间（秒）
        self.last_message_time = 0
    
    def process(self, results: List[dict]) -> List[dict]:
        """
        处理识别结果，合并同一说话人的连续消息
        
        Args:
            results: 原始识别结果列表
            
        Returns:
            合并后的结果列表（可能为空，表示消息被缓存等待合并）
        """
        if not results:
            return []
        
        output = []
        current_time = time.time()
        
        for result in results:
            speaker = result.get("speaker_id", "") or result.get("diarization_label", "未知")
            text = result.get("text", "").strip()
            
            if not text:
                continue
            
            # 检查是否需要先发送之前缓存的消息
            if self.pending_message:
                # 如果说话人变了，或者超时了，先发送缓存的消息
                if speaker != self.last_speaker or (current_time - self.last_message_time) > self.merge_timeout:
                    output.append(self.pending_message)
                    self.pending_message = None
            
            # 合并或创建新消息
            if speaker == self.last_speaker and self.pending_message:
                # 同一说话人，合并文本
                self.pending_message["text"] += " " + text
                self.pending_message["duration"] = round(
                    self.pending_message.get("duration", 0) + result.get("duration", 0), 2
                )
                # 更新结束时间
                if "end" in result:
                    self.pending_message["end"] = result["end"]
            else:
                # 新说话人，创建新的待发送消息
                self.pending_message = {
                    "speaker_id": result.get("speaker_id", ""),
                    "score": result.get("score", 0.0),
                    "text": text,
                    "duration": result.get("duration", 0),
                    "diarization_label": result.get("diarization_label", ""),
                    "start": result.get("start", 0),
                    "end": result.get("end", 0),
                    "is_merged": False  # 标记是否为合并消息
                }
                self.last_speaker = speaker
            
            self.last_message_time = current_time
        
        return output
    
    def flush(self) -> Optional[dict]:
        """强制发送缓存的消息"""
        if self.pending_message:
            msg = self.pending_message
            self.pending_message = None
            self.last_speaker = None
            return msg
        return None


@router.websocket("/realtime")
async def realtime_conversation(
    websocket: WebSocket,
    token: str = Query(..., description="API Token"),
    speaker_ids: str = Query("", description="候选说话人ID，逗号分隔"),
    merge: str = Query("true", description="是否合并同一说话人连续消息")
):
    """
    实时对话识别 WebSocket 接口
    
    连接: ws://host:port/api/v1/voiceprint/realtime?token=xxx&speaker_ids=doctor,patient&merge=true
    
    客户端发送: PCM 16bit 16kHz 单声道音频数据（二进制）
    服务端返回: JSON {"speaker_id": "xxx", "score": 0.85, "text": "说话内容", "duration": 2.5}
    
    参数:
        merge: 是否合并同一说话人连续消息，默认 true
    """
    if token != settings.api_token:
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    await websocket.accept()
    logger.info(f"WebSocket 连接建立，候选说话人: {speaker_ids}, 消息合并: {merge}")
    
    candidate_ids = [x.strip() for x in speaker_ids.split(",") if x.strip()] if speaker_ids else None
    processor = RealtimeProcessor(candidate_ids)
    
    # 是否启用消息合并
    enable_merge = merge.lower() in ("true", "1", "yes")
    merger = ConversationMerger() if enable_merge else None
    
    chunk_count = 0
    
    # 定时检查是否需要发送缓存的消息
    async def check_pending_message():
        while True:
            await asyncio.sleep(0.5)
            if merger and merger.pending_message:
                # 检查是否超时
                if (time.time() - merger.last_message_time) > merger.merge_timeout:
                    msg = merger.flush()
                    if msg and msg.get("text"):
                        try:
                            await websocket.send_json(msg)
                            logger.info(f"[合并超时] 发送: {msg.get('speaker_id', '未知')} - {msg.get('text', '')}")
                        except:
                            break
    
    # 启动定时检查任务
    check_task = asyncio.create_task(check_pending_message()) if enable_merge else None
    
    try:
        while True:
            data = await websocket.receive_bytes()
            chunk_count += 1
            
            if chunk_count % 100 == 0:
                logger.info(f"已接收 {chunk_count} 个音频块")
            
            results = processor.process_chunk(data)
            if results:
                if merger:
                    # 使用合并器处理
                    merged_results = merger.process(results)
                    for result in merged_results:
                        if result.get("text"):
                            await websocket.send_json(result)
                            logger.info(f"[合并] 实时识别: {result.get('speaker_id', '未知')} - {result.get('text', '')}")
                else:
                    # 不合并，直接发送
                    for result in results:
                        if result.get("text"):
                            await websocket.send_json(result)
                            logger.info(f"实时识别: {result.get('speaker_id', '未知')} - {result.get('text', '')}")
    
    except WebSocketDisconnect:
        logger.info("WebSocket 连接断开")
        
        # 处理剩余的音频
        results = processor.flush()
        if results:
            if merger:
                merged_results = merger.process(results)
                for result in merged_results:
                    try:
                        if result.get("text"):
                            await websocket.send_json(result)
                    except:
                        pass
            else:
                for result in results:
                    try:
                        if result.get("text"):
                            await websocket.send_json(result)
                    except:
                        pass
        
        # 发送合并器中缓存的消息
        if merger:
            final_msg = merger.flush()
            if final_msg and final_msg.get("text"):
                try:
                    await websocket.send_json(final_msg)
                except:
                    pass
                    
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        try:
            await websocket.close(code=4000, reason=str(e))
        except:
            pass
    finally:
        # 取消定时检查任务
        if check_task:
            check_task.cancel()
            try:
                await check_task
            except asyncio.CancelledError:
                pass
