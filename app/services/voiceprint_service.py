import os
# 设置 Hugging Face 离线模式，必须在 import pyannote 之前设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
import time
import threading
import soundfile as sf
import tempfile
import requests
from typing import Dict, List, Optional, Tuple
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from ..core.config import settings
from ..core.logger import get_logger
from ..database.voiceprint_db import voiceprint_db
from ..utils.audio_utils import audio_processor

logger = get_logger(__name__)

# SenseVoice ASR 服务配置
SENSEVOICE_URL = "http://192.168.0.207:8001"

# 是否使用 Pyannote 进行说话人分离（更强的分离能力）
USE_PYANNOTE = True  # 设为 False 则使用原来的 3DSpeaker 分离


class VoiceprintService:
    """声纹识别服务类"""

    def __init__(self):
        self._pipeline = None
        self._diarization_pipeline = None  # 3DSpeaker 说话人分离 pipeline
        self._pyannote_pipeline = None  # Pyannote 说话人分离 pipeline
        self.similarity_threshold = settings.similarity_threshold
        self._pipeline_lock = threading.Lock()  # 添加线程锁
        self._init_pipeline()
        self._init_diarization_pipeline()  # 初始化说话人分离模型
        self._warmup_model()  # 添加模型预热
        logger.info(f"ASR服务使用远程 SenseVoice: {SENSEVOICE_URL}")

    def _init_pipeline(self) -> None:
        """初始化声纹识别模型"""
        start_time = time.time()
        logger.start("初始化声纹识别模型")

        try:
            # 检查CUDA可用性
            if torch.cuda.is_available():
                device = "gpu"
                logger.info(f"使用GPU设备: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("使用CPU设备")

            logger.info("开始加载模型: iic/speech_campplus_sv_zh-cn_3dspeaker_16k")
            self._pipeline = pipeline(
                task=Tasks.speaker_verification,
                model="iic/speech_campplus_sv_zh-cn_3dspeaker_16k",
                device=device,
            )

            init_time = time.time() - start_time
            logger.complete("初始化声纹识别模型", init_time)
        except Exception as e:
            init_time = time.time() - start_time
            logger.fail(f"声纹模型加载失败，耗时: {init_time:.3f}秒，错误: {e}")
            raise

    def _init_diarization_pipeline(self) -> None:
        """初始化说话人分离模型"""
        start_time = time.time()
        logger.start("初始化说话人分离模型")

        # 优先尝试加载 Pyannote（更强的分离能力）
        if USE_PYANNOTE:
            try:
                logger.info("尝试加载 Pyannote 说话人分离模型...")
                from pyannote.audio import Pipeline as PyannotePipeline
                
                # Pyannote 4.0+ 使用 token 参数（离线模式已在文件顶部设置）
                hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                
                # 从本地缓存加载（离线模式）
                self._pyannote_pipeline = PyannotePipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token
                )
                
                # 移动到 GPU
                if torch.cuda.is_available():
                    self._pyannote_pipeline.to(torch.device("cuda"))
                    logger.info("Pyannote 3.1 使用 GPU 加速")
                
                init_time = time.time() - start_time
                logger.complete("初始化 Pyannote 3.1 说话人分离模型", init_time)
                return
            except ImportError:
                logger.warning("pyannote.audio 未安装，回退到 3DSpeaker 分离模型")
            except Exception as e:
                logger.warning(f"Pyannote 加载失败: {e}，回退到 3DSpeaker 分离模型")

        # 回退到 3DSpeaker 说话人分离
        try:
            if torch.cuda.is_available():
                device = "gpu"
            else:
                device = "cpu"

            logger.info("开始加载模型: iic/speech_campplus_speaker-diarization_common")
            self._diarization_pipeline = pipeline(
                task=Tasks.speaker_diarization,
                model="iic/speech_campplus_speaker-diarization_common",
                device=device,
            )

            init_time = time.time() - start_time
            logger.complete("初始化 3DSpeaker 说话人分离模型", init_time)
        except Exception as e:
            init_time = time.time() - start_time
            logger.warning(f"说话人分离模型加载失败，耗时: {init_time:.3f}秒，错误: {e}")

    def _warmup_model(self) -> None:
        """模型预热，避免第一次推理的延迟"""
        start_time = time.time()
        logger.start("开始模型预热")

        try:
            # 预热librosa重采样组件
            logger.debug("预热librosa重采样组件...")
            import librosa
            import soundfile as sf
            import tempfile

            # 生成一个更真实的测试音频（1秒的随机音频，模拟真实语音）
            sample_rate = 16000
            duration = 1.0  # 1秒
            samples = int(sample_rate * duration)

            # 生成随机音频数据，模拟真实语音
            np.random.seed(42)  # 固定随机种子，确保可重现
            test_audio = (
                np.random.randn(samples).astype(np.float32) * 0.1
            )  # 小幅度随机音频

            # 创建不同采样率的音频文件进行预热
            test_rates = [8000, 22050, 44100, 16000]  # 测试不同采样率

            for test_rate in test_rates:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpf:
                    # 生成测试采样率的音频
                    test_samples = int(test_rate * duration)
                    test_audio_resampled = librosa.resample(
                        test_audio, orig_sr=sample_rate, target_sr=test_rate
                    )
                    sf.write(tmpf.name, test_audio_resampled, test_rate)
                    temp_audio_path = tmpf.name

                try:
                    # 使用音频处理器处理这个文件（预热音频处理流程）
                    with open(temp_audio_path, "rb") as f:
                        audio_bytes = f.read()

                    # 预热音频处理
                    processed_path = audio_processor.ensure_16k_wav(audio_bytes)

                    # 预热模型推理
                    with self._pipeline_lock:
                        result = self._pipeline([processed_path], output_emb=True)
                        emb = self._to_numpy(result["embs"][0]).astype(np.float32)
                        logger.debug(
                            f"模型预热完成 ({test_rate}Hz -> 16kHz)，特征维度: {emb.shape}"
                        )

                    # 清理处理后的文件
                    audio_processor.cleanup_temp_file(processed_path)

                finally:
                    # 清理临时文件
                    import os

                    if os.path.exists(temp_audio_path):
                        os.unlink(temp_audio_path)

            warmup_time = time.time() - start_time
            logger.complete("模型预热完成", warmup_time)

        except Exception as e:
            warmup_time = time.time() - start_time
            logger.warning(f"模型预热失败，耗时: {warmup_time:.3f}秒，错误: {e}")
            # 预热失败不影响服务启动，只记录警告

    def _to_numpy(self, x) -> np.ndarray:
        """
        将torch tensor或其他类型转为numpy数组

        Args:
            x: 输入数据

        Returns:
            np.ndarray: numpy数组
        """
        return x.cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

    def extract_voiceprint(self, audio_path: str) -> np.ndarray:
        """
        从音频文件中提取声纹特征

        Args:
            audio_path: 音频文件路径

        Returns:
            np.ndarray: 声纹特征向量
        """
        start_time = time.time()
        logger.start(f"提取声纹特征，音频文件: {audio_path}")

        try:
            # 使用线程锁确保模型推理的线程安全
            with self._pipeline_lock:
                pipeline_start = time.time()
                logger.debug("开始模型推理...")

                # 检查pipeline是否可用
                if self._pipeline is None:
                    raise RuntimeError("声纹模型未初始化")

                result = self._pipeline([audio_path], output_emb=True)
                pipeline_time = time.time() - pipeline_start
                logger.debug(f"模型推理完成，耗时: {pipeline_time:.3f}秒")

            convert_start = time.time()
            emb = self._to_numpy(result["embs"][0]).astype(np.float32)
            convert_time = time.time() - convert_start
            logger.debug(f"数据转换完成，耗时: {convert_time:.3f}秒")

            total_time = time.time() - start_time
            logger.complete(f"提取声纹特征，维度: {emb.shape}", total_time)
            return emb
        except Exception as e:
            total_time = time.time() - start_time
            logger.fail(f"声纹特征提取失败，总耗时: {total_time:.3f}秒，错误: {e}")
            raise

    def calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        计算两个声纹特征的相似度

        Args:
            emb1: 声纹特征1
            emb2: 声纹特征2

        Returns:
            float: 相似度分数 (0-1)
        """
        try:
            # 使用余弦相似度
            similarity = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0

    def register_voiceprint(self, speaker_id: str, audio_bytes: bytes) -> bool:
        """
        注册声纹

        Args:
            speaker_id: 说话人ID
            audio_bytes: 音频字节数据

        Returns:
            bool: 注册是否成功
        """
        audio_path = None
        try:
            # 简化音频验证，只做基本检查
            if len(audio_bytes) < 1000:  # 文件太小
                logger.warning(f"音频文件过小: {speaker_id}")
                return False

            # 处理音频文件
            audio_path = audio_processor.ensure_16k_wav(audio_bytes)

            # 提取声纹特征
            emb = self.extract_voiceprint(audio_path)

            # 保存到数据库
            success = voiceprint_db.save_voiceprint(speaker_id, emb)

            if success:
                logger.info(f"声纹注册成功: {speaker_id}")
            else:
                logger.error(f"声纹注册失败: {speaker_id}")

            return success

        except Exception as e:
            logger.error(f"声纹注册异常 {speaker_id}: {e}")
            return False
        finally:
            # 清理临时文件
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)

    def _check_audio_has_speech(self, audio_path: str, min_energy: float = 0.01) -> bool:
        """
        检查音频是否包含有效语音（简单能量检测）
        
        Args:
            audio_path: 音频文件路径
            min_energy: 最小能量阈值
            
        Returns:
            bool: 是否包含有效语音
        """
        try:
            audio_data, sr = sf.read(audio_path)
            # 计算 RMS 能量
            energy = np.sqrt(np.mean(audio_data ** 2))
            # 计算有效语音帧比例（能量超过阈值的帧）
            frame_size = int(sr * 0.025)  # 25ms 帧
            hop_size = int(sr * 0.010)    # 10ms 步长
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_data) - frame_size, hop_size):
                frame = audio_data[i:i + frame_size]
                frame_energy = np.sqrt(np.mean(frame ** 2))
                total_frames += 1
                if frame_energy > min_energy:
                    speech_frames += 1
            
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            logger.debug(f"音频能量: {energy:.4f}, 语音帧比例: {speech_ratio:.2%}")
            
            # 至少 20% 的帧有语音，且整体能量超过阈值
            return energy > min_energy and speech_ratio > 0.2
        except Exception as e:
            logger.warning(f"语音检测失败: {e}")
            return True  # 检测失败时默认继续处理

    def identify_voiceprint(
        self, speaker_ids: Optional[List[str]], audio_bytes: bytes
    ) -> Tuple[str, float]:
        """
        识别声纹

        Args:
            speaker_ids: 候选说话人ID列表（None表示匹配所有已注册声纹）
            audio_bytes: 音频字节数据

        Returns:
            Tuple[str, float]: (识别出的说话人ID, 相似度分数)
        """
        start_time = time.time()
        logger.info(f"开始声纹识别流程，候选说话人: {'全部' if speaker_ids is None else f'{len(speaker_ids)}个'}")

        audio_path = None
        try:
            # 简化音频验证
            if len(audio_bytes) < 1000:
                logger.warning("音频文件过小")
                return "", 0.0

            # 处理音频文件
            audio_process_start = time.time()
            audio_path = audio_processor.ensure_16k_wav(audio_bytes)
            audio_process_time = time.time() - audio_process_start
            logger.debug(f"音频文件处理完成，耗时: {audio_process_time:.3f}秒")

            # 检查音频是否包含有效语音
            if not self._check_audio_has_speech(audio_path):
                logger.info("音频中未检测到有效语音，跳过识别")
                return "", 0.0

            # 提取声纹特征
            extract_start = time.time()
            logger.debug("开始提取声纹特征...")
            test_emb = self.extract_voiceprint(audio_path)
            extract_time = time.time() - extract_start
            logger.debug(f"声纹特征提取完成，耗时: {extract_time:.3f}秒")

            # 获取候选声纹特征
            db_query_start = time.time()
            logger.debug("开始查询数据库获取候选声纹特征...")
            voiceprints = voiceprint_db.get_voiceprints(speaker_ids)
            db_query_time = time.time() - db_query_start
            logger.debug(
                f"数据库查询完成，获取到{len(voiceprints)}个声纹特征，耗时: {db_query_time:.3f}秒"
            )

            if not voiceprints:
                logger.info("未找到候选说话人声纹")
                return "", 0.0

            # 计算相似度
            similarity_start = time.time()
            logger.debug("开始计算相似度...")
            similarities = {}
            for name, emb in voiceprints.items():
                similarity = self.calculate_similarity(test_emb, emb)
                similarities[name] = similarity
            similarity_time = time.time() - similarity_start
            logger.debug(
                f"相似度计算完成，共计算{len(similarities)}个，耗时: {similarity_time:.3f}秒"
            )

            # 找到最佳匹配
            if not similarities:
                return "", 0.0

            match_name = max(similarities, key=similarities.get)
            match_score = similarities[match_name]

            # 检查是否超过阈值
            if match_score < self.similarity_threshold:
                logger.info(
                    f"未识别到说话人，最高分: {match_score:.4f}，阈值: {self.similarity_threshold}"
                )
                total_time = time.time() - start_time
                logger.info(f"声纹识别流程完成，总耗时: {total_time:.3f}秒")
                return "", match_score

            total_time = time.time() - start_time
            logger.info(
                f"识别到说话人: {match_name}, 分数: {match_score:.4f}, 总耗时: {total_time:.3f}秒"
            )
            return match_name, match_score

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"声纹识别异常，总耗时: {total_time:.3f}秒，错误: {e}")
            return "", 0.0
        finally:
            # 清理临时文件
            cleanup_start = time.time()
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)
            cleanup_time = time.time() - cleanup_start
            logger.debug(f"临时文件清理完成，耗时: {cleanup_time:.3f}秒")

    def delete_voiceprint(self, speaker_id: str) -> bool:
        """
        删除声纹

        Args:
            speaker_id: 说话人ID

        Returns:
            bool: 删除是否成功
        """
        return voiceprint_db.delete_voiceprint(speaker_id)

    def get_voiceprint_count(self) -> int:
        """
        获取声纹总数

        Returns:
            int: 声纹总数
        """
        start_time = time.time()
        logger.info("开始获取声纹总数...")

        try:
            count = voiceprint_db.count_voiceprints()
            total_time = time.time() - start_time
            logger.info(f"声纹总数获取完成: {count}，耗时: {total_time:.3f}秒")
            return count
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"获取声纹总数失败，总耗时: {total_time:.3f}秒，错误: {e}")
            raise

    def diarize_and_identify(
        self, speaker_ids: Optional[List[str]], audio_bytes: bytes
    ) -> List[Dict]:
        """
        多说话人分离与识别
        
        使用 Pyannote 3.1 进行说话人分离（如果可用），3DSpeaker CAM++ 进行声纹识别

        Args:
            speaker_ids: 候选说话人ID列表（None表示匹配所有）
            audio_bytes: 音频字节数据

        Returns:
            List[Dict]: 识别结果列表
        """
        start_time = time.time()
        logger.info("开始多说话人分离与识别")

        # 检查分离模型是否可用
        if self._pyannote_pipeline is None and self._diarization_pipeline is None:
            logger.error("说话人分离模型未初始化")
            raise RuntimeError("说话人分离模型未初始化")

        audio_path = None
        try:
            if len(audio_bytes) < 1000:
                logger.warning("音频文件过小")
                return []

            audio_path = audio_processor.ensure_16k_wav(audio_bytes)
            logger.info(f"音频文件处理完成: {audio_path}")

            # 说话人分离 - 优先使用 Pyannote
            diarize_start = time.time()
            segments = []
            
            if self._pyannote_pipeline is not None:
                logger.info("使用 Pyannote 3.1 进行说话人分离...")
                segments = self._diarize_with_pyannote(audio_path)
            else:
                logger.info("使用 3DSpeaker 进行说话人分离...")
                with self._pipeline_lock:
                    diarization_result = self._diarization_pipeline(audio_path)
                logger.info(f"分离原始结果类型: {type(diarization_result)}")
                logger.info(f"分离原始结果: {diarization_result}")
                segments = self._parse_diarization_result(diarization_result)
            
            diarize_time = time.time() - diarize_start
            logger.info(f"说话人分离完成，耗时: {diarize_time:.3f}秒")

            if not segments:
                logger.info("未检测到说话人片段")
                return []

            logger.info(f"检测到 {len(segments)} 个说话人片段")

            # 获取候选声纹特征
            voiceprints = voiceprint_db.get_voiceprints(speaker_ids)
            if not voiceprints:
                return [{"start": seg["start"], "end": seg["end"], "speaker_id": "", "score": 0.0, "diarization_label": seg["speaker"]} for seg in segments]

            # 读取原始音频数据
            audio_data, sr = sf.read(audio_path)

            # 对每个片段进行声纹识别
            results = []
            for seg in segments:
                seg_start = seg["start"]
                seg_end = seg["end"]
                
                start_sample = int(seg_start * sr)
                end_sample = int(seg_end * sr)
                segment_audio = audio_data[start_sample:end_sample]

                if len(segment_audio) < sr * 0.5:
                    results.append({
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "speaker_id": "",
                        "score": 0.0,
                        "diarization_label": seg["speaker"]
                    })
                    continue

                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.tmp_dir) as tmpf:
                    sf.write(tmpf.name, segment_audio, sr)
                    segment_path = tmpf.name

                try:
                    seg_emb = self.extract_voiceprint(segment_path)

                    best_match = ""
                    best_score = 0.0
                    for name, emb in voiceprints.items():
                        similarity = self.calculate_similarity(seg_emb, emb)
                        if similarity > best_score:
                            best_score = similarity
                            best_match = name

                    if best_score < self.similarity_threshold:
                        best_match = ""

                    results.append({
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "speaker_id": best_match,
                        "score": round(best_score, 4),
                        "diarization_label": seg["speaker"]
                    })
                finally:
                    audio_processor.cleanup_temp_file(segment_path)

            total_time = time.time() - start_time
            logger.info(f"多说话人识别完成，共 {len(results)} 个片段，总耗时: {total_time:.3f}秒")
            return results

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"多说话人识别异常，总耗时: {total_time:.3f}秒，错误: {e}")
            raise
        finally:
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)

    def transcribe_audio(self, audio_path: str) -> str:
        """
        对音频进行语音转文字（调用 SenseVoice 服务）

        Args:
            audio_path: 音频文件路径

        Returns:
            str: 转写的文本
        """
        try:
            # 调用 SenseVoice 服务 POST /api/v1/asr
            with open(audio_path, "rb") as f:
                files = {"files": ("audio.wav", f, "audio/wav")}
                data = {
                    "keys": "audio",
                    "lang": "zh"
                }
                response = requests.post(
                    f"{SENSEVOICE_URL}/api/v1/asr",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"SenseVoice ASR原始结果: {result}")
                
                # 解析返回的JSON，提取clean_text
                if isinstance(result, dict) and "result" in result:
                    # 格式: {"result": [{"key": "audio", "text": "...", "clean_text": "..."}]}
                    results = result.get("result", [])
                    if results and isinstance(results, list):
                        clean_text = results[0].get("clean_text", "") or results[0].get("text", "")
                        return clean_text
                elif isinstance(result, list) and len(result) > 0:
                    # 格式: [{"key": "audio", "text": "...", "clean_text": "..."}]
                    clean_text = result[0].get("clean_text", "") or result[0].get("text", "")
                    return clean_text
                elif isinstance(result, str):
                    return result
                
                return ""
            else:
                logger.error(f"SenseVoice ASR请求失败: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到 SenseVoice 服务: {SENSEVOICE_URL}")
            return ""
        except Exception as e:
            logger.error(f"语音转文字失败: {e}")
            return ""

    def diarize_and_transcribe(
        self, speaker_ids: Optional[List[str]], audio_bytes: bytes
    ) -> List[Dict]:
        """
        多说话人分离、识别与转写（医生-患者对话场景）
        
        使用 Pyannote 3.1 进行说话人分离（如果可用），3DSpeaker CAM++ 进行声纹识别

        Args:
            speaker_ids: 候选说话人ID列表（如 ["doctor_001", "patient_001"]）
            audio_bytes: 音频字节数据

        Returns:
            List[Dict]: 识别结果列表，包含说话人身份和说话内容
        """
        start_time = time.time()
        logger.info("开始多说话人分离、识别与转写")

        # 检查分离模型是否可用
        if self._pyannote_pipeline is None and self._diarization_pipeline is None:
            logger.error("说话人分离模型未初始化")
            raise RuntimeError("说话人分离模型未初始化")

        audio_path = None
        try:
            if len(audio_bytes) < 1000:
                logger.warning("音频文件过小")
                return []

            audio_path = audio_processor.ensure_16k_wav(audio_bytes)
            logger.info(f"音频文件处理完成: {audio_path}")

            # 说话人分离 - 优先使用 Pyannote
            diarize_start = time.time()
            segments = []
            
            if self._pyannote_pipeline is not None:
                logger.info("使用 Pyannote 3.1 进行说话人分离...")
                segments = self._diarize_with_pyannote(audio_path)
            else:
                logger.info("使用 3DSpeaker 进行说话人分离...")
                with self._pipeline_lock:
                    diarization_result = self._diarization_pipeline(audio_path)
                segments = self._parse_diarization_result(diarization_result)
            
            diarize_time = time.time() - diarize_start
            logger.info(f"说话人分离完成，耗时: {diarize_time:.3f}秒")

            if not segments:
                logger.info("未检测到说话人片段")
                return []

            logger.info(f"检测到 {len(segments)} 个说话人片段")

            # 获取候选声纹特征（如果未指定则获取所有已注册的声纹）
            voiceprints = voiceprint_db.get_voiceprints(speaker_ids)
            if voiceprints:
                logger.info(f"获取到 {len(voiceprints)} 个候选声纹: {list(voiceprints.keys())}")
            else:
                logger.warning("数据库中没有已注册的声纹")

            # 读取原始音频数据
            audio_data, sr = sf.read(audio_path)

            # 对每个片段进行声纹识别（3DSpeaker CAM++）和ASR转写
            results = []
            for seg in segments:
                seg_start = seg["start"]
                seg_end = seg["end"]
                
                start_sample = int(seg_start * sr)
                end_sample = int(seg_end * sr)
                segment_audio = audio_data[start_sample:end_sample]

                # 片段太短，跳过
                if len(segment_audio) < sr * 0.3:
                    continue

                # 保存片段到临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.tmp_dir) as tmpf:
                    sf.write(tmpf.name, segment_audio, sr)
                    segment_path = tmpf.name

                try:
                    # 声纹识别（使用 3DSpeaker CAM++）
                    best_match = ""
                    best_score = 0.0
                    
                    if voiceprints:
                        seg_emb = self.extract_voiceprint(segment_path)
                        for name, emb in voiceprints.items():
                            similarity = self.calculate_similarity(seg_emb, emb)
                            if similarity > best_score:
                                best_score = similarity
                                best_match = name

                        if best_score < self.similarity_threshold:
                            best_match = ""

                    # ASR转写
                    transcript = self.transcribe_audio(segment_path)

                    results.append({
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2),
                        "speaker_id": best_match,
                        "score": round(best_score, 4),
                        "diarization_label": seg["speaker"],
                        "text": transcript
                    })
                finally:
                    audio_processor.cleanup_temp_file(segment_path)

            total_time = time.time() - start_time
            logger.info(f"多说话人识别与转写完成，共 {len(results)} 个片段，总耗时: {total_time:.3f}秒")
            return results

        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"多说话人识别与转写异常，总耗时: {total_time:.3f}秒，错误: {e}")
            raise
        finally:
            if audio_path:
                audio_processor.cleanup_temp_file(audio_path)

    def _diarize_with_pyannote(self, audio_path: str) -> List[Dict]:
        """
        使用 Pyannote 3.1 进行说话人分离
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            List[Dict]: 分离结果 [{"start": float, "end": float, "speaker": str}, ...]
        """
        segments = []
        try:
            # Pyannote 分离
            diarization = self._pyannote_pipeline(audio_path)
            logger.info(f"Pyannote 返回类型: {type(diarization)}")
            
            # 获取实际的分离结果
            annotation = None
            
            # 新版本 DiarizeOutput 有 speaker_diarization 属性
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
                logger.info("使用 speaker_diarization 属性")
            # 或者直接就是 Annotation 对象
            elif hasattr(diarization, 'itertracks'):
                annotation = diarization
                logger.info("直接使用 diarization 对象")
            
            # 解析 Annotation
            if annotation is not None and hasattr(annotation, 'itertracks'):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segments.append({
                        "start": turn.start,
                        "end": turn.end,
                        "speaker": speaker
                    })
            
            logger.info(f"Pyannote 分离出 {len(segments)} 个片段")
            
        except Exception as e:
            logger.error(f"Pyannote 分离失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return segments

    def _parse_diarization_result(self, result) -> List[Dict]:
        """解析说话人分离结果"""
        segments = []
        try:
            if isinstance(result, dict):
                # 格式: {'text': [[start, end, speaker_id], ...]}
                if "text" in result:
                    for item in result["text"]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            segments.append({
                                "start": item[0],
                                "end": item[1],
                                "speaker": f"spk_{item[2]}" if len(item) > 2 else "unknown"
                            })
                elif "sentences" in result:
                    for sent in result["sentences"]:
                        segments.append({
                            "start": sent.get("start", 0),
                            "end": sent.get("end", 0),
                            "speaker": sent.get("speaker", "unknown")
                        })
                elif "labels" in result:
                    for label in result["labels"]:
                        segments.append({
                            "start": label[0],
                            "end": label[1],
                            "speaker": label[2] if len(label) > 2 else "unknown"
                        })
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        segments.append({
                            "start": item[0],
                            "end": item[1],
                            "speaker": item[2] if len(item) > 2 else "unknown"
                        })
                    elif isinstance(item, dict):
                        segments.append({
                            "start": item.get("start", 0),
                            "end": item.get("end", 0),
                            "speaker": item.get("speaker", "unknown")
                        })
        except Exception as e:
            logger.error(f"解析分离结果失败: {e}")
        return segments

    def identify_and_transcribe_segment(
        self, audio_path: str, voiceprints: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        对单个音频片段进行声纹识别和ASR转写（供实时识别复用）

        Args:
            audio_path: 音频文件路径
            voiceprints: 候选声纹特征字典，None则从数据库获取所有

        Returns:
            Dict: {"speaker_id": str, "score": float, "text": str}
        """
        result = {
            "speaker_id": "",
            "score": 0.0,
            "text": ""
        }

        try:
            # 获取声纹特征
            if voiceprints is None:
                voiceprints = voiceprint_db.get_voiceprints(None)

            # 声纹识别
            if voiceprints:
                seg_emb = self.extract_voiceprint(audio_path)
                best_match = ""
                best_score = 0.0

                for name, emb in voiceprints.items():
                    similarity = self.calculate_similarity(seg_emb, emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = name

                if best_score >= self.similarity_threshold:
                    result["speaker_id"] = best_match
                    result["score"] = round(best_score, 4)

            # ASR 转写
            text = self.transcribe_audio(audio_path)
            result["text"] = text

        except Exception as e:
            logger.error(f"识别片段失败: {e}")

        return result


# 全局声纹服务实例
voiceprint_service = VoiceprintService()
