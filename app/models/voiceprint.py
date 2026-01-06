from pydantic import BaseModel
from typing import List, Optional


class VoiceprintRegisterRequest(BaseModel):
    """声纹注册请求模型"""

    speaker_id: str

    class Config:
        schema_extra = {"example": {"speaker_id": "user_001"}}


class VoiceprintRegisterResponse(BaseModel):
    """声纹注册响应模型"""

    success: bool
    msg: str

    class Config:
        schema_extra = {"example": {"success": True, "msg": "已登记: user_001"}}


class VoiceprintIdentifyRequest(BaseModel):
    """声纹识别请求模型"""

    speaker_ids: str  # 逗号分隔的候选说话人ID

    class Config:
        schema_extra = {"example": {"speaker_ids": "user_001,user_002,user_003"}}


class VoiceprintIdentifyResponse(BaseModel):
    """声纹识别响应模型"""

    speaker_id: str
    score: float

    class Config:
        schema_extra = {"example": {"speaker_id": "user_001", "score": 0.85}}


class DiarizationResponse(BaseModel):
    """多说话人识别响应模型"""

    segments: list
    speaker_count: int

    class Config:
        schema_extra = {
            "example": {
                "segments": [
                    {"start": 0.0, "end": 2.5, "speaker_id": "user_001", "score": 0.85, "diarization_label": "spk_0"},
                    {"start": 2.5, "end": 5.0, "speaker_id": "user_002", "score": 0.78, "diarization_label": "spk_1"}
                ],
                "speaker_count": 2
            }
        }


class ConversationSegment(BaseModel):
    """对话片段模型"""
    start: float
    end: float
    speaker_id: str
    score: float
    diarization_label: str
    text: str


class ConversationResponse(BaseModel):
    """多人对话识别响应模型（含转写）"""

    segments: List[ConversationSegment]
    speaker_count: int
    transcript: str  # 完整对话文本

    class Config:
        schema_extra = {
            "example": {
                "segments": [
                    {"start": 0.0, "end": 3.5, "speaker_id": "doctor_001", "score": 0.92, "diarization_label": "spk_0", "text": "你好，请问哪里不舒服？"},
                    {"start": 3.5, "end": 8.0, "speaker_id": "patient_001", "score": 0.88, "diarization_label": "spk_1", "text": "医生，我最近头疼得厉害"}
                ],
                "speaker_count": 2,
                "transcript": "[doctor_001]: 你好，请问哪里不舒服？\n[patient_001]: 医生，我最近头疼得厉害"
            }
        }
