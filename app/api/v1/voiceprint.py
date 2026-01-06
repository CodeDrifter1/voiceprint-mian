from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.security import HTTPBearer
from typing import List
import time
from ...models.voiceprint import VoiceprintRegisterResponse, VoiceprintIdentifyResponse, DiarizationResponse, ConversationResponse
from ...services.voiceprint_service import voiceprint_service
from ...database.voiceprint_db import voiceprint_db
from ...api.dependencies import AuthorizationToken
from ...core.logger import get_logger

# 创建安全模式
security = HTTPBearer(description="接口令牌")

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/list",
    summary="获取声纹列表",
    description="获取所有已注册的声纹ID列表",
    dependencies=[Depends(security)],
)
async def list_voiceprints(token: AuthorizationToken):
    """获取所有已注册声纹的ID列表"""
    try:
        voiceprints = voiceprint_db.list_voiceprints()
        return {
            "total": len(voiceprints),
            "voiceprints": voiceprints
        }
    except Exception as e:
        logger.error(f"获取声纹列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取声纹列表失败: {str(e)}")


@router.post(
    "/register",
    summary="声纹注册",
    response_model=VoiceprintRegisterResponse,
    description="注册新的声纹特征",
    dependencies=[Depends(security)],
)
async def register_voiceprint(
    token: AuthorizationToken,
    speaker_id: str = Form(..., description="说话人ID"),
    file: UploadFile = File(..., description="WAV音频文件"),
):
    """
    注册声纹接口

    Args:
        token: 接口令牌（Header）
        speaker_id: 说话人ID
        file: 说话人音频文件（WAV）

    Returns:
        VoiceprintRegisterResponse: 注册结果
    """
    try:
        # 验证文件类型
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="只支持WAV格式音频文件")

        # 读取音频数据
        audio_bytes = await file.read()

        # 注册声纹
        success = voiceprint_service.register_voiceprint(speaker_id, audio_bytes)

        if success:
            return VoiceprintRegisterResponse(success=True, msg=f"已登记: {speaker_id}")
        else:
            raise HTTPException(status_code=500, detail="声纹注册失败")

    except HTTPException:
        raise
    except Exception as e:
        logger.fail(f"声纹注册异常: {e}")
        raise HTTPException(status_code=500, detail=f"声纹注册失败: {str(e)}")


@router.post(
    "/identify",
    summary="声纹识别",
    response_model=VoiceprintIdentifyResponse,
    description="识别音频中的说话人",
    dependencies=[Depends(security)],
)
async def identify_voiceprint(
    token: AuthorizationToken,
    file: UploadFile = File(..., description="WAV音频文件"),
    speaker_ids: str = Form("", description="候选说话人ID，逗号分隔（留空则匹配所有）"),
):
    """
    声纹识别接口

    Args:
        token: 接口令牌（Header）
        speaker_ids: 候选说话人ID，逗号分隔（留空则匹配所有已注册声纹）
        file: 待识别音频文件（WAV）

    Returns:
        VoiceprintIdentifyResponse: 识别结果
    """
    start_time = time.time()
    logger.info(f"开始声纹识别请求 - 候选说话人: {speaker_ids or '全部'}, 文件: {file.filename}")

    try:
        # 验证文件类型
        validation_start = time.time()
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="只支持WAV格式音频文件")
        validation_time = time.time() - validation_start
        logger.info(f"文件类型验证完成，耗时: {validation_time:.3f}秒")

        # 解析候选说话人ID（留空则为None，表示匹配所有）
        parse_start = time.time()
        candidate_ids = [x.strip() for x in speaker_ids.split(",") if x.strip()] if speaker_ids.strip() else None
        parse_time = time.time() - parse_start
        logger.info(
            f"候选说话人ID解析完成，{'全部匹配' if candidate_ids is None else f'共{len(candidate_ids)}个'}，耗时: {parse_time:.3f}秒"
        )

        # 读取音频数据
        read_start = time.time()
        audio_bytes = await file.read()
        read_time = time.time() - read_start
        logger.info(
            f"音频文件读取完成，大小: {len(audio_bytes)}字节，耗时: {read_time:.3f}秒"
        )

        # 识别声纹
        identify_start = time.time()
        logger.info("开始调用声纹识别服务...")
        match_name, match_score = voiceprint_service.identify_voiceprint(
            candidate_ids, audio_bytes
        )
        identify_time = time.time() - identify_start
        logger.info(f"声纹识别服务调用完成，耗时: {identify_time:.3f}秒")

        total_time = time.time() - start_time
        logger.info(
            f"声纹识别请求完成，总耗时: {total_time:.3f}秒，识别结果: {match_name}, 分数: {match_score:.4f}"
        )

        return VoiceprintIdentifyResponse(speaker_id=match_name, score=match_score)

    except HTTPException:
        total_time = time.time() - start_time
        logger.error(f"声纹识别请求失败，总耗时: {total_time:.3f}秒")
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"声纹识别异常，总耗时: {total_time:.3f}秒，错误: {e}")
        raise HTTPException(status_code=500, detail=f"声纹识别失败: {str(e)}")


@router.post(
    "/diarize",
    summary="多说话人识别",
    response_model=DiarizationResponse,
    description="对音频进行说话人分离，并识别每个片段的说话人身份",
    dependencies=[Depends(security)],
)
async def diarize_and_identify(
    token: AuthorizationToken,
    file: UploadFile = File(..., description="WAV音频文件"),
    speaker_ids: str = Form("", description="候选说话人ID，逗号分隔（留空则匹配所有）"),
):
    """多说话人分离与识别接口"""
    start_time = time.time()
    logger.info(f"开始多说话人识别请求 - 文件: {file.filename}")

    try:
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="只支持WAV格式音频文件")

        candidate_ids = [x.strip() for x in speaker_ids.split(",") if x.strip()] if speaker_ids.strip() else None
        audio_bytes = await file.read()

        segments = voiceprint_service.diarize_and_identify(candidate_ids, audio_bytes)

        unique_speakers = set()
        for seg in segments:
            if seg.get("speaker_id"):
                unique_speakers.add(seg["speaker_id"])
            else:
                unique_speakers.add(seg.get("diarization_label", "unknown"))

        total_time = time.time() - start_time
        logger.info(f"多说话人识别完成，总耗时: {total_time:.3f}秒")

        return DiarizationResponse(segments=segments, speaker_count=len(unique_speakers))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多说话人识别异常: {e}")
        raise HTTPException(status_code=500, detail=f"多说话人识别失败: {str(e)}")


@router.post(
    "/conversation",
    summary="多人对话识别",
    response_model=ConversationResponse,
    description="对多人对话音频进行说话人分离、身份识别和语音转文字（适用于医生-患者对话场景）",
    dependencies=[Depends(security)],
)
async def conversation_recognize(
    token: AuthorizationToken,
    file: UploadFile = File(..., description="WAV音频文件"),
    speaker_ids: str = Form("", description="候选说话人ID，逗号分隔（如：doctor_001,patient_001）"),
):
    """
    多人对话识别接口（医生-患者场景）
    
    功能：
    1. 自动分离不同说话人的语音片段
    2. 识别每个片段是谁在说话（需预先注册声纹）
    3. 将每个片段转写为文字
    
    使用流程：
    1. 先用 /register 接口分别注册医生和患者的声纹
    2. 录制对话音频
    3. 调用此接口，传入音频和候选说话人ID
    
    Args:
        token: 接口令牌（Header）
        file: 对话音频文件（WAV格式）
        speaker_ids: 候选说话人ID，逗号分隔

    Returns:
        ConversationResponse: 包含每个片段的说话人、时间、内容
    """
    start_time = time.time()
    logger.info(f"开始多人对话识别请求 - 文件: {file.filename}")

    try:
        if not file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="只支持WAV格式音频文件")

        candidate_ids = [x.strip() for x in speaker_ids.split(",") if x.strip()] if speaker_ids.strip() else None
        audio_bytes = await file.read()

        segments = voiceprint_service.diarize_and_transcribe(candidate_ids, audio_bytes)

        # 统计说话人数量
        unique_speakers = set()
        for seg in segments:
            if seg.get("speaker_id"):
                unique_speakers.add(seg["speaker_id"])
            else:
                unique_speakers.add(seg.get("diarization_label", "unknown"))

        # 生成完整对话文本
        transcript_lines = []
        for seg in segments:
            speaker = seg.get("speaker_id") or seg.get("diarization_label", "unknown")
            text = seg.get("text", "")
            if text:
                transcript_lines.append(f"[{speaker}]: {text}")
        
        full_transcript = "\n".join(transcript_lines)

        total_time = time.time() - start_time
        logger.info(f"多人对话识别完成，总耗时: {total_time:.3f}秒")

        return ConversationResponse(
            segments=segments, 
            speaker_count=len(unique_speakers),
            transcript=full_transcript
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"多人对话识别异常: {e}")
        raise HTTPException(status_code=500, detail=f"多人对话识别失败: {str(e)}")


@router.delete(
    "/{speaker_id}",
    summary="删除声纹",
    description="删除指定说话人的声纹特征",
    dependencies=[Depends(security)],
)
async def delete_voiceprint(
    token: AuthorizationToken,
    speaker_id: str,
):
    """
    删除声纹接口

    Args:
        token: 接口令牌（Header）
        speaker_id: 说话人ID

    Returns:
        dict: 删除结果
    """
    try:
        success = voiceprint_service.delete_voiceprint(speaker_id)

        if success:
            return {"success": True, "msg": f"已删除: {speaker_id}"}
        else:
            raise HTTPException(status_code=404, detail=f"未找到说话人: {speaker_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除声纹异常 {speaker_id}: {e}")
        raise HTTPException(status_code=500, detail=f"删除声纹失败: {str(e)}")
