"""
电子病历生成接口
基于医患对话内容，使用大模型生成结构化电子病历
"""
import os
import json
import uuid
import io
from datetime import datetime
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Optional, List
from pydantic import BaseModel

try:
    from pypinyin import lazy_pinyin
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# 病历存储目录
MEDICAL_RECORDS_DIR = "data/medical_records"
PDF_EXPORT_DIR = "data/pdf_exports"

from ...services.voiceprint_service import voiceprint_service
from ...core.config import settings
from ...core.logger import get_logger
from ...api.dependencies import get_authorization_token

logger = get_logger(__name__)

router = APIRouter()

# Ollama 配置
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"  # 根据你的模型名称调整

# 电子病历生成的 Prompt
MEDICAL_RECORD_PROMPT = """你是一个医疗记录整理助手。请严格根据以下医患对话内容，提取并整理信息。

## 重要规则：
1. **只提取对话中明确说出的内容**，绝对不要自己推断、补充或编造任何信息
2. 如果对话中没有提到某项内容，必须填写"对话中未提及"
3. 医生评估部分：只记录医生在对话中明确说出的诊断、建议等，如果医生没说就写"对话中未提及"
4. 不要根据症状自己推断诊断结果

## 患者基本信息：
{patient_basic}

## 医患对话记录：
{conversation}

## 请按以下格式输出（JSON格式）：

```json
{{
  "patient_info": {{
    "chief_complaint": "患者描述的主要症状和持续时间（只写患者说的）",
    "present_illness": "患者描述的症状详情（只写患者说的）",
    "past_history": "患者提到的既往病史（没提到就写'对话中未提及'）",
    "personal_history": "患者提到的生活习惯（没提到就写'对话中未提及'）"
  }},
  "doctor_assessment": {{
    "physical_exam": "医生在对话中提到的检查结果（没说就写'对话中未提及'）",
    "preliminary_diagnosis": "医生在对话中明确说出的诊断（没说就写'对话中未提及'）",
    "differential_diagnosis": "医生在对话中提到的鉴别诊断（没说就写'对话中未提及'）",
    "treatment_plan": "医生在对话中给出的治疗建议（没说就写'对话中未提及'）",
    "follow_up": "医生在对话中提到的随访建议（没说就写'对话中未提及'）"
  }},
  "summary": "简要概括对话内容（不要添加推断）"
}}
```
"""


class MedicalRecordResponse(BaseModel):
    """电子病历响应"""
    success: bool
    conversation: List[dict]  # 原始对话记录
    medical_record: Optional[dict] = None  # 结构化病历
    raw_response: Optional[str] = None  # 大模型原始响应
    error: Optional[str] = None
    record_id: Optional[str] = None  # 病历ID，用于查询历史


class MedicalRecordItem(BaseModel):
    """病历列表项"""
    record_id: str
    patient_name: str
    patient_gender: str
    patient_age: str
    created_at: str
    summary: Optional[str] = None


def save_medical_record(
    record_id: str,
    patient_name: str,
    patient_gender: str,
    patient_age: str,
    conversation: List[dict],
    medical_record: Optional[dict],
    audio_filename: str
) -> bool:
    """保存病历到文件"""
    try:
        os.makedirs(MEDICAL_RECORDS_DIR, exist_ok=True)
        record_data = {
            "record_id": record_id,
            "patient_name": patient_name,
            "patient_gender": patient_gender,
            "patient_age": patient_age,
            "conversation": conversation,
            "medical_record": medical_record,
            "audio_filename": audio_filename,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        filepath = os.path.join(MEDICAL_RECORDS_DIR, f"{record_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record_data, f, ensure_ascii=False, indent=2)
        logger.info(f"病历已保存: {filepath}")
        return True
    except Exception as e:
        logger.error(f"保存病历失败: {e}")
        return False


def load_medical_record(record_id: str) -> Optional[dict]:
    """加载单个病历"""
    try:
        filepath = os.path.join(MEDICAL_RECORDS_DIR, f"{record_id}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"加载病历失败: {e}")
    return None


def list_medical_records() -> List[dict]:
    """列出所有病历"""
    records = []
    try:
        if not os.path.exists(MEDICAL_RECORDS_DIR):
            return records
        for filename in os.listdir(MEDICAL_RECORDS_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(MEDICAL_RECORDS_DIR, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        summary = None
                        if data.get("medical_record") and data["medical_record"].get("summary"):
                            summary = data["medical_record"]["summary"]
                        records.append({
                            "record_id": data.get("record_id", filename.replace(".json", "")),
                            "patient_name": data.get("patient_name", "未知"),
                            "patient_gender": data.get("patient_gender", ""),
                            "patient_age": data.get("patient_age", ""),
                            "created_at": data.get("created_at", ""),
                            "summary": summary
                        })
                except:
                    pass
        # 按时间倒序
        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception as e:
        logger.error(f"列出病历失败: {e}")
    return records


def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """调用 Ollama API"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # 低温度，更确定性的输出
                    "num_predict": 2000
                }
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            logger.error(f"Ollama API 错误: {response.status_code} - {response.text}")
            return ""
    except requests.exceptions.ConnectionError:
        logger.error(f"无法连接到 Ollama 服务: {OLLAMA_URL}")
        return ""
    except Exception as e:
        logger.error(f"Ollama 调用失败: {e}")
        return ""


def parse_medical_record(response: str) -> Optional[dict]:
    """从大模型响应中解析 JSON 格式的病历"""
    import json
    import re
    
    try:
        # 尝试提取 JSON 块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
        
        # 尝试直接解析整个响应
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("无法解析大模型返回的 JSON")
        return None


def format_conversation(segments: List[dict]) -> str:
    """
    将对话片段格式化为文本
    
    规则：
    - 识别出声纹的 → 医生（因为只有医生注册了声纹）
    - 识别不出声纹的 → 患者/家属
    """
    lines = []
    for seg in segments:
        speaker_id = seg.get("speaker_id", "")
        text = seg.get("text", "")
        if text:
            # 有声纹匹配 = 医生，无匹配 = 患者/家属
            if speaker_id:
                role = f"医生({speaker_id})"
            else:
                role = "患者/家属"
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


@router.post("/medical-record", response_model=MedicalRecordResponse)
async def generate_medical_record(
    file: UploadFile = File(..., description="医患对话音频文件"),
    speaker_ids: str = Form("", description="候选说话人ID，逗号分隔"),
    patient_name: str = Form("", description="患者姓名"),
    patient_gender: str = Form("", description="患者性别"),
    patient_age: str = Form("", description="患者年龄"),
    _: str = Depends(get_authorization_token)
):
    """
    生成电子病历
    
    上传医患对话音频，自动进行：
    1. 说话人分离（区分医生和患者）
    2. 语音转文字
    3. 大模型分析生成结构化病历
    4. 保存音频文件（患者姓名_时间.wav）
    """
    import os
    from datetime import datetime
    
    try:
        # 读取音频文件
        audio_bytes = await file.read()
        if len(audio_bytes) < 1000:
            raise HTTPException(status_code=400, detail="音频文件过小")
        
        # 解析候选说话人
        candidate_ids = [x.strip() for x in speaker_ids.split(",") if x.strip()] if speaker_ids else None
        
        # 构建患者基本信息
        patient_basic_parts = []
        if patient_name:
            patient_basic_parts.append(f"姓名：{patient_name}")
        if patient_gender:
            patient_basic_parts.append(f"性别：{patient_gender}")
        if patient_age:
            patient_basic_parts.append(f"年龄：{patient_age}岁")
        patient_basic = "、".join(patient_basic_parts) if patient_basic_parts else "未提供"
        
        logger.info(f"患者基本信息: {patient_basic}")
        
        # 保存音频文件
        audio_save_dir = "data/recordings"
        os.makedirs(audio_save_dir, exist_ok=True)
        
        # 生成文件名：患者姓名拼音_时间戳.wav
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if patient_name and HAS_PYPINYIN:
            # 中文转拼音，如 "张三" -> "zhangsan"
            pinyin_name = "".join(lazy_pinyin(patient_name))
        elif patient_name:
            # 没有 pypinyin，用原名（过滤非法字符）
            pinyin_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-'))
        else:
            pinyin_name = "unknown"
        audio_filename = f"{pinyin_name}_{timestamp}.wav"
        audio_path = os.path.join(audio_save_dir, audio_filename)
        
        logger.info(f"保存录音文件: {audio_filename}, 患者: {patient_name or '未知'}")
        
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        logger.info(f"音频已保存: {audio_path}")
        
        # 调用对话识别服务
        logger.info("开始处理医患对话音频...")
        segments = voiceprint_service.diarize_and_transcribe(candidate_ids, audio_bytes)
        
        if not segments:
            return MedicalRecordResponse(
                success=False,
                conversation=[],
                error="未能识别出对话内容"
            )
        
        logger.info(f"识别出 {len(segments)} 个对话片段")
        
        # 格式化对话内容
        conversation_text = format_conversation(segments)
        logger.info(f"对话内容:\n{conversation_text}")
        
        # 调用大模型生成病历
        logger.info("调用大模型生成电子病历...")
        prompt = MEDICAL_RECORD_PROMPT.format(
            patient_basic=patient_basic,
            conversation=conversation_text
        )
        llm_response = call_ollama(prompt)
        
        if not llm_response:
            return MedicalRecordResponse(
                success=False,
                conversation=segments,
                error="大模型调用失败，请检查 Ollama 服务是否运行"
            )
        
        # 解析病历
        medical_record = parse_medical_record(llm_response)
        
        # 生成病历ID：姓名拼音_日期时间
        if patient_name and HAS_PYPINYIN:
            pinyin_name = "".join(lazy_pinyin(patient_name))
        elif patient_name:
            pinyin_name = "".join(c for c in patient_name if c.isalnum() or c in ('_', '-'))
        else:
            pinyin_name = "unknown"
        record_id = f"{pinyin_name}_{timestamp}"
        
        save_medical_record(
            record_id=record_id,
            patient_name=patient_name,
            patient_gender=patient_gender,
            patient_age=patient_age,
            conversation=segments,
            medical_record=medical_record,
            audio_filename=audio_filename
        )
        
        return MedicalRecordResponse(
            success=True,
            conversation=segments,
            medical_record=medical_record,
            raw_response=llm_response if not medical_record else None,
            record_id=record_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"生成电子病历失败: {e}")
        return MedicalRecordResponse(
            success=False,
            conversation=[],
            error=str(e)
        )


@router.post("/analyze-conversation")
async def analyze_conversation(
    conversation: List[dict],
    _: str = Depends(get_authorization_token)
):
    """
    分析已有的对话记录，生成电子病历
    
    适用于实时识别后，对累积的对话进行分析
    """
    try:
        if not conversation:
            raise HTTPException(status_code=400, detail="对话记录为空")
        
        # 格式化对话内容
        conversation_text = format_conversation(conversation)
        
        # 调用大模型
        prompt = MEDICAL_RECORD_PROMPT.format(
            patient_basic="未提供",
            conversation=conversation_text
        )
        llm_response = call_ollama(prompt)
        
        if not llm_response:
            return {
                "success": False,
                "error": "大模型调用失败"
            }
        
        medical_record = parse_medical_record(llm_response)
        
        return {
            "success": True,
            "medical_record": medical_record,
            "raw_response": llm_response if not medical_record else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分析对话失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/medical-records")
async def get_medical_records(_: str = Depends(get_authorization_token)):
    """获取病历历史列表"""
    records = list_medical_records()
    return {
        "success": True,
        "total": len(records),
        "records": records
    }


@router.get("/medical-records/{record_id}")
async def get_medical_record_detail(
    record_id: str,
    _: str = Depends(get_authorization_token)
):
    """获取单个病历详情"""
    record = load_medical_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="病历不存在")
    return {
        "success": True,
        "record": record
    }


@router.delete("/medical-records/{record_id}")
async def delete_medical_record(
    record_id: str,
    _: str = Depends(get_authorization_token)
):
    """删除病历"""
    try:
        filepath = os.path.join(MEDICAL_RECORDS_DIR, f"{record_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"病历已删除: {record_id}")
            return {"success": True, "message": "删除成功"}
        else:
            raise HTTPException(status_code=404, detail="病历不存在")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除病历失败: {e}")
        return {"success": False, "error": str(e)}


class MedicalRecordUpdate(BaseModel):
    """病历更新请求"""
    patient_name: Optional[str] = None
    patient_gender: Optional[str] = None
    patient_age: Optional[str] = None
    medical_record: Optional[dict] = None


@router.put("/medical-records/{record_id}")
async def update_medical_record(
    record_id: str,
    update_data: MedicalRecordUpdate,
    _: str = Depends(get_authorization_token)
):
    """更新病历"""
    try:
        filepath = os.path.join(MEDICAL_RECORDS_DIR, f"{record_id}.json")
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="病历不存在")
        
        # 读取现有病历
        with open(filepath, "r", encoding="utf-8") as f:
            record = json.load(f)
        
        # 更新字段
        if update_data.patient_name is not None:
            record["patient_name"] = update_data.patient_name
        if update_data.patient_gender is not None:
            record["patient_gender"] = update_data.patient_gender
        if update_data.patient_age is not None:
            record["patient_age"] = update_data.patient_age
        if update_data.medical_record is not None:
            record["medical_record"] = update_data.medical_record
        
        # 记录更新时间
        record["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        logger.info(f"病历已更新: {record_id}")
        return {"success": True, "message": "更新成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新病历失败: {e}")
        return {"success": False, "error": str(e)}


def generate_pdf(record: dict) -> bytes:
    """生成病历 PDF"""
    if not HAS_REPORTLAB:
        raise HTTPException(status_code=500, detail="PDF 生成库未安装，请安装 reportlab")
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, 
        topMargin=25*mm, bottomMargin=20*mm,
        leftMargin=20*mm, rightMargin=20*mm
    )
    
    # 尝试注册中文字体
    font_name = "Helvetica"
    try:
        font_paths = [
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        for fp in font_paths:
            if os.path.exists(fp):
                pdfmetrics.registerFont(TTFont("Chinese", fp))
                font_name = "Chinese"
                break
    except Exception as e:
        logger.warning(f"注册中文字体失败: {e}")
    
    styles = getSampleStyleSheet()
    
    # 标题样式
    title_style = ParagraphStyle(
        'Title', fontName=font_name, fontSize=20, alignment=1,
        spaceAfter=20, textColor=colors.HexColor('#0066cc')
    )
    # 副标题（患者信息）
    subtitle_style = ParagraphStyle(
        'Subtitle', fontName=font_name, fontSize=12, alignment=1,
        spaceAfter=8, textColor=colors.HexColor('#333333')
    )
    # 章节标题
    heading_style = ParagraphStyle(
        'Heading', fontName=font_name, fontSize=13, 
        spaceBefore=16, spaceAfter=8, 
        textColor=colors.HexColor('#0066cc'),
        borderPadding=(0, 0, 4, 0),
        leftIndent=0
    )
    # 正文
    normal_style = ParagraphStyle(
        'Normal', fontName=font_name, fontSize=11, 
        leading=20, spaceBefore=4, spaceAfter=6,
        leftIndent=10
    )
    # 对话样式
    dialog_doctor_style = ParagraphStyle(
        'DialogDoctor', fontName=font_name, fontSize=10,
        leading=16, spaceBefore=2, spaceAfter=2,
        leftIndent=10, textColor=colors.HexColor('#28a745')
    )
    dialog_patient_style = ParagraphStyle(
        'DialogPatient', fontName=font_name, fontSize=10,
        leading=16, spaceBefore=2, spaceAfter=2,
        leftIndent=10, textColor=colors.HexColor('#fd7e14')
    )
    
    elements = []
    
    # 获取患者姓名和时间
    patient_name = record.get("patient_name", "未知")
    created_at = record.get("created_at", "")
    
    # 标题：XX的就诊记录
    title_text = f"{patient_name}的就诊记录"
    elements.append(Paragraph(title_text, title_style))
    
    # 患者基本信息行
    info_parts = []
    if record.get("patient_gender"):
        info_parts.append(f"性别：{record['patient_gender']}")
    if record.get("patient_age"):
        info_parts.append(f"年龄：{record['patient_age']}岁")
    if created_at:
        info_parts.append(f"就诊时间：{created_at}")
    
    if info_parts:
        elements.append(Paragraph("  |  ".join(info_parts), subtitle_style))
    
    # 分隔线
    elements.append(Spacer(1, 10))
    from reportlab.platypus import HRFlowable
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
    elements.append(Spacer(1, 10))
    
    medical = record.get("medical_record", {})
    
    def is_valid(val):
        if not val:
            return False
        empty = ['未提及', '对话中未提及', '无', '暂无', '-']
        return val.strip() not in empty
    
    # 患者病情
    patient = medical.get("patient_info", {})
    patient_items = []
    
    if is_valid(patient.get("chief_complaint")):
        patient_items.append(f"<b>主诉：</b>{patient['chief_complaint']}")
    if is_valid(patient.get("present_illness")):
        patient_items.append(f"<b>现病史：</b>{patient['present_illness']}")
    if is_valid(patient.get("past_history")):
        patient_items.append(f"<b>既往史：</b>{patient['past_history']}")
    if is_valid(patient.get("personal_history")):
        patient_items.append(f"<b>个人史：</b>{patient['personal_history']}")
    
    if patient_items:
        elements.append(Paragraph("▎患者病情", heading_style))
        for item in patient_items:
            elements.append(Paragraph(item, normal_style))
    
    # 医生评估
    doctor = medical.get("doctor_assessment", {})
    doctor_items = []
    
    if is_valid(doctor.get("physical_exam")):
        doctor_items.append(f"<b>体格检查：</b>{doctor['physical_exam']}")
    if is_valid(doctor.get("preliminary_diagnosis")):
        doctor_items.append(f"<b>初步诊断：</b>{doctor['preliminary_diagnosis']}")
    if is_valid(doctor.get("differential_diagnosis")):
        doctor_items.append(f"<b>鉴别诊断：</b>{doctor['differential_diagnosis']}")
    if is_valid(doctor.get("treatment_plan")):
        doctor_items.append(f"<b>治疗方案：</b>{doctor['treatment_plan']}")
    if is_valid(doctor.get("follow_up")):
        doctor_items.append(f"<b>随访建议：</b>{doctor['follow_up']}")
    
    if doctor_items:
        elements.append(Paragraph("▎医生评估", heading_style))
        for item in doctor_items:
            elements.append(Paragraph(item, normal_style))
    
    # 病历摘要
    if is_valid(medical.get("summary")):
        elements.append(Paragraph("▎病历摘要", heading_style))
        elements.append(Paragraph(medical['summary'], normal_style))
    
    # 对话记录
    conversation = record.get("conversation", [])
    if conversation:
        elements.append(Spacer(1, 10))
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e0e0e0')))
        elements.append(Paragraph("▎对话记录", heading_style))
        # 获取患者姓名用于显示
        patient_display_name = patient_name if patient_name else "患者/家属"
        for seg in conversation:
            text = seg.get("text", "")
            if text:
                speaker_id = seg.get("speaker_id")
                if speaker_id:
                    # 医生显示声纹ID
                    elements.append(Paragraph(f"[{speaker_id}] {text}", dialog_doctor_style))
                else:
                    # 患者显示患者姓名
                    elements.append(Paragraph(f"[{patient_display_name}] {text}", dialog_patient_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


@router.get("/medical-records/{record_id}/pdf")
async def export_medical_record_pdf(
    record_id: str,
    _: str = Depends(get_authorization_token)
):
    """导出病历为 PDF"""
    record = load_medical_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="病历不存在")
    
    try:
        pdf_bytes = generate_pdf(record)
        
        # 生成文件名：姓名的就诊记录_时间.pdf
        patient_name = record.get("patient_name", "未知")
        created_at = record.get("created_at", "")
        
        # 从 created_at 提取日期时间（格式：2025-12-16 11:21:04）
        if created_at:
            time_str = created_at.replace("-", "").replace(":", "").replace(" ", "_")
        else:
            time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 文件名用拼音避免编码问题
        if patient_name and HAS_PYPINYIN:
            name_pinyin = "".join(lazy_pinyin(patient_name))
        else:
            name_pinyin = "unknown"
        
        filename = f"{name_pinyin}_jiuzhen_{time_str}.pdf"
        
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"生成 PDF 失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成 PDF 失败: {str(e)}")
