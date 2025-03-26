import os
import yt_dlp
import re
import subprocess
import json
import shutil
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# 환경 변수 불러오기
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 폴더 구조 설정
BASE_DIR = "youtube_summarizer"
DOWNLOADS_DIR = os.path.join(BASE_DIR, "downloaded_videos")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_clips")

# 필요한 디렉토리 생성
def create_directories():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(DOWNLOADS_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

# 비디오 ID로 작업 디렉토리 생성
def create_video_directories(video_id):
    video_dir = os.path.join(GENERATED_DIR, video_id)
    os.makedirs(video_dir, exist_ok=True)
    return video_dir

# 영상 ID 기반으로 이미 처리된 영상인지 확인하는 함수
def is_video_already_processed(video_id):
    # 다운로드된 비디오 확인
    video_path = os.path.join(DOWNLOADS_DIR, f"{video_id}.mp4")
    if not os.path.exists(video_path):
        return False, None, None
    
    # 생성된 요약 영상 확인
    video_dir = os.path.join(GENERATED_DIR, video_id)
    summary_path = os.path.join(video_dir, f"{video_id}_summary.mp4")
    segments_json = os.path.join(video_dir, f"{video_id}_segments.json")
    
    if os.path.exists(summary_path) and os.path.exists(segments_json):
        # 세그먼트 정보 로드
        with open(segments_json, 'r', encoding='utf-8') as f:
            segments_data = json.load(f)
        return True, summary_path, segments_data
    
    return True, None, None  # 다운로드는 되었지만 요약은 아직

# URL에서 비디오 ID 추출
def extract_video_id(youtube_url):
    if "youtube.com" in youtube_url:
        video_id = youtube_url.split("v=")[1].split("&")[0] if "v=" in youtube_url else None
    elif "youtu.be" in youtube_url:
        video_id = youtube_url.split("/")[-1].split("?")[0]
    else:
        raise ValueError("유효한 YouTube URL이 아닙니다.")
    
    if not video_id:
        raise ValueError("YouTube 비디오 ID를 추출할 수 없습니다.")
    
    return video_id

# Youtube 영상 다운로드 함수 수정
def download_youtube(youtube_url):
    # 필요한 디렉토리 생성
    create_directories()
    
    # URL에서 비디오 ID 추출
    video_id = extract_video_id(youtube_url)
    
    # 이미 처리된 영상인지 확인
    already_downloaded, summary_path, segments_data = is_video_already_processed(video_id)
    if already_downloaded:
        print(f"이미 다운로드된 영상입니다: {video_id}")
        video_path = os.path.join(DOWNLOADS_DIR, f"{video_id}.mp4")
        return video_id, video_path, summary_path, segments_data
    
    # yt-dlp 옵션 설정
    ydl_opts = {
        'format': 'best',
        'outtmpl': os.path.join(DOWNLOADS_DIR, '%(id)s.%(ext)s'),
        'socket_timeout': 60,
        'retries': 10,
        'fragment_retries': 10,
        'skip_unavailable_fragments': True,
    }
    
    try:
        # yt-dlp를 사용하여 영상 다운로드
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_id = info_dict.get('id')
            ext = info_dict.get('ext', 'mp4')
            filename = os.path.join(DOWNLOADS_DIR, f"{video_id}.{ext}")
            print(f"다운로드 완료: {filename}")
            
            # 비디오 디렉토리 생성
            create_video_directories(video_id)
            
            return video_id, filename, None, None
    except Exception as e:
        print(f"다운로드 오류: {e}")
        # 이미 ID를 알고 있다면 자막만 처리
        if video_id:
            return video_id, None, None, None
        raise

# 유튜브 자막 가져오기 함수
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"자막 가져오기 오류: {e}")
        return None

# 비디오 세그먼트를 위한 데이터 모델 정의
class Segment(BaseModel):
    start_time: float = Field(description="세그먼트 시작 시간(초)")
    end_time: float = Field(description="세그먼트 종료 시간(초)")
    yt_title: str = Field(description="이 세그먼트를 바이럴 서브토픽으로 만들기 위한 유튜브 제목")
    description: str = Field(description="이 세그먼트를 바이럴로 만들기 위한 상세 설명")
    duration: float = Field(description="세그먼트 길이(초)")  # int에서 float로 변경

# 영상 유형 분류 함수
def classify_video_type(transcript):
    # LLM 설정
    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    
    # 영상 유형 분류를 위한 프롬프트
    classification_prompt = f"""다음은 YouTube 영상의 자막입니다. 이 자막을 분석하여 영상이 '강의' 유형인지 '스토리텔링' 유형인지 판단해주세요.

    '강의' 유형의 특징:
    - 특정 주제나 개념에 대한 설명이 중심
    - 정보 전달이 주요 목적
    - 체계적인 구조와 논리적 흐름
    - 전문 용어나 학술적 표현 사용
    - "오늘 배울 내용은...", "이 개념의 핵심은..." 등의 표현

    '스토리텔링' 유형의 특징:
    - 이야기, 경험, 사건 중심의 내용
    - 감정적 요소와 개인적 경험 포함
    - 시간 순서나 인과 관계에 따른 전개
    - 등장인물이나 상황 묘사가 풍부
    - "그때 내가...", "이런 일이 있었어요" 등의 표현

    다음 자막을 분석하여 '강의' 또는 '스토리텔링' 중 하나로만 응답해주세요.

    자막:
    {transcript}

    영상 유형(강의 또는 스토리텔링):"""

    # LLM 호출
    response = llm.invoke(classification_prompt)
    
    # 응답 분석
    response_text = response.content.strip().lower()
    print(f"{response_text=}")
    print(f"{transcript=}")
    if "강의" in response_text:
        return "lecture"
    elif "스토리텔링" in response_text:
        return "storytelling"
    else:
        # 기본값으로 강의 유형 반환
        return "lecture"

# 강의 유형을 위한 프롬프트 템플릿
lecture_prompt_template = """다음은 유튜브 동영상의 전체 자막입니다. 이 자막에서 전체 내용을 이해하는 데 **핵심적인 문맥이나 논리 흐름을 포함하는 전체의 10% 이내의 구간**만을 추출해주세요. 

- 반복되거나 의미가 중복되는 부분은 제외해주세요.
- 전체 내용의 흐름이 자연스럽게 이어질 수 있도록 핵심 구간을 선택해주세요.
- 가능한 한 요약에 필요한 정보만 포함하고, 불필요한 잡담, 인사말, 예시 반복은 생략해주세요.


응답은 다음 JSON 형식으로 제공해주세요:
{{
    "segments": [
        {{
            "start_time": float,  // 구간 시작 시간(초)
            "end_time": float,    // 구간 종료 시간(초)
            "yt_title": string,   // 교육적 제목 (학습 내용 중심)
            "description": string, // 학습 포인트 설명
            "duration": float     // 구간 길이(초)
        }},
        // 추가 구간...
    ]
}}

중요: 각 구간의 시작과 끝은 반드시 자연스러운 문장 경계나 주제 전환점에서 설정해주세요. 
문장이나 단어가 중간에 잘리지 않도록 해주세요.

다음은 분석할 강의 자막입니다:
{transcript}"""

# 스토리텔링 유형 영상을 위한 프롬프트 템플릿
storytelling_prompt_template = """다음은 유튜브 동영상의 전체 자막입니다. 이 자막에서 전체 내용을 이해하는 데 **핵심적인 문맥이나 논리 흐름을 포함하는 전체의 10% 이내의 구간**만을 추출해주세요. 

- 반복되거나 의미가 중복되는 부분은 제외해주세요.
- 전체 내용의 흐름이 자연스럽게 이어질 수 있도록 핵심 구간을 선택해주세요.
- 가능한 한 요약에 필요한 정보만 포함하고, 불필요한 잡담, 인사말, 예시 반복은 생략해주세요.


응답은 다음 JSON 형식으로 제공해주세요:
{{
    "segments": [
        {{
            "start_time": float,  // 구간 시작 시간(초)
            "end_time": float,    // 구간 종료 시간(초)
            "yt_title": string,   // 매력적인 장면 제목
            "description": string, // 장면 설명 및 스토리적 중요성
            "duration": float     // 구간 길이(초)
        }},
        // 추가 구간...
    ]
}}

중요: 각 구간의 시작과 끝은 반드시 자연스러운 대화 경계나 장면 전환점에서 설정해주세요. 
문장이나 대화가 중간에 잘리지 않도록 해주세요.

다음은 분석할 스토리텔링 영상의 자막입니다:
{transcript}"""

# 유튜브 자막 요약 함수
def summarize_transcript(transcript, video_id, summary_ratio=None, segment_count=None):
    try:
        # 영상 유형 분류
        video_type = classify_video_type(transcript)
        
        # 프롬프트에 추가할 요구사항 생성
        additional_requirements = ""
        if summary_ratio is not None:
            additional_requirements += f"\n- 전체 요약 영상 길이가 원본의 {summary_ratio:.1%}가 되도록 구간을 선택해주세요."
        if segment_count is not None:
            additional_requirements += f"\n- 정확히 {segment_count}개의 구간으로 나누어 주세요."
        
        # 기본 프롬프트 템플릿에 추가 요구사항 삽입
        if video_type == "lecture":
            prompt_template = lecture_prompt_template.replace(
                "기술적 요구사항:",
                f"기술적 요구사항:{additional_requirements}"
            )
        else:
            prompt_template = storytelling_prompt_template.replace(
                "기술적 요구사항:",
                f"기술적 요구사항:{additional_requirements}"
            )
        
        # 프롬프트 생성 및 나머지 처리
        prompt = prompt_template.format(transcript=transcript)
        
        # LLM 설정
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        # LLM 호출
        response = llm.invoke(prompt)
        
        # 응답 파싱
        try:
            # JSON 부분 추출
            content = response.content
            json_str = content
            
            # JSON 형식이 아닌 텍스트가 포함된 경우 JSON 부분만 추출
            if not content.strip().startswith('{'):
                import re
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
            
            # JSON 파싱
            parsed_data = json.loads(json_str)
            segments_data = parsed_data.get('segments', [])
            
            # 비디오 디렉토리 경로
            video_dir = os.path.join(GENERATED_DIR, video_id)
            
            # 세그먼트 정보 저장
            segments_json_path = os.path.join(video_dir, f"{video_id}_segments.json")
            with open(segments_json_path, 'w', encoding='utf-8') as f:
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            
            # Segment 객체 리스트로 변환
            segments = []
            for item in segments_data:
                # 종료 시간에만 버퍼 추가
                segment = Segment(
                    start_time=item["start_time"],
                    end_time=item["end_time"] + 0.5,
                    yt_title=item["yt_title"],
                    description=item["description"],
                    duration=float(item["duration"]) + 0.5 if isinstance(item["duration"], (int, float)) else 
                              float(item["end_time"] - item["start_time"]) + 0.5
                )
                segments.append(segment)
            
            # 결과 출력
            for segment in segments:
                print(f"제목: {segment.yt_title}")
                print(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                print(f"설명: {segment.description}")
                print(f"길이: {segment.duration}초")
                print("---")
            
            return segments
        except Exception as e:
            print(f"파싱 오류: {e}")
            import traceback
            print(traceback.format_exc())
            # 빈 리스트 반환
            return []
    except Exception as e:
        print(f"영상 유형 분류 오류: {e}")
        return []

# 비디오 요약 함수
def summarize_video(segments, video_id, video_path):
    if not segments:
        return None, None
    
    # 비디오 디렉토리 경로
    video_dir = os.path.join(GENERATED_DIR, video_id)
    
    # 클립 목록을 저장할 텍스트 파일 경로
    concat_txt_path = os.path.join(video_dir, f"{video_id}_concat.txt")
    
    # 최종 병합된 비디오 경로
    merged_video_path = os.path.join(video_dir, f"{video_id}_summary.mp4")
    
    # 클립 목록 파일 생성
    with open(concat_txt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            # 클립 파일 경로
            clip_path = os.path.join(video_dir, f"{video_id}_clip_{i+1}.mp4")
            
            # 세그먼트 라벨 파일 경로
            label_path = os.path.join(video_dir, f"{video_id}_label_{i+1}.txt")
            
            # 세그먼트 라벨 저장
            with open(label_path, "w", encoding="utf-8") as label_file:
                label_file.write(f"제목: {segment.yt_title}\n")
                label_file.write(f"설명: {segment.description}\n")
                label_file.write(f"시간: {segment.start_time}초 ~ {segment.end_time}초\n")
                label_file.write(f"길이: {segment.duration}초\n")
            
            # 시작 및 종료 시간
            start_time = segment.start_time
            end_time = segment.end_time
            
            # FFmpeg를 사용하여 클립 추출
            clip_command = f'ffmpeg -y -ss {start_time} -to {end_time} -i "{video_path}" -c:v libx264 -c:a aac -avoid_negative_ts 1 "{clip_path}"'
            
            try:
                subprocess.run(clip_command, shell=True, check=True)
                # 클립 파일 경로를 concat 파일에 추가 - 상대 경로 사용
                # 중요: 여기서 절대 경로가 아닌 상대 경로를 사용해야 함
                f.write(f"file '{os.path.basename(clip_path)}'\n")
            except subprocess.CalledProcessError as e:
                st.error(f"클립 추출 오류: {e}")
    
    return concat_txt_path, merged_video_path

# 클립 병합 함수
def merge_clips(concat_txt_path, merged_video_path, video_id, segments):
    if not concat_txt_path or not os.path.exists(concat_txt_path):
        return None
    
    # FFmpeg를 사용하여 클립 병합
    merge_command = f'ffmpeg -y -f concat -safe 0 -i "{concat_txt_path}" -c copy "{merged_video_path}"'
    
    try:
        subprocess.run(merge_command, shell=True, check=True)
        print(f"클립 병합 완료: {merged_video_path}")
        return merged_video_path
    except subprocess.CalledProcessError as e:
        print(f"클립 병합 오류: {e}")
        return None 


# 메인 함수
def main():
    st.set_page_config(page_title="YouTube AI Summarizer", layout="centered")
    
    # CSS 스타일 적용
    st.markdown("""
        <style>
        .main {
            background-color: #f5f5f5;
        }
        .stApp {
            max-width: 800px;  # 최대 너비를 800px로 제한
            margin: 0 auto;
        }
        h1 {
            color: #1e3a8a;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3b5cb8;
        }
        .solution-box {
            background-color: #e6f3ff;
            padding: 1rem;
            border-radius: 5px;
            margin-top: 1rem;
            border-left: 5px solid #1e3a8a;
        }
        .stTextArea>div>div>textarea {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
        }
        .block-container {
            max-width: 800px;  # 컨테이너 최대 너비도 제한
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        "<img src='https://raw.githubusercontent.com/kimstitute/Medical_AI/main/sangsang2.png' \
        style='height: 60px; margin-right: 10px; margin-bottom: 18px;' alt='bugi'>"
        "<span style='font-size: 40px; font-weight: bold;'>Youtube AI Summarizer</span> "
        "<span style='font-size: 10px; color: gray;'>Created by HSU AI</span>",
        unsafe_allow_html=True
    )

    # YouTube URL 입력
    subject = st.text_area("Youtube URL:", height=10)

    # 두 개의 컬럼 생성
    col1, col2 = st.columns(2)

    # 축약 비율 입력 (0~1 사이)
    with col1:
        summary_ratio = st.number_input(
            "축약 비율 (0~1 사이, 빈칸은 자동)",
            min_value=0.0,
            max_value=1.0,
            value=None,
            help="원본 영상 길이의 몇 배로 축약할지 설정 (예: 0.3 = 30%)"
        )

    # 구간 개수 입력
    with col2:
        segment_count = st.number_input(
            "구간 개수 (빈칸은 자동)",
            min_value=1,
            max_value=10,
            value=None,
            help="몇 개의 구간으로 나눌지 설정"
        )

    if st.button("Start"):
        with st.spinner("now generating..."):
            try:
                # 필요한 디렉토리 생성
                create_directories()
                
                # YouTube URL에서 비디오 ID 추출
                if "youtube.com" in subject or "youtu.be" in subject:
                    try:
                        # URL에서 비디오 ID 추출
                        if "youtube.com" in subject:
                            video_id = subject.split("v=")[1].split("&")[0] if "v=" in subject else None
                        else:  # youtu.be 형식
                            video_id = subject.split("/")[-1].split("?")[0]
                        
                        if not video_id:
                            st.error("YouTube 비디오 ID를 추출할 수 없습니다.")
                            return
                        
                        # 비디오 디렉토리 생성
                        video_dir = create_video_directories(video_id)
                        
                        # 이미 처리된 영상인지 확인
                        video_path = os.path.join(DOWNLOADS_DIR, f"{video_id}.mp4")
                        summary_path = os.path.join(video_dir, f"{video_id}_summary.mp4")
                        segments_json = os.path.join(video_dir, f"{video_id}_segments.json")
                        
                        # 이미 요약 영상이 있는 경우
                        if os.path.exists(summary_path) and os.path.exists(segments_json):
                            st.success("이미 처리된 영상입니다. 기존 요약 영상을 표시합니다.")
                            st.markdown("### 생성된 영상")
                            st.video(summary_path)
                            
                            # 세그먼트 정보 로드 및 표시
                            with open(segments_json, 'r', encoding='utf-8') as f:
                                segments_data = json.load(f)
                            
                            st.markdown("### 세그먼트 정보")
                            for i, segment in enumerate(segments_data.get('segments', [])):
                                st.markdown(f"**세그먼트 {i+1}**: {segment['yt_title']}")
                                st.markdown(f"시간: {segment['start_time']}초 ~ {segment['end_time']}초")
                                st.markdown(f"설명: {segment['description']}")
                                st.markdown(f"길이: {segment['duration']}초")
                                st.markdown("---")
                            
                            st.markdown(f"<div class='solution-box'>영상 저장 경로: {summary_path}</div>", unsafe_allow_html=True)
                            st.write("※본 영상은 AI에 의해 편집된 요약 영상이며, 부정확할 수도 있습니다. 참고용으로 사용해주세요.")
                            return
                        
                        # 영상 다운로드 (이미 다운로드된 경우 스킵)
                        if not os.path.exists(video_path):
                            # yt-dlp 옵션 설정
                            ydl_opts = {
                                'format': 'best',
                                'outtmpl': os.path.join(DOWNLOADS_DIR, '%(id)s.%(ext)s'),
                                'socket_timeout': 60,
                                'retries': 10,
                                'fragment_retries': 10,
                                'skip_unavailable_fragments': True,
                            }
                            
                            # yt-dlp를 사용하여 영상 다운로드
                            '''with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                                info_dict = ydl.extract_info(subject, download=True)
                                ext = info_dict.get('ext', 'mp4')
                                video_path = os.path.join(DOWNLOADS_DIR, f"{video_id}.{ext}")
                                st.success(f"영상 다운로드 완료: {video_id}")'''
                        else:
                            st.info(f"이미 다운로드된 영상을 사용합니다: {video_id}")
                        
                    except Exception as e:
                        st.warning(f"비디오 다운로드 실패: {e}. 자막만 처리합니다.")
                        if "youtube.com" in subject:
                            video_id = subject.split("v=")[1].split("&")[0]
                        else:  # youtu.be 형식
                            video_id = subject.split("/")[-1].split("?")[0]
                        video_path = None
                        
                        # 비디오 디렉토리 생성
                        video_dir = create_video_directories(video_id)
                    
                    # 자막 가져오기
                    transcript = get_transcript(video_id)
                    
                    if not transcript:
                        st.error("자막을 가져올 수 없습니다. 다른 동영상을 시도해주세요.")
                        return
                    
                    # 자막 요약 및 처리
                    segments = summarize_transcript(transcript, video_id, summary_ratio, segment_count)
                    
                    if not segments:
                        st.error("세그먼트를 추출할 수 없습니다.")
                        return
                    
                    # 세그먼트 정보를 JSON으로 저장
                    segments_data = {
                        "segments": [
                            {
                                "start_time": segment.start_time,
                                "end_time": segment.end_time,
                                "yt_title": segment.yt_title,
                                "description": segment.description,
                                "duration": segment.duration
                            } for segment in segments
                        ]
                    }
                    
                    # JSON 파일로 저장
                    with open(segments_json, 'w', encoding='utf-8') as f:
                        json.dump(segments_data, f, ensure_ascii=False, indent=4)
                    
                    # 다운로드 성공 여부에 따라 다른 처리
                    if video_path and os.path.exists(video_path):
                        # 클립 목록을 저장할 텍스트 파일 경로
                        concat_txt_path = os.path.join(video_dir, f"{video_id}_concat.txt")
                        
                        # 클립 목록 파일 생성
                        with open(concat_txt_path, "w", encoding="utf-8") as f:
                            for i, segment in enumerate(segments):
                                # 클립 파일 경로
                                clip_path = os.path.join(video_dir, f"{video_id}_clip_{i+1}.mp4")
                                
                                # 세그먼트 라벨 파일 경로
                                label_path = os.path.join(video_dir, f"{video_id}_label_{i+1}.txt")
                                
                                # 세그먼트 라벨 저장
                                with open(label_path, "w", encoding="utf-8") as label_file:
                                    label_file.write(f"제목: {segment.yt_title}\n")
                                    label_file.write(f"설명: {segment.description}\n")
                                    label_file.write(f"시간: {segment.start_time}초 ~ {segment.end_time}초\n")
                                    label_file.write(f"길이: {segment.duration}초\n")
                                
                                # 시작 및 종료 시간
                                start_time = segment.start_time
                                end_time = segment.end_time
                                
                                # FFmpeg를 사용하여 클립 추출
                                clip_command = f'ffmpeg -y -ss {start_time} -to {end_time} -i "{video_path}" -c:v libx264 -c:a aac -avoid_negative_ts 1 "{clip_path}"'
                                
                                try:
                                    subprocess.run(clip_command, shell=True, check=True)
                                    # 클립 파일 경로를 concat 파일에 추가 - 상대 경로 사용
                                    # 중요: 여기서 절대 경로가 아닌 상대 경로를 사용해야 함
                                    f.write(f"file '{os.path.basename(clip_path)}'\n")
                                except subprocess.CalledProcessError as e:
                                    st.error(f"클립 추출 오류: {e}")
                        
                        # FFmpeg를 사용하여 클립 병합
                        merge_command = f'ffmpeg -y -f concat -safe 0 -i "{concat_txt_path}" -c copy "{summary_path}"'
                        
                        try:
                            subprocess.run(merge_command, shell=True, check=True)
                            st.success(f"클립 병합 완료: {summary_path}")
                            
                            # 결과 표시
                            st.markdown("### 생성된 영상")
                            st.video(summary_path)
                            
                            # 세그먼트 정보 표시
                            st.markdown("### 세그먼트 정보")
                            for i, segment in enumerate(segments):
                                st.markdown(f"**세그먼트 {i+1}**: {segment.yt_title}")
                                st.markdown(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                                st.markdown(f"설명: {segment.description}")
                                st.markdown(f"길이: {segment.duration}초")
                                st.markdown("---")
                            
                            st.markdown(f"<div class='solution-box'>영상 저장 경로: {summary_path}</div>", unsafe_allow_html=True)
                            st.write("※본 영상은 AI에 의해 편집된 요약 영상이며, 부정확할 수도 있습니다. 참고용으로 사용해주세요.")
                            print("영상 저장이 완료되었습니다.")
                        except subprocess.CalledProcessError as e:
                            st.error(f"클립 병합 오류: {e}")
                    else:
                        # 다운로드 실패 시 자막 요약만 표시
                        st.markdown("### 자막 요약 결과")
                        for i, segment in enumerate(segments):
                            st.markdown(f"**세그먼트 {i+1}**: {segment.yt_title}")
                            st.markdown(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                            st.markdown(f"설명: {segment.description}")
                            st.markdown(f"길이: {segment.duration}초")
                            st.markdown("---")
                else:
                    st.error("유효한 YouTube URL을 입력해주세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
