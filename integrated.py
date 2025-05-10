import os
import yt_dlp
import re
import json
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
from pytube import YouTube
import streamlit.components.v1 as components

from urllib.parse import urlparse, parse_qs
from pytubefix import YouTube
from pytubefix.cli import on_progress
from moviepy.editor import VideoFileClip, concatenate_videoclips  

# 환경 변수 불러오기
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#--------------------------------------------------------------------------
# segments 폴더 생성
SEGMENT_DIR = "segments"
if not os.path.exists(SEGMENT_DIR):
    os.makedirs(SEGMENT_DIR)

# 영상 다운로드 파일 저장 디렉토리 설정
DOWNLOAD_DIR = "downloaded_videos"
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# 세션 초기화: 실행 여부, 영상 다운로드 결과, 자막, 프롬프트, LLM 응답, 병합 결과 저장
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "download_file_name" not in st.session_state:
    st.session_state.download_file_name = None
if "download_video_bytes" not in st.session_state:
    st.session_state.download_video_bytes = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "prompt_template" not in st.session_state:
    st.session_state.prompt_template = None
if "llm_response" not in st.session_state:
    st.session_state.llm_response = None
if "merged_file" not in st.session_state:      
    st.session_state.merged_file = None
if "segments_json" not in st.session_state:      
    st.session_state.segments_json = None


# 자막 관련 함수
def extract_video_id(url):
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname
    if hostname in ('www.youtube.com', 'youtube.com'):
        qs = parse_qs(parsed_url.query)
        return qs.get('v', [None])[0]
    elif hostname == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

def seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fetch_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        return "올바른 유튜브 URL을 입력해주세요."

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['ko'])
            st.info("한국어 원본 자막을 찾았습니다.")
            transcript_data = transcript.fetch()
            
            # 디버깅을 위한 데이터 출력
            #st.write("자막 데이터 형식:", type(transcript_data))
            #if transcript_data:
            #    st.write("첫 번째 자막 데이터 예시:", transcript_data[0])
        except Exception as original_error:
            st.info("한국어 자막이 없으므로 영어 자막을 자동 번역합니다.")
            transcript = transcript_list.find_transcript(['en'])
            transcript = transcript.translate('ko')
            transcript_data = transcript.fetch()

        transcript_lines = []
        for snippet in transcript_data:
            # FetchedTranscriptSnippet 객체의 속성에 직접 접근
            start = snippet.start
            duration = snippet.duration
            text = snippet.text
            
            end = start + duration
            start_str = start #seconds_to_hms(start)
            end_str = end #seconds_to_hms(end)
            transcript_lines.append(f"[{start_str} ~ {end_str}] {text}")
        transcript_text = "\n".join(transcript_lines)
        return transcript_text

    except Exception as e:
        return "자막을 가져오는 중 오류 발생: " + str(e)

# ----- 영상 다운로드 관련 함수 -----
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def get_youtube_video_id(url):
    regex_patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/([\w\-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([\w\-]{11})"
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_video_to_bytes(url):
    video_id = get_youtube_video_id(url)
    if not video_id:
        st.error("유효한 유튜브 URL이 아닙니다.")
        return None, None

    try:
        yt = YouTube(url, on_progress_callback=on_progress)
        st.write("영상 제목:", yt.title)
        stream = yt.streams.get_highest_resolution()
        if stream is None:
            st.error("다운로드 가능한 스트림을 찾을 수 없습니다.")
            return None, None

        raw_file_name = f"{yt.title}.mp4"
        file_name = sanitize_filename(raw_file_name)
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        stream.download(output_path=DOWNLOAD_DIR, filename=file_name)

        with open(file_path, "rb") as file:
            video_bytes = file.read()
        return video_bytes, file_name
    except Exception as e:
        st.error(f"에러 발생: {e}")
        return None, None

def split_video_by_segments():
    llm_response = st.session_state.llm_response
    if not llm_response:
        st.error("LLM 응답이 없습니다.")
        return

    llm_response = llm_response.strip()
    pattern = r"^```(?:json)?\s*(\{.*\})\s*```$"
    match = re.search(pattern, llm_response, re.DOTALL)
    if match:
        pure_json_str = match.group(1)
    else:
        pure_json_str = llm_response

    try:
        segments_dict = json.loads(pure_json_str)
        segments = segments_dict.get("segments", [])
    except json.JSONDecodeError as e:
        st.error("LLM 응답 파싱 중 오류: " + str(e))
        st.write("파싱 대상 문자열:", pure_json_str)
        return

    video_file_path = os.path.join(DOWNLOAD_DIR, st.session_state.download_file_name)
    if not os.path.exists(video_file_path):
        st.error("다운로드된 영상 파일을 찾을 수 없습니다.")
        return

    try:
        video_clip = VideoFileClip(video_file_path)
    except Exception as e:
        st.error("영상을 열 수 없습니다: " + str(e))
        return

    video_duration = video_clip.duration
    output_files = []
    for idx, segment in enumerate(segments):
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        if start_time is None or end_time is None:
            continue

        if end_time > video_duration:
            end_time = video_duration

        try:
            subclip = video_clip.subclip(start_time, end_time)
        except Exception as e:
            st.error(f"Segment {idx+1} 자르는 중 에러: " + str(e))
            continue

        # 파일 이름에 3자리 패딩을 적용하여 정렬 문제 해결 (예: _segment_001.mp4)
        base_name, ext = os.path.splitext(st.session_state.download_file_name)
        output_filename = f"{base_name}_segment_{idx+1:03d}{ext}"
        output_path = os.path.join(SEGMENT_DIR, output_filename)

        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        st.write(f"Segment {idx+1} 저장 완료: {output_filename}")
        output_files.append(output_filename)

    video_clip.close()
    return output_files

def merge_video_segments(segment_dir, output_filename="merged_video.mp4"):
    segment_files = [f for f in os.listdir(segment_dir) if f.endswith(".mp4")]
    if not segment_files:
        st.error("병합할 세그먼트 파일이 없습니다.")
        return None
    # 자연 정렬을 위해 세그먼트 번호 추출 후 정렬
    def sort_key(f):
        m = re.search(r"segment_(\d+)", f)
        return int(m.group(1)) if m else 0

    segment_files = sorted(segment_files, key=sort_key)

    clips = []
    for f in segment_files:
        file_path = os.path.join(segment_dir, f)
        clip = VideoFileClip(file_path)
        clips.append(clip)

    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    final_clip.close()
    for clip in clips:
        clip.close()

    return output_filename


#--------------------------------------------------------------------------



# 유튜브 자막 가져오기 함수
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        return transcript_text
    except Exception as e:
        print(f"자막 가져오기 오류: {e}")
        return None


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
lecture_prompt_template = """다음은 유튜브 동영상의 전체 자막이야.
요약 영상 및 자막은 전체 영상길이의 20%를 넘지 않아야 해,

요약할 때는 다음 기준을 따르세요:
- 각 전환점을 기준으로 핵심 내용과 주요 이슈를 골라 요약해.
- 핵심 문맥은 영상에서 가장 중요한 메시지와 정보, 그리고 논리 흐름을 유지하는 문장들이야.
- 가능한 한 요약에 필요한 정보만 포함하고, 불필요한 잡담, 인사말, 예시 반복은 생략해줘.
- 데이터는 유튜브 자막으로, 타임스탬프가 포함되어 있어.
- 원하는 요청사항: 이 전체 자막을 20% 이내 분량으로 줄이고 싶어.
- 필요한 내용들의 타임스탬프만 추출하여, 그 총합이 전체 영상 길이의 20%를 넘지 않아야 해.
- 연속적인 구간끼리 자연스럽게 묶을 수 있지만, 개별 구간에 중요한 내용이 있다면 반드시 묶어야 하는 것은 아니야. 
각 구간이 개별적으로 중요한 경우 그대로 선택할 수 있도록 유연하게 고려해.
- 추출된 구간들은 자연스럽게 이어지는 흐름으로 표시해야 해.

응답은 다음 JSON 형식으로 제공해주세요:
{{
    "segments": [
        {{
            "start_time": float,  // 구간 시작 시간 (초)
            "end_time": float,    // 구간 종료 시간 (초)
            "duration": float,    // 구간 길이 (초)
            "content": str        // 해당 구간의 핵심 내용에 대한 간결한 설명
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
            "start_time": float,  // 구간 시작 시간 (초)
            "end_time": float,    // 구간 종료 시간 (초)
            "duration": float,    // 구간 길이 (초)
            "content": str        // 해당 구간의 핵심 내용에 대한 간결한 설명
        }},
        // 추가 구간...
    ]
}}

중요: 각 구간의 시작과 끝은 반드시 자연스러운 대화 경계나 장면 전환점에서 설정해주세요. 
문장이나 대화가 중간에 잘리지 않도록 해주세요.

다음은 분석할 스토리텔링 영상의 자막입니다:
{transcript}"""


# 비디오 세그먼트를 위한 데이터 모델 정의
class Segment(BaseModel):
    start_time: float = Field(description="세그먼트 시작 시간(초)")
    end_time: float = Field(description="세그먼트 종료 시간(초)")
    duration: float = Field(description="세그먼트 길이(초)")  # int에서 float로 변경
    content: str = Field(description="해당 구간의 핵심 내용에 대한 간결한 설명")
    


# 유튜브 자막 요약 함수
def summarize_transcript(transcript, video_id, summary_ratio=None, segment_count=None):
    try:
        # 영상 유형 분류
        video_type = classify_video_type(transcript)
        print(f"{video_type=}")

        # 프롬프트에 추가할 요구사항 생성
        additional_requirements = ""
        if summary_ratio is not None:
            additional_requirements += f"\n- 전체 요약 영상 길이가 원본의 {summary_ratio}%가 되도록 구간을 선택해주세요."
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
              
            # Segment 객체 리스트로 변환
            segments = []
            for item in segments_data:
                # 종료 시간에만 버퍼 추가
                segment = Segment(
                    start_time=item["start_time"],
                    end_time=item["end_time"] + 0.5,
                    duration=float(item["duration"]) + 0.5 if isinstance(item["duration"], (int, float)) else 
                              float(item["end_time"] - item["start_time"]) + 0.5,
                    content=item["content"]
                )
                segments.append(segment)
            
            # 결과 출력
            for segment in segments:
                print(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                print(f"설명: {segment.content}")
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
    

def get_video_duration(url):
    try:
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)  # 영상 길이(초)
            if duration == 0:
                print("영상 길이를 가져올 수 없습니다.")
                return None
            return duration
    except Exception as e:
        print(f"영상 정보 가져오기 오류: {e}")
        return None


# 메인 함수
def main():
    st.set_page_config(page_title="YouTube AI Summarizer", layout="centered")
    
    # 사이드바 메뉴 추가
    with st.sidebar:
        st.markdown("### 🎯 메뉴 선택")
        menu = st.radio(
            "원하는 요약 방식을 선택하세요",
            ["📝 스트리밍 방식", "🎬 다운로드 방식"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ 사용 방법")
        st.markdown("""
        1. 영상 URL을 입력하세요
        2. 축약 비율을 설정하세요
        3. Start 버튼을 클릭하세요
        """)
    
    # CSS 스타일 적용
    st.markdown("""
        <style>
            .stTextArea textarea {
                background-color: white;
                font-size: 16px;
                line-height: 1.5;
            }
            .solution-box {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                line-height: 1.5;
                color: black;
                max-height: 300px;
                overflow-y: auto;
            }
            /* 사이드바 스타일링 */
            .css-1d391kg {
                padding: 1rem;
            }
            .sidebar .sidebar-content {
                background-color: #f8f9fa;
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
    subject = st.text_area("Youtube URL:", height=68)

    # 단일 컨테이너 생성
    with st.container():
        summary_ratio = st.number_input(
            "축약 비율 (1-100% 빈칸은 자동)",
            min_value=1,
            max_value=100,
            value=None,
            help="원본 영상 길이의 몇 %로 축약할지 설정 (예: 30 = 30%)"
        )

    if st.button("Start"):
        with st.spinner("now generating..."):
            if menu == "📝 스트리밍 방식":
                try:
                    # YouTube URL에서 비디오 ID 추출
                    if "youtube.com" in subject or "youtu.be" in subject:
                        # URL에서 비디오 ID 추출
                        if "youtube.com" in subject:
                            video_id = subject.split("v=")[1].split("&")[0] if "v=" in subject else None
                        else:  # youtu.be 형식
                            video_id = subject.split("/")[-1].split("?")[0]
                        
                        if not video_id:
                            st.error("YouTube 비디오 ID를 추출할 수 없습니다.")
                            return
                        
                        # 자막 가져오기
                        #transcript = get_transcript(video_id)
                        transcript = fetch_transcript(subject)

                        if not transcript:
                            st.error("자막을 가져올 수 없습니다. 다른 동영상을 시도해주세요.")
                            return
                        
                        # 자막 요약 및 처리
                        segments = summarize_transcript(transcript, video_id, summary_ratio)
                        
                        if not segments:
                            st.error("세그먼트를 추출할 수 없습니다.")
                            return
                        
                        # 원본 영상 길이 불러오기
                        video_duration = get_video_duration(subject)
                        
                        if video_duration is None:
                            st.error("영상 길이를 가져올 수 없습니다.")
                            return
                        
                        st.info(f"영상 길이: {video_duration}초")
                        
                        # 유효한 세그먼트만 필터링하고 딕셔너리로 변환
                        filtered_segments = []
                        for segment in segments:
                            start_time = min(segment.start_time, video_duration - 1)
                            end_time = min(segment.end_time, video_duration)
                            if start_time < end_time:
                                filtered_segments.append({
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "content": segment.content,
                                    "duration": segment.duration
                                })
                        
                        # 세그먼트 정보를 JSON으로 변환
                        segments_json = json.dumps(filtered_segments)
                        st.session_state.segments_json = segments_json

                        if filtered_segments:    
                            # YouTube 플레이어와 컨트롤러 HTML 생성
                            player_html = f"""
                            <div style="margin-bottom: 20px;">
                            <iframe
                                id="player"
                                width="100%"
                                height="400"
                                src="https://www.youtube.com/embed/{video_id}?enablejsapi=1"
                                frameborder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope"
                                allowfullscreen
                            ></iframe>
                            </div>

                            <script>
                            let segments = {segments_json};
                            let currentSegment = 0;
                            let player;
                            let segmentTimeout = null;

                            // YouTube API가 준비되면 호출
                            function onYouTubeIframeAPIReady() {{
                                if (!player) {{
                                    player = new YT.Player('player', {{
                                        events: {{
                                            'onReady': onPlayerReady,
                                            'onStateChange': onPlayerStateChange
                                        }}
                                    }});
                                }}
                            }}

                            function onPlayerReady(event) {{
                                playSegment(currentSegment);
                            }}

                            function onPlayerStateChange(event) {{
                                if (event.data === YT.PlayerState.PLAYING) {{
                                    clearTimeout(segmentTimeout);
                                    const segment = segments[currentSegment];
                                    const duration = (segment.end_time - segment.start_time) * 1000;
                                    segmentTimeout = setTimeout(() => {{
                                        if (currentSegment < segments.length - 1) {{
                                            currentSegment++;
                                            playSegment(currentSegment);
                                        }} else {{
                                            player.pauseVideo();
                                        }}
                                    }}, duration);
                                }}
                            }}

                            function playSegment(index) {{
                                currentSegment = index;
                                const segment = segments[currentSegment];
                                player.seekTo(segment.start_time);
                                player.playVideo();
                            }}

                            function playPreviousSegment() {{
                                if (currentSegment > 0) {{
                                    playSegment(currentSegment - 1);
                                }}
                            }}

                            function playNextSegment() {{
                                if (currentSegment < segments.length - 1) {{
                                    playSegment(currentSegment + 1);
                                }}
                            }}

                            // 컨트롤 버튼을 동적으로 생성
                            window.addEventListener('DOMContentLoaded', (event) => {{
                                if (!document.getElementById('segment-controls')) {{
                                    const controls = document.createElement('div');
                                    controls.id = 'segment-controls';
                                    controls.style = 'margin-top: 10px; text-align: center;';
                                    controls.innerHTML = `
                                        <button onclick='playPreviousSegment()' style='margin: 5px;'>이전 세그먼트</button>
                                        <button onclick='playNextSegment()' style='margin: 5px;'>다음 세그먼트</button>
                                    `;
                                    document.body.appendChild(controls);
                                }}
                            }});

                            </script>
                            <script src="https://www.youtube.com/iframe_api"></script>
                            """
                            
                            # 세그먼트 정보 표시
                            st.markdown("### 세그먼트 정보")
                            with st.expander("📋 세그먼트 정보 보기"):
                                for i, segment in enumerate(segments):
                                    st.markdown(f"**세그먼트 {i+1}**")
                                    st.markdown(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                                    st.markdown(f"설명: {segment.content}")
                                    st.markdown(f"길이: {segment.duration}초")
                                    st.markdown("---")
                            
                            # YouTube 플레이어 삽입
                            components.html(player_html, height=500)
                            
                            st.write("※본 영상은 AI에 의해 편집된 요약 영상이며, 부정확할 수도 있습니다. 참고용으로 사용해주세요.")
                            
                        else:
                            st.error("세그먼트를 추출할 수 없습니다.")
                            
                    else:
                        st.error("유효한 YouTube URL을 입력해주세요.")
                        
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")

            elif menu == "🎬 다운로드 방식":
                st.header("1. 영상 다운로드 및 미리보기")
                video_bytes, filename = download_video_to_bytes(subject)
                if video_bytes is not None:
                    st.session_state.download_file_name = filename
                    st.session_state.download_video_bytes = video_bytes
                else:
                    st.session_state.download_file_name = None
                    st.session_state.download_video_bytes = None

                transcript = fetch_transcript(subject)
                if transcript is None or transcript.startswith("자막을 가져오는 중 오류") or transcript.startswith("올바른"):
                    st.error(transcript)
                    st.session_state.transcript = None
                else:
                    st.session_state.transcript = transcript
                    # 영상 총 길이 (초) 계산: 다운로드 완료 후 영상 파일에서
                    video_file_path = os.path.join(DOWNLOAD_DIR, st.session_state.download_file_name)
                    try:
                        clip = VideoFileClip(video_file_path)
                        total_duration = clip.duration  # 초 단위
                        clip.close()
                    except Exception as e:
                        st.error("다운로드된 영상에서 길이 확인 실패: " + str(e))
                        total_duration = None

                    if total_duration is None:
                        st.error("영상 길이 정보를 가져올 수 없어 요약 프롬프트 생성에 실패했습니다.")
                    else:
                        
                        summary_duration = total_duration * (summary_ratio / 100)
                        # 새로운 프롬프트 템플릿 (두 단계 접근)
                        prompt_template = f"""다음은 유튜브 동영상의 전체 자막 데이터입니다.
            전체 영상 길이: {total_duration:.2f}초
            요약 시간: {summary_duration:.2f}초  (전체 영상 길이의 {summary_ratio}%에 해당)

            먼저, 전체 자막을 꼼꼼하게 분석하여 영상에서 정말 중요한 핵심 키워드들을 추출하세요.
            주의사항:
            1. 반드시 핵심적인 키워드만 선택하세요. (예: 강의의 주요 개념, 중요한 용어 등)
            2. 키워드 사이의 부수적인 내용은 배제하고 오직 핵심 내용만 고려하세요.

            그리고 추출한 핵심 키워드에 기반하여, 각 키워드가 가장 명확하게 설명되는 구간을 아래 조건에 따라 선택하세요.
            - 선택된 구간들의 총 길이가 반드시 {summary_duration:.2f}초(전체 영상 길이의 {summary_ratio}%에 해당)여야 합니다.
            - 각 구간은 해당 핵심 키워드가 가장 집중된 순간이어야 하며, 관련성이 낮은 잡담이나 부수적 내용은 반드시 배제하세요.
            - 만약 지정한 요약 길이를 초과할 우려가 있다면, 가장 핵심적인 구간들만 남겨 총 길이가 정확히 {summary_duration:.2f}초를 넘지 않도록 조절하세요.

            각 핵심 구간은 아래 JSON 구조로 표현되어야 합니다.

            $$
            {{
                "segments": [
                    {{
                        "start_time": float,  // 구간 시작 시간 (초)
                        "end_time": float,    // 구간 종료 시간 (초)
                        "duration": float,    // 구간 길이 (초)
                        "content": str        // 해당 구간의 핵심 내용에 대한 간결한 설명
                    }}
                    // 추가 구간...
                ]
            }}
            $$

            응답은 반드시 오직 위 JSON 형식만 포함해야 하며, 추가 설명 없이 순수 JSON만 반환해 주세요.

            아래에 전체 자막 내용이 주어집니다:
            ::::
            {transcript}
            ::::"""
                        st.session_state.prompt_template = prompt_template

                        st.info("LangChain을 통해 LLM 모델에 프롬프트 전달 중입니다. 잠시 기다려주세요...")
                        llm = ChatOpenAI(
                            model_name="gpt-4o",
                            openai_api_key=OPENAI_API_KEY,
                        )
                        try:
                            response = llm.invoke(prompt_template)
                            st.session_state.llm_response = response.content
                        except Exception as e:
                            st.error("LLM 호출 중 오류 발생: " + str(e))
                            st.session_state.llm_response = None

                st.session_state.has_run = True

    if st.session_state.has_run and menu == "🎬 다운로드 방식":
        st.header("영상 다운로드 결과")
        if st.session_state.download_video_bytes and st.session_state.download_file_name:
            st.write("다운로드된 영상 파일:", st.session_state.download_file_name)
            st.download_button(
                label="영상 파일 다운로드",
                data=st.session_state.download_video_bytes,
                file_name=st.session_state.download_file_name,
                mime="video/mp4"
            )
        else:
            st.info("영상 다운로드에 실패했습니다.")

        st.header("저장된 영상 목록 및 미리보기")
        videos = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".mp4")]
        if videos:
            selected_video = st.selectbox("영상 선택", videos)
            if st.button("선택한 영상 미리보기", key="view_video"):
                file_path = os.path.join(DOWNLOAD_DIR, selected_video)
                try:
                    with open(file_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                except Exception as e:
                    st.error("영상을 불러오는 데 실패했습니다: " + str(e))
        else:
            st.info("저장된 영상이 없습니다.")

        st.header("2. 자막 분석 결과")
        st.subheader("LLM 응답 결과 (요약)")
        if st.session_state.llm_response:
            st.text_area("응답", st.session_state.llm_response, height=300)
        else:
            st.info("LLM 응답 없음")

        st.header("3. 영상 분할 (세그먼트 적용)")
        if st.button("영상 분할 실행"):
            output_files = split_video_by_segments()
            if output_files:
                st.write("분할된 영상 파일 목록:")
                for file in output_files:
                    st.write(file)
                    file_path = os.path.join(SEGMENT_DIR, file)
                    with open(file_path, "rb") as f:
                        video_bytes = f.read()
                    st.download_button(
                        label=f"{file} 다운로드",
                        data=video_bytes,
                        file_name=file,
                        mime="video/mp4"
                    )

        st.header("4. 영상 병합")
        if st.button("영상 병합 실행"):
            merged_file = merge_video_segments(SEGMENT_DIR, output_filename="merged_video.mp4")
            if merged_file:
                st.session_state.merged_file = merged_file
                st.success("영상 병합 완료!")

        if st.session_state.get("merged_file"):
            file_path = os.path.join(".", st.session_state.merged_file)
            try:
                with open(file_path, "rb") as f:
                    video_bytes = f.read()
                st.download_button(
                    label="병합된 영상 다운로드",
                    data=video_bytes,
                    file_name=st.session_state.merged_file,
                    mime="video/mp4"
                )
            except Exception as e:
                st.error("병합된 영상을 불러오지 못했습니다: " + str(e))



if __name__ == "__main__":
    main()
