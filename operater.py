import os
import yt_dlp
import subprocess
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

# 환경 변수 불러오기
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 유튜브 자막 가져오기 함수
def get_transcript(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        # 시작 시간과 지속 시간 정보를를 포함
        transcript_text = ' '.join([f"[{item['start']}초~{item['start']+item['duration']}초] {item['text']}" for item in transcript_list])
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


# 비디오 세그먼트를 위한 데이터 모델 정의
class Segment(BaseModel):
    start_time: float = Field(description="세그먼트 시작 시간(초)")
    end_time: float = Field(description="세그먼트 종료 시간(초)")
    yt_title: str = Field(description="이 세그먼트를 바이럴 서브토픽으로 만들기 위한 유튜브 제목")
    description: str = Field(description="이 세그먼트를 바이럴로 만들기 위한 상세 설명")
    duration: float = Field(description="세그먼트 길이(초)")  # int에서 float로 변경


# 유튜브 자막 요약 함수
def summarize_transcript(transcript, video_id, summary_ratio=None, segment_count=None):
    try:
        # 영상 유형 분류
        video_type = classify_video_type(transcript)
        print(f"{video_type=}")

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
              
            # Segment 객체 리스트로 변환
            segments = []
            for item in segments_data:
                segment = Segment(
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                    yt_title=item["yt_title"],
                    description=item["description"],
                    duration=float(item["duration"]) if isinstance(item["duration"], (int, float)) else 
                              float(item["end_time"] - item["start_time"])
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
                max-height: 300px; /* 최대 높이 설정 */
                overflow-y: auto; /* 스크롤 추가 */
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
                    transcript = get_transcript(video_id)
                    
                    if not transcript:
                        st.error("자막을 가져올 수 없습니다. 다른 동영상을 시도해주세요.")
                        return
                    
                    # 자막 요약 및 처리
                    segments = summarize_transcript(transcript, video_id, summary_ratio, segment_count)
                    
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
                                "yt_title": segment.yt_title,
                                "description": segment.description,
                                "duration": segment.duration
                            })
                    
                    # 세그먼트 정보를 JSON으로 변환
                    segments_json = json.dumps(filtered_segments)
                    
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

                        <script src="https://www.youtube.com/iframe_api"></script>
                        <script>
                          const segments = {segments_json};
                          let currentSegment = 0;
                          let player;

                          // YouTube API가 준비되면 호출되는 함수
                          function onYouTubeIframeAPIReady() {{
                              player = new YT.Player('player', {{
                                  events: {{
                                      'onReady': onPlayerReady,
                                      'onStateChange': onPlayerStateChange
                                  }}
                              }});
                          }}

                          // 플레이어가 준비되면 호출되는 함수
                          function onPlayerReady(event) {{
                              playNextSegment();
                          }}

                          // 플레이어 상태가 변경될 때 호출되는 함수
                          function onPlayerStateChange(event) {{
                              // 현재 세그먼트가 끝나면 다음 세그먼트로 이동
                              if (event.data === YT.PlayerState.PLAYING) {{
                                  const segment = segments[currentSegment];
                                  const duration = (segment.end_time - segment.start_time) * 1000;
                                  setTimeout(() => {{
                                      if (currentSegment < segments.length - 1) {{
                                          currentSegment++;
                                          playNextSegment();
                                      }}
                                  }}, duration);
                              }}
                          }}

                          function playNextSegment() {{
                              if (currentSegment >= segments.length) return;
                              
                              const segment = segments[currentSegment];
                              player.seekTo(segment.start_time);
                              player.playVideo();
                          }}

                          // 이전 세그먼트 재생 버튼
                          function playPreviousSegment() {{
                              if (currentSegment > 0) {{
                                  currentSegment--;
                                  playNextSegment();
                              }}
                          }}

                          // 다음 세그먼트 재생 버튼
                          function playForwardSegment() {{
                              if (currentSegment < segments.length - 1) {{
                                  currentSegment++;  // 먼저 인덱스를 증가
                                  playNextSegment();
                              }}
                          }}

                          // 컨트롤 버튼 추가
                          document.write(`
                              <div style="margin-top: 10px; text-align: center;">
                                  <button onclick="playPreviousSegment()" style="margin: 5px;">이전 세그먼트</button>
                                  <button onclick="playForwardSegment()" style="margin: 5px;">다음 세그먼트</button>
                              </div>
                          `);
                        </script>
                        """
                        
                        # 세그먼트 정보 표시
                        st.markdown("### 세그먼트 정보")
                        with st.expander("📋 세그먼트 정보 보기"):
                            for i, segment in enumerate(segments):
                                st.markdown(f"**세그먼트 {i+1}**: {segment.yt_title}")
                                st.markdown(f"시간: {segment.start_time}초 ~ {segment.end_time}초")
                                st.markdown(f"설명: {segment.description}")
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


if __name__ == "__main__":
    main()
