# ⚽ FC Online Chat Bot

> **Streamlit** + **LangChain** + **Nexon API** + **YouTube Data API**  
> 축구 게임 팬들을 위한 **맞춤형** 동영상 검색 & 선수 통계 조회 **챗봇** 프로젝트

---

## ✨ Overview
이 프로젝트는 `Streamlit`을 이용해 **FC Online** 게임 관련 질의를 처리하고,  
- 🎞 **유튜브 영상 검색**  
- 📊 **선수 통계 조회**  

등의 기능을 제공하는 **챗봇/어시스턴트** 시스템을 구현한 사이드 프로젝트입니다.  
`OpenAI`의 GPT 모델(`LangChain`)과 `Nexon`의 오픈 API를 결합해 사용자의 질의에 따라 **동영상을 추천**하거나 **선수 통계를 조회**해 보여줍니다.

---

## 🌟 Features
- **LLM 기반 질의 분석**  
  - `langchain_core` 라이브러리를 활용해 GPT 출력(JSON)을 **자동 파싱**  
  - **프롬프트 템플릿**(`PromptTemplate`)으로 “질의 → 분석 → 액션 결정” 경로 설정
  
- **FC Online 통계 조회**

  ![ezgif com-speed (1)](https://github.com/user-attachments/assets/c862d655-c41d-46b5-b911-4c84ef86d7fa)

  - Nexon API를 사용해 특정 선수(시즌·매치 유형)의 **경기 평균 통계** 조회  
  - **Matplotlib** 박스플롯을 통해 시각화 후 **Streamlit**으로 표시  
- **유튜브 영상 검색**
  
![ezgif com-speed](https://github.com/user-attachments/assets/93845d92-0618-4c59-b69f-105b7c72d1ab)

  
  - **YouTube Data API**로 관련 영상 검색 → **좋아요 수** 기준 정렬  
  - 상위 1~N개의 결과를 카드 형식으로 보여줌  
- **Streamlit 기반 UI**  
  - 사용자와 **챗** 형태로 직관적 상호작용  
  - 메시지/동영상/그래프 등을 동일 화면에서 출력  

---

## ⚙️ Project Structure (Key Points)

1. **AssistantConfig**  
   - `.env` 파일에서 `YouTube`, `Nexon`, `OpenAI` **API Key** 로드  
   - 모델, 온도(temperature) 등 LLM **옵션** 설정

2. **AgentAction (BaseModel)**  
   - GPT가 생성하는 **JSON 응답**(`action`, `search_keyword`)을 Pydantic으로 검증·파싱

3. **Assistant**  
   - 전체 로직 담당  
   - `ChatOpenAI`(LangChain)으로 **질의→액션 결정** 진행  
   - **YouTube/Nexon API** 호출 + **시각화** 기능

4. **main_ / main__ / main**  
   - **Streamlit 재렌더링** 문제로 인해 **챗 입력**(`main_()`) & **선수 상세 입력**(`main__()`)를 분리  
   - `main()`에서 두 함수를 조합해 **최종 실행**  

---

## 🔧 Usage

1. **환경 변수 설정 (.env)**  
   ```dotenv
   YOUTUBE_API_KEY=YOUR_YOUTUBE_KEY
   OPENAI_API_KEY=YOUR_OPENAI_KEY
   API_KEY=YOUR_NEXON_KEY
   OPENAI_MODEL=gpt-4
   TEMPERATURE=0.0

---

## 🏃 Interact
- 웹 브라우저에서 `localhost:8501` 접속
- 입력창에 챗 메시지 (“메시 평균 스탯 알려줘” 등) 입력
- 시즌/매치 타입 선택 후 통계 그래프 확인 가능

---

## 🏗 Code Explanation

### 1. Environment & Config
- `.env` 파일에서 API KEY 로드
- `AssistantConfig`로 **LLM** 설정(모델·온도 등)

### 2. LLM & Prompt
- **PromptTemplate:**
   - GPT에게 질의 형식·분석 규칙 전달
- **RunnableSequence:**
   - (프롬프트 → LLM → 파서) 순서로 실행, GPT의 출력(JSON)을 자동 파싱

### 3. Action Parsing
- **JsonOutputParser + AgentAction (Pydantic)**
   - GPT가 `"search_video"`, `"additional_input"`, `"not_supported"` 중 하나를 **action**으로 응답
   - `search_keyword` 역시 자동 추출

### 4. Search Logic
- **search_videos**
   - 유튜브 검색 API → **좋아요 수** 기준 정렬 후 결과 반환
- **search_stat**
   - Nexon API로 특정 **선수 ID** 조회
   - 경기 스탯(슛, 골, 패스 등) 누적
   - **Matplotlib** 박스플롯 시각화

### 5. Streamlit Flow
- main_()
   - 채팅 입력 후 → `process_query` 호출
   - GPT가 결정한 `action`을 **세션**에 저장
- main__()
   - `action == "additional_input"` 시 **시즌/매치** 선택 폼 표시
   - 선택 값에 따라 search_stat를 불러와 통계 그래프 표시
- main()
   - `main_()` + `main__()` 합쳐 **2단계 입력** 흐름 완성

---

## 🧩 Trouble Shooting

### 문제 요약
- Streamlit에서 사용자가 입력할 때마다 새로고침이 발생
  → 세션 정보가 초기화되어 추가 입력 단계로 못 넘어가는 문제

### 해결 과정
1. 하나의 `main` 함수에서 전부 처리 시,
   - 새로고침 → 이전 action 사라짐
2. **함수 분할**
   - `main_()` = 챗 입력으로 `action` 결정
   - `main__()` = `action == "additional_input"` → **시즌/매치** 입력 + **선수 통계** 조회
3. st.session_state 활용
   - 이전 state(액션·키워드) 유지
   - 2단계 입력(챗 → 세부 정보) 원활히 진행
