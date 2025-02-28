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
  - Nexon API를 사용해 특정 선수(시즌·매치 유형)의 **경기 평균 통계** 조회  
  - **Matplotlib** 박스플롯을 통해 시각화 후 **Streamlit**으로 표시  
- **유튜브 영상 검색**  
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
