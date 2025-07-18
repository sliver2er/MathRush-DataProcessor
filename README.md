# 🤖 MathRush-DataProcessor

MathRush 프로젝트용 수학 문제 **수동 처리** 및 라벨링 시스템

## 📋 프로젝트 개요

수동으로 분할된 수학 문제 이미지를 GPT-4o-mini를 통해 구조화된 데이터로 변환하는 도구입니다.
간단한 2단계 워크플로우로 문제 내용 추출과 정답 입력을 분리하여 효율적으로 처리합니다.

## 🎯 주요 기능

- **수동 이미지 분할**: PDF에서 개별 문제 이미지 수동 추출
- **GPT 내용 추출**: 문제 본문, 선택지 자동 추출 (GPT-4o-mini)
- **수동 정답 입력**: 대화형 또는 배치 방식 정답 입력
- **데이터베이스 저장**: Supabase에 구조화된 데이터 저장
- **교육과정 분류**: 파일명 기반 자동 분류
- **비용 최적화**: 설명 제외로 GPT 비용 절약

## 🛠️ 기술 스택

- **Python 3.8+**
- **OpenAI GPT-4o-mini**: 문제 내용 추출 (비용 최적화)
- **Supabase**: 데이터베이스 저장
- **Pillow**: 이미지 처리
- **python-dotenv**: 환경변수 관리

## 📦 설치

```bash
# 저장소 클론
git clone https://github.com/sliver2er/MathRush-DataProcessor.git
cd MathRush-DataProcessor

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 API 키들 입력
```

## ⚙️ 환경변수 설정

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# 기타 설정
BATCH_SIZE=5
MAX_RETRIES=3
```

## 🚀 사용법

### 📋 워크플로우 (2단계)

1. **단계 1**: `simple_processor.py` - GPT로 문제 내용 추출 후 DB에 저장
2. **단계 2**: `manual_answer_input.py` - 기존 레코드에 정답 수동 입력

### 📁 이미지 준비

먼저 PDF에서 개별 문제 이미지를 수동으로 분할하고 다음과 같이 배치:

```
input/2020-12-03_suneung_가형/
├── 2020-12-03_suneung_가형_problem_01.png
├── 2020-12-03_suneung_가형_problem_01_diagram.png  # 수학 내용 이미지 (선택사항)
├── 2020-12-03_suneung_가형_problem_02.png
└── ...
```

### 🤖 단계 1: 문제 내용 추출

```bash
# 특정 시험 처리
python simple_processor.py input/2020-12-03_suneung_가형/

# 모든 시험 일괄 처리
python simple_processor.py input/ --recursive

# 자세한 로그 출력
python simple_processor.py input/2020-12-03_suneung_가형/ --verbose
```

### ✍️ 단계 2: 정답 입력

```bash
# 대화형 정답 입력
python manual_answer_input.py input/2020-12-03_suneung_가형/

# 단일 문제 정답 입력
python manual_answer_input.py input/2020-12-03_suneung_가형/ --problem 27 --answer 5 --type subjective

# 쉼표로 구분된 정답 일괄 입력
python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers "1,2,3,4,5"

# JSON 파일로 정답 입력
python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers-file input/2020-12-03_suneung_가형/answers.json

# 텍스트 파일로 정답 입력 (쉼표로 구분)
python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers-file input/2020-12-03_suneung_가형/answers.txt

# 사용 가능한 시험 목록 보기
python manual_answer_input.py input/ --list
```

### 📄 정답 파일 형식

정답 파일은 각 시험 디렉토리 내에 위치하며, 다음 형식을 지원합니다:

#### 1. 텍스트 파일 (.txt)
```
# answers.txt
1,2,3,4,5
```

#### 2. JSON 파일 (.json)
```json
{
  "1": "3",
  "2": "4", 
  "3": "2",
  "4": "1",
  "5": "5"
}
```

또는 상세 형식:
```json
{
  "1": {"answer": "3", "type": "multiple_choice"},
  "2": {"answer": "4", "type": "multiple_choice"},
  "3": {"answer": "서술형답안", "type": "subjective"}
}
```

### 🎯 유틸리티

```bash
# 데이터베이스 연결 테스트
python check_db.py

# 개별 이미지 테스트 (디버그용)
python lightweight_gpt_extractor.py path/to/problem.png --verbose
```

## 📁 프로젝트 구조

```
MathRush-DataProcessor/
├── manual_answer_input.py      # 수동 답안 입력 유틸리티
├── simple_processor.py         # 간단한 이미지 처리기
├── lightweight_gpt_extractor.py # 경량 GPT 추출기
├── processors/
│   ├── __init__.py
│   ├── gpt_extractor.py        # GPT 문제 추출
│   └── db_saver.py             # 데이터베이스 저장
├── config/
│   ├── __init__.py
│   ├── settings.py             # 설정 관리
│   └── prompts.py              # GPT 프롬프트
├── utils/
│   ├── __init__.py
│   └── filename_parser.py      # 파일명 파싱
├── input/                      # 수동 분할된 문제 이미지
│   └── 2020-12-03_suneung_가형/
│       ├── 2020-12-03_suneung_가형_problem_01.png
│       ├── 2020-12-03_suneung_가형_problem_01_diagram.png
│       ├── answers.txt         # 정답 파일 (텍스트 형식)
│       ├── answers.json        # 정답 파일 (JSON 형식)
│       └── ...
├── samples/                    # 테스트용 PDF 샘플
├── CLAUDE.md                   # Claude 협업 가이드
├── requirements.txt            # Python 의존성
├── .env.example               # 환경변수 예시
└── README.md
```

## 📊 데이터 출력 형식

### 데이터베이스 스키마 (Supabase)

```sql
-- problems 테이블 구조
{
  id: UUID 기본키
  content: TEXT                    -- 문제 본문 (필수)
  problem_type: TEXT              -- multiple_choice | subjective (필수)
  correct_answer: TEXT            -- 정답 (필수)
  choices: JSONB                  -- 객관식 선택지 (옵션)
  explanation: TEXT               -- 해설 (옵션, 현재 미사용)
  level: TEXT                     -- 중1~고3 | 고1~고3 (옵션)
  subject: TEXT                   -- 수학상|수학하|수필1|미적분|확률과통계|기하 (옵션)
  chapter: TEXT                   -- 단원명 (옵션)
  difficulty: TEXT                -- easy | medium | hard (옵션)
  source_info: JSONB              -- 출처 정보 (옵션)
  tags: TEXT[]                    -- 태그 배열 (옵션)
  images: TEXT[]                  -- 이미지 파일 경로 배열 (옵션)
  created_at: TIMESTAMP           -- 생성일시
  updated_at: TIMESTAMP           -- 수정일시
}
```

### JSON 데이터 예시

```json
{
  "problems": [
    {
      "problem_number": 1,
      "content": "문제 본문 전체",
      "problem_type": "multiple_choice",
      "choices": {
        "1": "선택지1",
        "2": "선택지2",
        "3": "선택지3",
        "4": "선택지4",
        "5": "선택지5"
      },
      "correct_answer": "3",
      "explanation": "",
      "level": "고3",
      "subject": "수학영역",
      "chapter": "미적분",
      "difficulty": "medium",
      "tags": ["gpt_extracted"],
      "images": ["2020-12-03_suneung_problem_1.png"],
      "source_info": {
        "exam_type": "수능",
        "exam_date": "2020-12-03",
        "exam_name": "2020-12-03_suneung_가형",
        "problem_number": 1,
        "filename": "2020-12-03_suneung_가형_problem_01.png",
        "gpt_processed": true,
        "gpt_processed_timestamp": "2025-01-17T12:00:00"
      }
    }
  ]
}
```

## 📋 지원하는 시험 유형

### 수능/모의고사 과목별
- **2020-12-03_suneung_가형** (과거 가형)
- **2020-12-03_suneung_나형** (과거 나형)
- **2022-11-17_suneung** (2022+ 수능 공통, 1-22번)
- **2022-11-17_suneung_공통** (명시적 공통 지정)
- **2022-11-17_suneung_미적분** (현재 미적분, 23-30번)
- **2022-11-17_suneung_기하** (현재 기하, 23-30번)
- **2022-11-17_suneung_확통** (현재 확률과통계, 23-30번)
- **2021-09-01_mock_가형** (모의고사 가형)

### 기본 과목
- **2024-03-15_school_수학상** (학교시험 수학상)
- **2024-06-10_monthly_수학1** (월례고사 수학Ⅰ)

## 💰 비용 정보

- **GPT-4o-mini**: 문제 1개당 약 $0.003-0.005 (내용 추출만)
- **1000문제 예상 비용**: $3-5 (설명 제외로 비용 절약)
- **Supabase**: 무료 티어 사용 가능

## 📝 개발 진행상황

- [x] 수동 이미지 분할 워크플로우 구현
- [x] GPT 문제 내용 추출 구현
- [x] 수동 정답 입력 시스템 구현
- [x] 데이터베이스 저장 로직 구현
- [x] 2단계 워크플로우 최적화
- [x] 파일명 기반 교육과정 분류
- [x] 배치 처리 및 에러 처리
- [x] 테스트 및 검증 완료

## 🔍 사용 예시

### 완전한 워크플로우 예시

```bash
# 1단계: 문제 내용 추출 (GPT 사용)
python simple_processor.py input/2020-12-03_suneung_가형/
# 출력: ✅ 1개 문제 처리 완료, 데이터베이스에 저장

# 2단계: 정답 입력 (수동)
# 방법 1: 단일 문제
python manual_answer_input.py input/2020-12-03_suneung_가형/ --problem 27 --answer 5 --type subjective

# 방법 2: 쉼표로 구분된 문자열
python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers "1,2,3,4,5"

# 방법 3: 텍스트 파일 사용 (권장)
echo "1,2,3,4,5" > input/2020-12-03_suneung_가형/answers.txt
python manual_answer_input.py input/2020-12-03_suneung_가형/ --answers-file input/2020-12-03_suneung_가형/answers.txt
```

## 🔗 관련 프로젝트

- [MathRush](https://github.com/sliver2er/MathRush) - 메인 게임 프로젝트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.