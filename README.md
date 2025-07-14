# 🤖 MathRush-DataProcessor

MathRush 프로젝트용 수학 문제 PDF 자동 추출 및 라벨링 시스템

## 📋 프로젝트 개요

PDF 형태의 수학 문제집을 자동으로 분석하여 구조화된 데이터로 변환하는 도구입니다.
GPT-4o-mini를 활용하여 문제, 선택지, 정답, 해설, 교육과정 분류 등을 자동으로 추출합니다.

## 🎯 주요 기능

- **PDF → 이미지 변환**: 고해상도 이미지로 변환
- **GPT 자동 추출**: 문제 내용, 선택지, 정답 자동 인식
- **교육과정 분류**: 2015/2022 개정 교육과정 기준 자동 분류
- **데이터베이스 저장**: Supabase에 구조화된 데이터 저장
- **배치 처리**: 대량 문제 자동 처리
- **진행상황 모니터링**: 실시간 처리 현황 추적

## 🛠️ 기술 스택

- **Python 3.8+**
- **pdf2image**: PDF → 이미지 변환
- **OpenAI GPT-4o-mini**: 문제 추출 및 분류
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

### 단일 PDF 처리
```bash
python main.py --input samples/test.pdf --output database
```

### 배치 처리
```bash
python main.py --input samples/ --batch-size 5 --output database
```

### 테스트 모드 (샘플 확인)
```bash
python main.py --input samples/test.pdf --output json --test-mode
```

## 📁 프로젝트 구조

```
MathRush-DataProcessor/
├── processors/
│   ├── __init__.py
│   ├── pdf_converter.py      # PDF → 이미지 변환
│   ├── gpt_extractor.py      # GPT 문제 추출
│   └── db_saver.py           # 데이터베이스 저장
├── config/
│   ├── __init__.py
│   ├── settings.py           # 설정 관리
│   └── prompts.py            # GPT 프롬프트
├── utils/
│   ├── __init__.py
│   ├── logger.py             # 로깅
│   └── validator.py          # 데이터 검증
├── tests/                    # 테스트 파일
├── samples/                  # 테스트용 PDF
├── output/                   # 처리 결과물
├── main.py                   # 메인 실행 파일
├── requirements.txt          # Python 의존성
├── .env.example             # 환경변수 예시
└── README.md
```

## 📊 데이터 출력 형식

```json
{
  "problems": [
    {
      "content": "문제 본문",
      "problem_type": "multiple_choice",
      "choices": {
        "1": "선택지 1",
        "2": "선택지 2",
        "3": "선택지 3",
        "4": "선택지 4",
        "5": "선택지 5"
      },
      "correct_answer": "3",
      "explanation": "해설 내용",
      "curriculum": "2015개정",
      "level": "고3",
      "subject": "미적분",
      "chapter": "도함수의 활용",
      "difficulty": "medium",
      "tags": ["최댓값", "미분"],
      "source_info": {
        "exam_type": "수능",
        "year": 2023,
        "month": 11,
        "subject": "미적분",
        "problem_number": 15,
        "total_points": 4
      }
    }
  ]
}
```

## 💰 비용 정보

- **GPT-4o-mini**: 문제 1개당 약 $0.005-0.01
- **1000문제 예상 비용**: $5-10
- **Supabase**: 무료 티어 사용 가능

## 📝 개발 진행상황

- [ ] PDF 변환 기능 구현
- [ ] GPT 추출 로직 개발
- [ ] 데이터 검증 시스템
- [ ] 배치 처리 최적화
- [ ] 에러 처리 및 재시도
- [ ] 테스트 코드 작성

## 🔗 관련 프로젝트

- [MathRush](https://github.com/sliver2er/MathRush) - 메인 게임 프로젝트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.
