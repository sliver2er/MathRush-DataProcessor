# Claude 협업 가이드 - MathRush DataProcessor

## 📚 프로젝트 개요
**GitHub**: https://github.com/sliver2er/MathRush-DataProcessor

MathRush 프로젝트용 수학 문제 **수동 처리** 및 라벨링 시스템입니다. GPT-4o-mini를 활용하여 수동으로 분할된 문제 이미지를 구조화된 데이터로 변환합니다.

## 🎯 현재 목표
- **메인 목표**: 수동 분할된 수학 문제 이미지 → Supabase DB 저장 시스템
- **예상 작업량**: 1,000-2,000문제
- **예상 비용**: $3-10 (GPT-4o-mini 기준, 설명 추출 제외)
- **예상 소요시간**: 1-2시간 (수동 답안 입력 + 자동 내용 추출)

## 📁 프로젝트 구조 (간소화)
```
MathRush-DataProcessor/
├── manual_answer_input.py      # 수동 답안 입력 유틸리티
├── simple_processor.py         # 간단한 이미지 처리기
├── processors/
│   ├── gpt_extractor.py        # GPT 추출기
│   └── db_saver.py             # 데이터베이스 저장
├── config/
│   └── settings.py             # 설정 관리
├── utils/
│   └── filename_parser.py      # 파일명 파싱
├── input/                      # 수동 분할된 문제 이미지
└── samples/                    # 테스트용 PDF 샘플
```

## 🔄 처리 파이프라인 (간소화)

### **단계 1: 수동 준비 작업** (사용자)
1. **문제 이미지 분할**
   - PDF에서 수동으로 개별 문제 이미지 추출
   - 파일명 형식: `{exam_name}_problem_{number:02d}.png`
   - 수학 내용 이미지: `{exam_name}_problem_{number:02d}_diagram.png`

2. **디렉토리 구성**
   ```
   input/2020-12-03_suneung_가형/
   ├── 2020-12-03_suneung_가형_problem_01.png
   ├── 2020-12-03_suneung_가형_problem_01_diagram.png
   ├── 2020-12-03_suneung_가형_problem_02.png
   └── ...
   ```

### **단계 2: 수동 답안 입력** (manual_answer_input.py)
1. **답안 입력 유틸리티 실행**
   ```bash
   python manual_answer_input.py input/2020-12-03_suneung_가형/
   ```
2. **대화식 답안 입력**
   - 문제 유형 선택 (객관식/주관식)
   - 정답 입력
   - 데이터베이스에 즉시 저장

### **단계 3: 자동 내용 추출** (simple_processor.py)
1. **이미지 처리 실행**
   ```bash
   python simple_processor.py input/2020-12-03_suneung_가형/
   ```
2. **GPT 내용 추출** (processors/gpt_extractor.py)
   - 문제 내용만 추출 (설명 제외)
   - 객관식 선택지 추출
   - 수동 답안과 자동 결합

3. **데이터베이스 업데이트** (db_saver.py)
   - 기존 답안 레코드에 내용 추가
   - 완성된 문제 데이터 저장

## 🛠️ 기술 스택 (간소화)
- **AI 모델**: GPT-4o-mini (OpenAI) - 내용 추출만
- **데이터베이스**: Supabase (PostgreSQL)
- **개발 환경**: Python 3.8+
- **이미지 처리**: 수동 분할 + GPT Vision

## 📊 데이터베이스 스키마 (Supabase)
```sql
-- problems 테이블 구조
{
  id: UUID 기본키
  content: TEXT                    -- 문제 본문
  problem_type: ENUM              -- multiple_choice | subjective
  choices: JSONB                  -- 객관식 선택지
  correct_answer: TEXT            -- 정답
  explanation: TEXT               -- 해설
  exam_name: TEXT                 -- 시험명 (필수, 고유 키의 일부)
  problem_number: INTEGER         -- 문제 번호 (필수, 고유 키의 일부)
  level: TEXT                     -- 중1~고3 | 고1~고3
  subject: TEXT                   -- 수학상|수학하|수필1|미적분|확률과통계|기하
  chapter: TEXT                   -- 단원명
  difficulty: ENUM                -- easy | medium | hard
  correct_rate: FLOAT             -- 정답률 (0.0-100.0, 향후 난이도 결정용)
  source_info: JSONB              -- 출처 정보
  tags: TEXT[]                    -- 태그 배열
  images: TEXT[]                  -- 이미지 파일 경로 배열
  created_at: TIMESTAMP
  updated_at: TIMESTAMP
}
```

## 📝 JSON 출력 형식
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
      "explanation": "해설 내용",
      "curriculum": "2015개정",
      "level": "고3",
      "subject": "수학영역",
      "chapter": "미적분",
      "difficulty": "medium",
      "correct_rate": 45.2,
      "tags": ["최댓값", "미분"],
      "images": ["2020-12-03_suneung_problem_1_img_1.png"],
      "source_info": {
        "exam_type": "수능",
        "exam_date": "2020-12-03",
        "problem_number": 1,
        "total_points": 4,
        "problem_pdf": "2020-12-03_suneung_problems.pdf",
        "solution_pdf": "2020-12-03_suneung_solutions.pdf"
      }
    }
  ]
}
```

## ✅ 현재 진행상황 체크리스트

### 📦 환경 설정
- [ ] OpenAI API 키 설정
- [ ] Supabase 프로젝트 생성 및 연결
- [ ] problems 테이블 생성
- [ ] Python 라이브러리 설치 (pdf2image, openai, supabase)

### 🧪 테스트 단계 (10문제 샘플)
- [ ] PDF 샘플 2-3페이지 준비
- [ ] 수동 추출 테스트 (Claude 채팅)
- [ ] GPT 프롬프트 최적화
- [ ] 데이터 품질 및 정확도 검증

### 🚀 자동화 개발
- [ ] PDF 처리 스크립트 개발
- [ ] GPT API 연동 및 배치 처리
- [ ] 데이터 검증 로직
- [ ] Supabase 저장 로직
- [ ] 진행상황 모니터링 시스템

### 🔄 대량 처리
- [ ] 전체 PDF 처리 (1,000-2,000문제)
- [ ] 오류 처리 및 재시도
- [ ] 데이터 품질 검증 및 후처리

## 🎯 Claude와의 협업 포인트

### 1. 프롬프트 최적화
- GPT-4o-mini용 문제 추출 프롬프트 개발
- 교육과정 분류 정확도 향상
- JSON 형식 안정성 확보

### 2. 코드 구현 지원
- Python 스크립트 작성 및 디버깅
- 에러 처리 로직 구현
- 배치 처리 최적화

### 3. 데이터 검증
- 추출된 데이터 품질 검증
- 중복 제거 알고리즘
- 정답 일치성 확인

### 4. 테스트 및 디버깅
- 단위 테스트 작성
- 통합 테스트 수행
- 성능 최적화

## 🔧 실행 방법 (간소화)
```bash
# 1단계: 수동 답안 입력
python manual_answer_input.py input/2020-12-03_suneung_가형/

# 2단계: 이미지 내용 처리
python simple_processor.py input/2020-12-03_suneung_가형/

# 여러 시험 일괄 처리
python simple_processor.py input/ --recursive

# 사용 가능한 시험 목록 보기
python manual_answer_input.py input/ --list

# 데이터베이스 연결 테스트
python check_db.py

# 개별 이미지 테스트 (디버그용)
python -m processors.gpt_extractor path/to/problem.png --verbose
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

## 💡 다음 우선순위 작업 (간소화)
1. **수동 이미지 분할** - 시험 문제를 개별 이미지로 분할
2. **답안 입력 테스트** - manual_answer_input.py로 답안 입력
3. **내용 추출 테스트** - simple_processor.py로 GPT 추출
4. **데이터 검증** - 데이터베이스에 올바르게 저장되는지 확인
5. **대량 처리** - 전체 문제집 처리
6. **난이도 결정** - correct_rate 데이터 수집 및 difficulty 업데이트 (향후 작업)

## 📞 협업 시 참고사항
- **코드 리뷰**: 모든 핵심 로직은 Claude와 함께 검토
- **테스트 우선**: 작은 단위로 테스트하며 점진적 개발
- **문서화**: 변경사항은 즉시 이 문서에 반영
- **에러 처리**: 모든 외부 API 호출에 적절한 예외 처리