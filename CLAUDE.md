# Claude 협업 가이드 - MathRush DataProcessor

## 📚 프로젝트 개요
**GitHub**: https://github.com/sliver2er/MathRush-DataProcessor

MathRush 프로젝트용 수학 문제 PDF 자동 추출 및 라벨링 시스템입니다. GPT-4o-mini를 활용하여 PDF 형태의 수학 문제집을 구조화된 데이터로 변환합니다.

## 🎯 현재 목표
- **메인 목표**: PDF 수학 문제 → Supabase DB 자동 저장 시스템 구축
- **예상 작업량**: 1,000-2,000문제
- **예상 비용**: $5-20 (GPT-4o-mini 기준)
- **예상 소요시간**: 2-5시간 (자동화 완료 후)

## 📁 프로젝트 구조
```
MathRush-DataProcessor/
├── processors/
│   ├── pdf_converter.py    # PDF → 이미지 변환
│   ├── gpt_extractor.py    # GPT 문제 추출
│   ├── math_processor.py   # 통합 처리 파이프라인
│   └── db_saver.py         # 데이터베이스 저장
├── config/
│   └── settings.py         # 설정 관리
├── utils/
│   ├── filename_parser.py  # 파일명 파싱
│   ├── image_extractor.py  # 이미지 추출
│   ├── math_content_extractor.py  # 수학 내용 추출
│   ├── problem_segmenter.py  # 문제 분할
│   └── solution_parser.py  # 해답 파싱
├── samples/                # 테스트용 PDF 샘플
├── output/                 # 처리 결과물
│   ├── images/            # 추출된 이미지
│   └── processed_images/  # 처리된 이미지
└── temp_new_process_method.py  # 임시 처리 방법
```

## 🔄 처리 파이프라인

1. **PDF → 이미지 변환** (pdf_converter.py)
   - pdf2image 라이브러리 사용
   - 고해상도 이미지로 변환 (problems/solutions 분리)

2. **이미지에서 문제별 추출** (image_extractor.py)
   - PDF 페이지에서 개별 문제 이미지 추출
   - 그래프, 도표, 도형 등 시각적 요소 분리
   - 문제별 이미지 파일 생성

3. **문제 분할 및 파싱** (problem_segmenter.py)
   - 페이지 단위 이미지를 개별 문제로 분할
   - 문제 번호 및 구조 인식
   - 객관식/주관식 문제 타입 분류

4. **GPT 문제 추출** (gpt_extractor.py)
   - 문제와 해답을 별도 처리
   - 교육과정 기준 프롬프트 제공
   - JSON 형식으로 구조화된 데이터 반환

5. **문제-해답 매칭** (math_processor.py)
   - 문제와 해답을 번호 기준으로 매칭
   - 누락된 문제/해답 탐지 및 처리
   - 완성된 문제 세트 생성

6. **데이터 검증 및 정제** (db_saver.py)
   - 필수 필드 체크
   - 중복 문제 탐지
   - 데이터 형식 정규화

7. **Supabase DB 저장** (db_saver.py)
   - problems 테이블에 일괄 삽입
   - 이미지 파일 경로 저장
   - 성공/실패 로그 기록

## 🛠️ 기술 스택
- **AI 모델**: GPT-4o-mini (OpenAI)
- **PDF 처리**: pdf2image
- **데이터베이스**: Supabase (PostgreSQL)
- **개발 환경**: Python 3.8+
- **이미지 저장**: Supabase Storage

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
  curriculum: ENUM                -- 2015개정 | 2022개정
  level: TEXT                     -- 중1~고3 | 고1~고3
  subject: TEXT                   -- 수학상|수학하|수필1|미적분|확률과통계|기하
  chapter: TEXT                   -- 단원명
  difficulty: ENUM                -- easy | medium | hard
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

## 🔧 실행 방법
```bash
# 단일 PDF 쌍 처리
python processors/math_processor.py --problems samples/2020-12-03_suneung_problems.pdf --solutions samples/2020-12-03_suneung_solutions.pdf

# 디렉토리 전체 PDF 쌍 처리
python processors/math_processor.py --directory samples/

# 동시 처리 개수 조정 (기본값: 3)
python processors/math_processor.py --directory samples/ --concurrent 5

# 디버그 모드 (이미지 및 JSON 저장)
python processors/math_processor.py --problems file1.pdf --solutions file2.pdf --save-images --save-json

# 데이터베이스 연결 테스트
python processors/math_processor.py --test-db
```

## 💡 다음 우선순위 작업
1. **환경 설정 완료** - API 키, Supabase 연결
2. **프롬프트 테스트** - 소규모 샘플로 정확도 검증
3. **PDF 변환 로직** - pdf2image 구현
4. **GPT 추출 로직** - OpenAI API 연동
5. **데이터베이스 저장** - Supabase 연동

## 📞 협업 시 참고사항
- **코드 리뷰**: 모든 핵심 로직은 Claude와 함께 검토
- **테스트 우선**: 작은 단위로 테스트하며 점진적 개발
- **문서화**: 변경사항은 즉시 이 문서에 반영
- **에러 처리**: 모든 외부 API 호출에 적절한 예외 처리