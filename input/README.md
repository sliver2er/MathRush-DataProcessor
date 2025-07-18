# Input Directory

This directory is for manually segmented problem images.

## Directory Structure

```
input/
├── {exam_name}/                           # Exam directory with subject
│   ├── {exam_name}_problem_01.png         # Individual problem images
│   ├── {exam_name}_problem_02.png
│   ├── {exam_name}_problem_03.png
│   ├── {exam_name}_problem_01_diagram.png # Math content images (optional)
│   ├── {exam_name}_problem_02_graph.png   # Math content images (optional)
│   └── ...

# Examples:
├── 2020-12-03_suneung_가형/
│   ├── 2020-12-03_suneung_가형_problem_01.png
│   ├── 2020-12-03_suneung_가형_problem_02.png
│   └── 2020-12-03_suneung_가형_problem_01_diagram.png
├── 2022-11-17_suneung_미적분/
│   ├── 2022-11-17_suneung_미적분_problem_01.png
│   └── 2022-11-17_suneung_미적분_problem_02.png
```

## File Naming Convention

### Problem Images
- Format: `{exam_name}_problem_{number:02d}.png`
- Examples:
  - `2020-12-03_suneung_가형_problem_01.png`
  - `2022-11-17_suneung_미적분_problem_02.png`
  - `2021-09-01_mock_기하_problem_01.png`

### Math Content Images (Optional)
- Format: `{exam_name}_problem_{number:02d}_{content_type}.png`
- Content types: `diagram`, `graph`, `figure`, `content`, `chart`, etc.
- Examples:
  - `2020-12-03_suneung_가형_problem_01_diagram.png`
  - `2022-11-17_suneung_미적분_problem_03_graph.png`
  - `2021-09-01_mock_기하_problem_05_figure.png`

## Exam Name Format

### Basic Format
- Format: `YYYY-MM-DD_ExamType` or `YYYY-MM-DD_ExamType_Subject`

### With Subject Specification
- Format: `YYYY-MM-DD_ExamType_Subject`
- Examples:
  - `2020-12-03_suneung_가형` (수능 가형)
  - `2020-12-03_suneung_나형` (수능 나형)
  - `2022-11-17_suneung_공통` (수능 공통 1-22번)
  - `2022-11-17_suneung_미적분` (수능 미적분 23-30번)
  - `2022-11-17_suneung_기하` (수능 기하 23-30번)
  - `2022-11-17_suneung_확통` (수능 확률과통계 23-30번)
  - `2021-09-01_mock_가형` (모의고사 가형)
  - `2023-06-01_mock_미적분` (모의고사 미적분)

### Basic Format (Without Subject)
- Examples:
  - `2022-11-17_suneung` (2022+ 수능, 자동으로 '공통'으로 인식)
  - `2024-03-15_school` (학교시험)
  - `2024-06-10_monthly` (월례고사)

## Subject Types Supported

### 과거 수능/모의고사 (2017-2021)
- **가형**: 미적분, 확률과통계, 기하 (고3)
- **나형**: 수학Ⅰ, 수학Ⅱ, 확률과통계 (고3)

### 현재 수능/모의고사 (2022~)
- **공통**: 공통 문제 (문제 1-22번, 수학Ⅰ+수학Ⅱ) (고3)
- **미적분**: 미적분 선택과목 (문제 23-30번) (고3)
- **기하**: 기하 선택과목 (문제 23-30번) (고3) 
- **확통**: 확률과통계 선택과목 (문제 23-30번) (고3)

### 기본 과목
- **수학상**: 수학(상) (고1)
- **수학하**: 수학(하) (고1)
- **수학1**: 수학Ⅰ (고2)
- **수학2**: 수학Ⅱ (고2)

## Usage

### For 2022+ 수능 공통 problems (1-22번)
```bash
mkdir input/2022-11-17_suneung          # No subject = 공통
# or explicitly:
mkdir input/2022-11-17_suneung_공통

python manual_answer_input.py input/2022-11-17_suneung/
python simple_processor.py input/2022-11-17_suneung/
```

### For specific subjects
```bash
mkdir input/2020-12-03_suneung_가형     # 과거 가형
mkdir input/2022-11-17_suneung_미적분   # 현재 미적분

python manual_answer_input.py input/2020-12-03_suneung_가형/
python simple_processor.py input/2020-12-03_suneung_가형/
```

## Database Fields

When processed, each problem will include:
- **exam_name** + **problem_number**: Unique identifier pair
- **content**: Problem text extracted by GPT
- **problem_type**: multiple_choice or subjective
- **choices**: Answer choices for multiple choice problems
- **correct_answer**: Manually input answer
- **difficulty**: easy/medium/hard (defaults to medium)
- **correct_rate**: Answer success rate (0.0-100.0%) for future difficulty determination
- **images**: Paths to math content images only

## Notes

- Images should be high quality PNG files (JPG/JPEG also supported)
- Each problem should be cleanly segmented
- Math content images are optional but recommended for problems with diagrams
- Make sure problem numbers are sequential (01, 02, 03, ...)
- The system prevents duplicate records using exam_name + problem_number as unique keys