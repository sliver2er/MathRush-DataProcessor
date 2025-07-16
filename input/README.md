# Input Directory

This directory is for manually segmented problem images.

## Directory Structure

```
input/
├── {exam_name}/                    # Exam directory (e.g., 2020-12-03_suneung)
│   ├── {exam_name}_problem_01.png  # Individual problem images
│   ├── {exam_name}_problem_02.png
│   ├── {exam_name}_problem_03.png
│   ├── {exam_name}_problem_01_diagram.png  # Math content images (optional)
│   ├── {exam_name}_problem_02_graph.png    # Math content images (optional)
│   └── ...
```

## File Naming Convention

### Problem Images
- Format: `{exam_name}_problem_{number:02d}.png`
- Examples:
  - `2020-12-03_suneung_problem_01.png`
  - `2020-12-03_suneung_problem_02.png`
  - `2021-11-18_mock_problem_01.png`

### Math Content Images (Optional)
- Format: `{exam_name}_problem_{number:02d}_{content_type}.png`
- Content types: `diagram`, `graph`, `figure`, `content`, `chart`, etc.
- Examples:
  - `2020-12-03_suneung_problem_01_diagram.png`
  - `2020-12-03_suneung_problem_03_graph.png`
  - `2020-12-03_suneung_problem_05_figure.png`

## Exam Name Format

- Format: `YYYY-MM-DD_ExamType`
- Examples:
  - `2020-12-03_suneung` (수능)
  - `2021-11-18_mock` (모의고사)
  - `2024-03-15_school` (학교시험)

## Usage

1. **Create exam directory**: `mkdir input/2020-12-03_suneung`
2. **Add problem images**: Place manually segmented images
3. **Input answers**: `python manual_answer_input.py input/2020-12-03_suneung/`
4. **Process images**: `python simple_processor.py input/2020-12-03_suneung/`

## Notes

- Images should be high quality PNG files
- Each problem should be cleanly segmented
- Math content images are optional but recommended for problems with diagrams
- Make sure problem numbers are sequential (01, 02, 03, ...)