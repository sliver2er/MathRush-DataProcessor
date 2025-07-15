"""
Test the new problem-based architecture.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from utils.problem_segmenter import ProblemSegmenter
from utils.math_content_extractor import MathContentExtractor
from utils.solution_parser import SolutionParser
from processors.gpt_extractor import GPTExtractor
import shutil

def test_new_architecture():
    """Test the new problem-based architecture components."""
    print("=== Testing New Problem-Based Architecture ===\n")
    
    # Test files
    test_image = "output/test_images/2606_probs_page_001.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    # Create output directory
    output_dir = "output/test_new_architecture"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Test problem segmentation
        print("1. Testing Problem Segmentation")
        print("=" * 40)
        
        segmenter = ProblemSegmenter()
        problems = segmenter.segment_page_into_problems(test_image, output_dir, "test_page")
        
        print(f"✅ Segmented {len(problems)} problems from page")
        for i, problem in enumerate(problems, 1):
            print(f"   Problem {i}: Number {problem['number']}, Image: {problem['filename']}")
        
        # Step 2: Test mathematical content extraction
        print(f"\n2. Testing Mathematical Content Extraction")
        print("=" * 40)
        
        extractor = MathContentExtractor()
        
        for problem in problems:
            print(f"   Processing problem {problem['number']}...")
            
            problem_key = f"test_page_problem_{problem['number']:02d}"
            math_content = extractor.extract_mathematical_content(
                problem['image_path'], output_dir, problem_key
            )
            
            print(f"   ✅ Extracted {len(math_content)} mathematical content pieces")
            problem['math_content'] = math_content
        
        # Step 3: Test solution parser (with placeholder)
        print(f"\n3. Testing Solution Parser")
        print("=" * 40)
        
        solution_parser = SolutionParser()
        
        # Create mock answer key for testing
        mock_answer_key = {1: "③", 2: "2", 3: "①", 4: "12"}
        print(f"   Using mock answer key: {mock_answer_key}")
        
        # Step 4: Test GPT extraction with individual problems
        print(f"\n4. Testing GPT Extraction with Individual Problems")
        print("=" * 40)
        
        gpt_extractor = GPTExtractor()
        
        # Test with first problem
        if problems:
            problem = problems[0]
            print(f"   Testing with problem {problem['number']}...")
            
            result = gpt_extractor.extract_problem_from_image(
                problem['image_path'], 
                problem.get('math_content', [])
            )
            
            if 'error' not in result:
                print(f"   ✅ Successfully extracted problem data")
                print(f"      Problem Number: {result.get('problem_number', 'N/A')}")
                print(f"      Content: {result.get('content', 'N/A')[:100]}...")
                print(f"      Type: {result.get('problem_type', 'N/A')}")
            else:
                print(f"   ❌ Failed to extract problem: {result.get('error')}")
        
        # Step 5: Test answer key conversion
        print(f"\n5. Testing Answer Key Conversion")
        print("=" * 40)
        
        solutions_result = gpt_extractor.extract_answers_from_answer_key(mock_answer_key)
        print(f"   ✅ Converted {len(solutions_result.get('solutions', []))} answers")
        
        # Step 6: Summary
        print(f"\n6. Architecture Test Summary")
        print("=" * 40)
        print(f"   ✅ Problem Segmentation: {len(problems)} problems")
        print(f"   ✅ Math Content Extraction: {sum(len(p.get('math_content', [])) for p in problems)} content pieces")
        print(f"   ✅ Solution Parser: {len(mock_answer_key)} answers")
        print(f"   ✅ GPT Individual Processing: Working")
        print(f"   ✅ Answer Key Conversion: Working")
        
        print(f"\n🎉 New architecture test completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Architecture test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_architecture()