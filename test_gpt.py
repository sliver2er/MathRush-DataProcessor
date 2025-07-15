"""
Test GPT extraction with a single image to debug the issue.
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))

from processors.gpt_extractor import GPTExtractor
import json

def test_gpt_extraction():
    """Test GPT extraction with a single image."""
    print("=== Testing GPT Extraction ===\n")
    
    # Find a sample image
    sample_image = "output/test_images/2606_probs_page_001.png"
    
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        
        # Try to find any image from our processing
        image_dir = "output/images/2020-12-03_suneung/"
        if os.path.exists(image_dir):
            images = [f for f in os.listdir(image_dir) if f.endswith('.png')]
            if images:
                sample_image = os.path.join(image_dir, images[0])
                print(f"📸 Using extracted image: {sample_image}")
            else:
                print("❌ No images found for testing")
                return
        else:
            print("❌ No images available for testing")
            return
    
    try:
        # Initialize GPT extractor
        extractor = GPTExtractor()
        print("✅ GPT Extractor initialized")
        
        # Test problem extraction
        print(f"\n🔍 Testing problem extraction with: {sample_image}")
        problems_result = extractor.extract_problems_from_image(sample_image)
        
        print(f"📊 Problems extraction result:")
        print(f"  Total problems found: {len(problems_result.get('problems', []))}")
        
        if problems_result.get('problems'):
            print("  Problem details:")
            for i, problem in enumerate(problems_result['problems'][:3], 1):  # Show first 3
                print(f"    Problem {i}:")
                print(f"      Number: {problem.get('problem_number', 'N/A')}")
                print(f"      Content: {problem.get('content', 'N/A')[:100]}...")
                print(f"      Type: {problem.get('problem_type', 'N/A')}")
                print(f"      Level: {problem.get('level', 'N/A')}")
                print(f"      Subject: {problem.get('subject', 'N/A')}")
                print()
        else:
            print("  ❌ No problems found")
            if problems_result.get('error'):
                print(f"  Error: {problems_result['error']}")
            if problems_result.get('raw_response'):
                print(f"  Raw response: {problems_result['raw_response'][:300]}...")
        
        # Test solution extraction
        print(f"\n🔍 Testing solution extraction with: {sample_image}")
        solutions_result = extractor.extract_solutions_from_image(sample_image)
        
        print(f"📊 Solutions extraction result:")
        print(f"  Total solutions found: {len(solutions_result.get('solutions', []))}")
        
        if solutions_result.get('solutions'):
            print("  Solution details:")
            for i, solution in enumerate(solutions_result['solutions'][:3], 1):  # Show first 3
                print(f"    Solution {i}:")
                print(f"      Problem Number: {solution.get('problem_number', 'N/A')}")
                print(f"      Answer: {solution.get('correct_answer', 'N/A')}")
                print(f"      Explanation: {solution.get('explanation', 'N/A')[:100]}...")
                print()
        else:
            print("  ❌ No solutions found")
            if solutions_result.get('error'):
                print(f"  Error: {solutions_result['error']}")
            if solutions_result.get('raw_response'):
                print(f"  Raw response: {solutions_result['raw_response'][:300]}...")
        
        # Save results for debugging
        debug_file = "debug_gpt_results.json"
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump({
                'problems': problems_result,
                'solutions': solutions_result
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Debug results saved to: {debug_file}")
        
    except Exception as e:
        print(f"❌ GPT extraction test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpt_extraction()