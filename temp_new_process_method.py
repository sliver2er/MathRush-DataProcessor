    def process_pdf_pair(self, problems_pdf: str, solutions_pdf: str, resume: bool = False) -> Dict[str, Any]:
        """
        Process a single PDF pair using the new problem-based workflow.
        
        New workflow: PDF → Pages → Problems → Math Content → GPT → Database
        
        Args:
            problems_pdf: Path to problems PDF
            solutions_pdf: Path to solutions PDF
            resume: Whether to resume from checkpoint
            
        Returns:
            Processing results summary
        """
        start_time = time.time()
        
        # Parse filenames and validate
        prob_info = self.filename_parser.parse_filename(problems_pdf)
        sol_info = self.filename_parser.parse_filename(solutions_pdf)
        
        if not prob_info['is_valid'] or not sol_info['is_valid']:
            raise ValueError(f"Invalid filename format. Problems: {prob_info.get('error')}, Solutions: {sol_info.get('error')}")
        
        # Validate that both files are from same exam
        if (prob_info['exam_date'] != sol_info['exam_date'] or 
            prob_info['exam_type'] != sol_info['exam_type']):
            raise ValueError(f"PDF pair mismatch: {prob_info['exam_date']} vs {sol_info['exam_date']}")
        
        # Create exam metadata
        exam_metadata = {
            'exam_date': prob_info['exam_date'],
            'exam_year': prob_info['exam_year'],
            'exam_month': prob_info['exam_month'],
            'exam_day': prob_info['exam_day'],
            'exam_type': prob_info['exam_type'],
            'problems_file': os.path.basename(problems_pdf),
            'solutions_file': os.path.basename(solutions_pdf),
            'base_name': prob_info['base_name']
        }
        
        logger.info(f"Processing exam pair: {exam_metadata['exam_date']} ({exam_metadata['exam_type']})")
        
        try:
            # Step 1: Convert PDFs to page images
            logger.info("Step 1: Converting PDFs to page images")
            
            # Create temporary directories
            temp_base = os.path.join(settings.TEMP_DIR, f"processing_{exam_metadata['base_name']}_{int(time.time())}")
            problems_image_dir = os.path.join(temp_base, "problems")
            solutions_image_dir = os.path.join(temp_base, "solutions")
            content_image_dir = os.path.join(temp_base, "content")
            
            os.makedirs(problems_image_dir, exist_ok=True)
            os.makedirs(solutions_image_dir, exist_ok=True)
            os.makedirs(content_image_dir, exist_ok=True)
            
            # Convert PDFs to page images
            problems_pages = self.pdf_converter.convert_pdf_to_images(problems_pdf, problems_image_dir)
            solutions_pages = self.pdf_converter.convert_pdf_to_images(solutions_pdf, solutions_image_dir)
            
            logger.info(f"Converted {len(problems_pages)} problem pages and {len(solutions_pages)} solution pages")
            
            # Step 2: Parse answer key from first solution page
            logger.info("Step 2: Parsing answer key from solutions")
            
            answer_key = {}
            if solutions_pages:
                answer_key = self.solution_parser.parse_answer_key(solutions_pages[0])
                logger.info(f"Parsed {len(answer_key)} answers from answer key")
            
            # Convert answer key to expected format
            solutions_result = self.gpt_extractor.extract_answers_from_answer_key(answer_key)
            
            # Step 3: Segment problem pages into individual problems
            logger.info("Step 3: Segmenting pages into individual problems")
            
            all_problems = []
            all_problem_images = []
            
            for page_idx, page_image_path in enumerate(problems_pages):
                if self._interrupted:
                    break
                    
                page_key = f"page_{page_idx + 1:03d}"
                logger.info(f"Processing {page_key}: {os.path.basename(page_image_path)}")
                
                # Segment page into individual problems
                problems = self.problem_segmenter.segment_page_into_problems(
                    page_image_path, content_image_dir, page_key
                )
                
                all_problems.extend(problems)
                logger.info(f"Segmented {len(problems)} problems from {page_key}")
            
            # Step 4: Extract mathematical content from each problem
            logger.info("Step 4: Extracting mathematical content from problems")
            
            for problem in all_problems:
                if self._interrupted:
                    break
                    
                problem_key = f"{problem['page_key']}_problem_{problem['number']:02d}"
                
                # Extract mathematical content within this problem
                math_content_images = self.math_content_extractor.extract_mathematical_content(
                    problem['image_path'], content_image_dir, problem_key
                )
                
                problem['math_content_images'] = math_content_images
                logger.debug(f"Extracted {len(math_content_images)} mathematical content pieces from problem {problem['number']}")
            
            # Step 5: Process each problem with GPT
            logger.info("Step 5: Processing problems with GPT")
            
            extracted_problems = []
            
            for problem in all_problems:
                if self._interrupted:
                    break
                    
                logger.info(f"Processing problem {problem['number']} with GPT")
                
                # Extract problem data using GPT
                problem_data = self.gpt_extractor.extract_problem_from_image(
                    problem['image_path'], 
                    problem.get('math_content_images', [])
                )
                
                if 'error' not in problem_data:
                    # Add image references
                    problem_images = [problem['filename']]  # Main problem image
                    if problem.get('math_content_images'):
                        problem_images.extend([os.path.basename(img) for img in problem['math_content_images']])
                    
                    problem_data['images'] = problem_images
                    extracted_problems.append(problem_data)
                    logger.debug(f"Successfully extracted problem {problem['number']}")
                else:
                    logger.error(f"Failed to extract problem {problem['number']}: {problem_data.get('error')}")
            
            # Step 6: Match problems with solutions
            logger.info("Step 6: Matching problems with solutions")
            
            # Convert to expected format for matching
            problems_results = [{'problems': extracted_problems}]
            solutions_results = [solutions_result]
            
            matched_problems, matching_report = self.gpt_extractor.match_problems_and_solutions_scoped(
                problems_results, solutions_results, exam_metadata
            )
            
            # Step 7: Save to database
            logger.info("Step 7: Saving to database")
            
            if matched_problems:
                inserted_count = self.db_saver.insert_problems_batch(matched_problems)
                logger.info(f"Successfully inserted {inserted_count} problems into database")
            else:
                logger.warning("No problems to insert into database")
            
            # Step 8: Save outputs if requested
            if self.save_images:
                logger.info("Step 8: Saving processed images")
                self._save_processed_images(temp_base, exam_metadata)
            
            if self.save_json:
                logger.info("Step 8: Saving JSON results")
                self._save_json_results(matched_problems, matching_report, exam_metadata)
            
            # Calculate final results
            processing_time = time.time() - start_time
            
            result = {
                'exam_metadata': exam_metadata,
                'processing_time': processing_time,
                'status': 'success',
                'problems_processed': len(all_problems),
                'problems_extracted': len(extracted_problems),
                'problems_matched': len(matched_problems),
                'problems_inserted': inserted_count if matched_problems else 0,
                'matching_report': matching_report,
                'answer_key_parsed': len(answer_key)
            }
            
            logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Summary: {result['problems_processed']} problems processed, "
                       f"{result['problems_inserted']} inserted to database")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF pair: {e}")
            return {
                'exam_metadata': exam_metadata,
                'processing_time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }
        finally:
            # Clean up temporary files
            if not self.save_images:
                self._cleanup_temp_files(temp_base)