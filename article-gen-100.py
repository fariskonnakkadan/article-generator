#!/usr/bin/env python3
"""
Gemini API Library Generator

This script generates a comprehensive library of articles on any topic using Google's Gemini API.
It creates a three-tier structure: Initial Prompt → 100 Main Topics → 100 Subtopics per Topic → Articles

Usage:
    python gemini_library_generator.py --api-key YOUR_API_KEY --prompt "Your initial prompt"

Requirements:
    pip install google-generativeai requests
"""

import argparse
import os
import time
import json
import re
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GeminiLibraryGenerator:
    def __init__(self, api_key: str, rate_limit_delay: float = 2.0):
        """Initialize the Gemini Library Generator.
        
        Args:
            api_key: Google Gemini API key
            rate_limit_delay: Delay between API calls in seconds (default 2.0 for safety)
        """
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self.setup_gemini()
        
    def setup_gemini(self):
        """Configure Gemini API client."""
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Safety settings to allow educational content
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def sanitize_filename(self, text: str) -> str:
        """Convert text to a safe filename."""
        # Remove special characters and replace spaces with underscores
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text.strip('_').lower()[:100]  # Limit length
    
    def make_api_call(self, prompt: str, max_retries: int = 3) -> str:
        """Make API call to Gemini with rate limiting and retry logic."""
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    safety_settings=self.safety_settings,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=8192,
                    )
                )
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                if response.text:
                    return response.text.strip()
                else:
                    print(f"Empty response on attempt {attempt + 1}")
                    
            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(self.rate_limit_delay * 2)  # Longer delay on retry
                    
        raise Exception("Failed to get response after all retries")
    
    def generate_main_topics(self, initial_prompt: str) -> List[str]:
        """Generate 100 main topics from the initial prompt."""
        print("🎯 Generating 100 main topics...")
        
        prompt = f"""
Based on the following request: "{initial_prompt}"

Generate exactly 100 distinct, specific, and comprehensive topics that would be valuable for someone interested in this subject. Each topic should be:
1. Specific enough to warrant detailed exploration
2. Broad enough to have multiple subtopics
3. Educational and informative
4. Unique from the other topics

Format your response as a numbered list from 1 to 100, with each topic on a new line.
Example format:
1. Topic Name One
2. Topic Name Two
...
100. Topic Name One Hundred

Do not include any additional text, explanations, or formatting - just the numbered list.
"""
        
        response = self.make_api_call(prompt)
        
        # Parse the topics
        topics = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('•', '-', '*'))):
                # Extract topic name (remove numbering and bullets)
                topic = re.sub(r'^\d+\.?\s*', '', line)
                topic = re.sub(r'^[•\-*]\s*', '', topic)
                if topic:
                    topics.append(topic.strip())
        
        if len(topics) < 50:  # Fallback if parsing failed
            print("⚠️ Topic parsing may have failed. Using fallback method...")
            topics = [line.strip() for line in lines if line.strip()]
        
        print(f"✅ Generated {len(topics)} main topics")
        return topics[:100]  # Ensure we don't exceed 100
    
    def generate_subtopics(self, main_topic: str, context: str) -> List[str]:
        """Generate 100 subtopics for a given main topic."""
        print(f"📚 Generating subtopics for: {main_topic}")
        
        prompt = f"""
Context: {context}
Main Topic: "{main_topic}"

Generate exactly 100 specific subtopics for this main topic. Each subtopic should be:
1. Specific and focused enough for a detailed article
2. Educational and informative
3. Unique and non-overlapping with other subtopics
4. Relevant to the main topic and overall context

Format your response as a numbered list from 1 to 100, with each subtopic on a new line.
Example format:
1. Subtopic Name One
2. Subtopic Name Two
...
100. Subtopic Name One Hundred

Do not include any additional text, explanations, or formatting - just the numbered list.
"""
        
        response = self.make_api_call(prompt)
        
        # Parse the subtopics
        subtopics = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith(('•', '-', '*'))):
                # Extract subtopic name
                subtopic = re.sub(r'^\d+\.?\s*', '', line)
                subtopic = re.sub(r'^[•\-*]\s*', '', subtopic)
                if subtopic:
                    subtopics.append(subtopic.strip())
        
        if len(subtopics) < 50:  # Fallback if parsing failed
            subtopics = [line.strip() for line in lines if line.strip()]
        
        print(f"  ✅ Generated {len(subtopics)} subtopics")
        return subtopics[:100]
    
    def generate_article(self, main_topic: str, subtopic: str, context: str) -> str:
        """Generate a comprehensive article for a subtopic."""
        prompt = f"""
Context: {context}
Main Topic: {main_topic}
Subtopic: {subtopic}

Write a comprehensive, well-structured article about "{subtopic}" in the context of "{main_topic}". 

The article should be:
1. Educational and informative (1500-2500 words)
2. Well-structured with clear headings and subheadings
3. Written in markdown format
4. Include practical examples where applicable
5. Be engaging and accessible to readers
6. Include relevant technical details without being overly complex

Structure the article with:
- A compelling title
- Introduction
- Main content with appropriate subheadings
- Key points or takeaways
- Conclusion

Write the article in markdown format, ready to be saved as a .md file.
"""
        
        return self.make_api_call(prompt)
    
    def create_directory_structure(self, base_path: str, main_topics: List[str]) -> Dict[str, str]:
        """Create directory structure for the library."""
        base_dir = Path(base_path)
        base_dir.mkdir(exist_ok=True)
        
        topic_dirs = {}
        for topic in main_topics:
            safe_topic_name = self.sanitize_filename(topic)
            topic_dir = base_dir / safe_topic_name
            topic_dir.mkdir(exist_ok=True)
            topic_dirs[topic] = str(topic_dir)
        
        return topic_dirs
    
    def save_article(self, article_content: str, file_path: str):
        """Save article content to a markdown file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(article_content)
        except Exception as e:
            print(f"❌ Error saving article to {file_path}: {str(e)}")
    
    def save_progress(self, data: dict, progress_file: str):
        """Save progress to JSON file for resuming if needed."""
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Could not save progress: {str(e)}")
    
    def generate_library(self, initial_prompt: str, output_dir: str = "gemini_library"):
        """Generate the complete library."""
        print(f"🚀 Starting library generation for: '{initial_prompt}'")
        print(f"📁 Output directory: {output_dir}")
        
        # Create base directory
        base_path = Path(output_dir)
        base_path.mkdir(exist_ok=True)
        
        # Step 1: Generate main topics
        main_topics = self.generate_main_topics(initial_prompt)
        
        # Create directory structure
        topic_dirs = self.create_directory_structure(output_dir, main_topics)
        
        # Progress tracking
        progress_file = base_path / "generation_progress.json"
        total_articles = len(main_topics) * 100
        completed_articles = 0
        
        print(f"📊 Total articles to generate: {total_articles}")
        print(f"⏰ Estimated time (with rate limiting): {total_articles * self.rate_limit_delay / 3600:.1f} hours")
        
        # Step 2 & 3: For each main topic, generate subtopics and articles
        for topic_idx, main_topic in enumerate(main_topics, 1):
            print(f"\n🔄 Processing topic {topic_idx}/{len(main_topics)}: {main_topic}")
            
            # Generate subtopics for this main topic
            subtopics = self.generate_subtopics(main_topic, initial_prompt)
            
            # Generate articles for each subtopic
            topic_dir = Path(topic_dirs[main_topic])
            
            for subtopic_idx, subtopic in enumerate(subtopics, 1):
                try:
                    print(f"  📝 Generating article {subtopic_idx}/100: {subtopic[:50]}...")
                    
                    # Generate article
                    article_content = self.generate_article(main_topic, subtopic, initial_prompt)
                    
                    # Save article
                    safe_subtopic_name = self.sanitize_filename(subtopic)
                    article_file = topic_dir / f"{subtopic_idx:03d}_{safe_subtopic_name}.md"
                    self.save_article(article_content, str(article_file))
                    
                    completed_articles += 1
                    
                    # Progress update
                    if completed_articles % 10 == 0:
                        progress = {
                            "initial_prompt": initial_prompt,
                            "completed_articles": completed_articles,
                            "total_articles": total_articles,
                            "current_topic": main_topic,
                            "progress_percentage": (completed_articles / total_articles) * 100
                        }
                        self.save_progress(progress, str(progress_file))
                        print(f"    📈 Progress: {completed_articles}/{total_articles} ({completed_articles/total_articles*100:.1f}%)")
                    
                except Exception as e:
                    print(f"    ❌ Failed to generate article for '{subtopic}': {str(e)}")
                    continue
        
        # Create summary file
        summary_content = f"""# Library Generation Summary

**Initial Prompt:** {initial_prompt}
**Generated:** {completed_articles} articles
**Topics:** {len(main_topics)}
**Structure:** {len(main_topics)} main topics × ~100 subtopics each

## Main Topics Generated:
"""
        for i, topic in enumerate(main_topics, 1):
            summary_content += f"{i}. {topic}\n"
        
        summary_file = base_path / "README.md"
        self.save_article(summary_content, str(summary_file))
        
        print(f"\n🎉 Library generation completed!")
        print(f"📊 Generated {completed_articles} articles across {len(main_topics)} topics")
        print(f"📁 Library saved to: {output_dir}")
        print(f"📄 Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive library using Gemini API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gemini_library_generator.py --api-key YOUR_KEY --prompt "Create a library for a science lover"
  python gemini_library_generator.py --api-key YOUR_KEY --prompt "Programming and software development" --output-dir my_library
        """
    )
    
    parser.add_argument(
        "--api-key",
        required=True,
        help="Google Gemini API key"
    )
    
    parser.add_argument(
        "--prompt",
        required=True,
        help="Initial prompt describing the library topic"
    )
    
    parser.add_argument(
        "--output-dir",
        default="gemini_library",
        help="Output directory for the generated library (default: gemini_library)"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=2.0,
        help="Delay between API calls in seconds (default: 2.0)"
    )
    
    args = parser.parse_args()
    
    # Validate API key format (basic check)
    if not args.api_key or len(args.api_key) < 10:
        print("❌ Invalid API key. Please provide a valid Gemini API key.")
        return 1
    
    try:
        # Initialize generator
        generator = GeminiLibraryGenerator(
            api_key=args.api_key,
            rate_limit_delay=args.rate_limit
        )
        
        # Generate library
        generator.generate_library(args.prompt, args.output_dir)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
