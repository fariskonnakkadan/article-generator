## Gemini API Library Generator

This script generates a comprehensive library of articles on any topic using Google's Gemini API.
It creates a three-tier structure: Initial Prompt → Configurable Main Topics → Configurable Subtopics per Topic → Articles

### Usage:
    `python gemini_library_generator.py --api-key YOUR_API_KEY --prompt "Your initial prompt" --topic-first-count 10 --topic-second-count 20`

### Requirements:
    `pip install google-generativeai requests`
