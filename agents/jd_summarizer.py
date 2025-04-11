# agents/jd_summarizer.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import json
import re

class JDSummarizer:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.prompt = ChatPromptTemplate.from_template(""" 
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        Analyze this job description and extract:
        1. List of required technical skills
        2. Years of experience required
        3. Educational qualifications
        4. Certifications needed
        5. Key daily responsibilities

        Format response as JSON with these keys:
        required_skills, required_experience, required_education, certifications, key_responsibilities

        DO NOT include any comments in the JSON. If information is missing, use null values.

        <|start_header_id|>user<|end_header_id|>
        JOB DESCRIPTION:
        {jd_text}
        """)

    def summarize(self, jd_text: str) -> dict:
        try:
            chain = self.prompt | self.llm
            result = chain.invoke({"jd_text": jd_text})
            
            # Log the result
            print(f"JD Summarizer Response: {result}")
            
            # Extract JSON content from the response
            result_text = result.content if hasattr(result, 'content') else str(result)
            
            # Try to find JSON in the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                
                # Remove comments (// style)
                json_str = re.sub(r'//.*?\n', '\n', json_str)
                # Remove any trailing commas before closing brackets
                json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                
                print(f"Cleaned JSON: {json_str}")
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"JSON Parse Error: {e} in {json_str}")
            
            # Fallback response
            return {
                "required_skills": [],
                "required_experience": None,
                "required_education": None,
                "certifications": [],
                "key_responsibilities": []
            }
        except Exception as e:
            print(f"JD Summarizer Error: {str(e)}")
            return {
                "required_skills": [],
                "required_experience": None,
                "required_education": None,
                "certifications": [],
                "key_responsibilities": []
            }