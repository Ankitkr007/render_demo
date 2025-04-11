# agents/cv_parser.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import json
import re

class CVParser:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        
    def parse(self, cv_text: str) -> dict:
        prompt = ChatPromptTemplate.from_template("""
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        Extract CV data into this JSON format:
        {{
            "name": string, 
            "email": string,
            "education": list of degrees,
            "experience": list of positions,
            "skills": list,
            "certifications": list
        }}
        <|start_header_id|>user<|end_header_id|>
        CV Content: {cv_text}
        """)
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({"cv_text": cv_text})
            print(f"CV Parser Response: {response}")
            
            # Extract JSON content from the response
            result_text = response.content if hasattr(response, 'content') else str(response)
            
            # Try to find JSON in the response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                
                # Remove comments (// style)
                json_str = re.sub(r'//.*?\n', '\n', json_str)
                # Remove any trailing commas before closing brackets
                json_str = re.sub(r',(\s*[\]}])', r'\1', json_str)
                
                print(f"Cleaned CV JSON: {json_str}")
                
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    print(f"CV JSON Parse Error: {e} in {json_str}")
            
            # Fallback with placeholder data
            return {
                "name": "Unknown Name",
                "email": "unknown@example.com",
                "education": [],
                "experience": [],
                "skills": [],
                "certifications": []
            }
        except Exception as e:
            print(f"CV Parser Error: {str(e)}")
            # Return placeholder data on error
            return {
                "name": "Unknown Name",
                "email": "unknown@example.com",
                "education": [],
                "experience": [],
                "skills": [],
                "certifications": []
            }