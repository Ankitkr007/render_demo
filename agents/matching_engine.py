# agents/matching_engine.py
import numpy as np
import torch
from typing import Union, List
import requests
import json
import time

class MatchingEngine:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/embeddings"
        
    def get_embedding(self, text: str) -> torch.Tensor:
        """Generate embeddings for text using Ollama API"""
        try:
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "prompt": text
            }
            
            # Make API call to Ollama with retry mechanism
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.ollama_url, json=payload)
                    response.raise_for_status()  # Raise an exception for 4XX/5XX responses
                    
                    # Parse the response
                    result = response.json()
                    if "embedding" in result:
                        # Convert to tensor and return
                        embedding = torch.tensor(result["embedding"], dtype=torch.float32)
                        print(f"Embedding shape: {embedding.shape}")
                        return embedding
                    else:
                        print(f"Unexpected response format: {result}")
                        break
                except requests.exceptions.RequestException as e:
                    print(f"Request failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        time.sleep(2 ** attempt)
                    else:
                        raise
            
            # If we get here, all retries failed or response was invalid
            print("Failed to get embedding after retries")
            return torch.zeros(768)  # Return zero tensor as fallback
            
        except Exception as e:
            print(f"Embedding error: {str(e)}")
            # Return zero tensor of expected shape as fallback
            return torch.zeros(768)
    
    def calculate_match(self, jd_embedding: torch.Tensor, cv_embedding: torch.Tensor) -> float:
        """Calculate match score between job description and CV using cosine similarity"""
        try:
            # Normalize embeddings
            jd_embedding_norm = jd_embedding / jd_embedding.norm()
            cv_embedding_norm = cv_embedding / cv_embedding.norm()
            
            # Calculate cosine similarity using dot product of normalized vectors
            similarity = torch.dot(jd_embedding_norm, cv_embedding_norm).item()
            
            # Convert to percentage (0-100)
            score = min(max(float(similarity * 100), 0), 100)
            print(f"Match score: {score}")
            return score
        except Exception as e:
            print(f"Match calculation error: {str(e)}")
            