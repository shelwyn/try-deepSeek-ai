from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os
import re
import json
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

class SecurityConfig:
    # Input validation
    MAX_MESSAGE_LENGTH = 2500
    MIN_MESSAGE_LENGTH = 1
    BLOCKED_KEYWORDS = {
        'sql', 'exec', 'eval', 'system', 'os.', 'subprocess', 
        'rm -rf', 'format', 'delete', 'drop table'
    }
    
    # Content moderation
    SENSITIVE_TOPICS = {
        'explicit_content': ['porn', 'xxx', 'nsfw'],
        'hate_speech': ['hate', 'racist', 'discrimination'],
        'violence': ['kill', 'murder', 'attack'],
        'personal_info': ['ssn', 'credit card', 'passport']
    }
    
    # Output validation
    MAX_RESPONSE_LENGTH = 4096
    RESTRICTED_PATTERNS = [
        r'(?i)(password|secret|key):\s*\w+',  # Sensitive data patterns
        r'(?i)(<script>|javascript:)',         # XSS patterns
        r'(?i)(SELECT|INSERT|UPDATE|DELETE)\s+FROM' # SQL patterns
    ]

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=SecurityConfig.MIN_MESSAGE_LENGTH, 
                        max_length=SecurityConfig.MAX_MESSAGE_LENGTH)
    context: Optional[str] = Field(None, max_length=500)
    
    @validator('message')
    def validate_message_content(cls, v):
        # Check for blocked keywords
        lower_message = v.lower()
        for keyword in SecurityConfig.BLOCKED_KEYWORDS:
            if keyword in lower_message:
                raise ValueError(f"Message contains prohibited content")
        
        # Check for sensitive topics
        for category, terms in SecurityConfig.SENSITIVE_TOPICS.items():
            if any(term in lower_message for term in terms):
                raise ValueError(f"Message contains inappropriate content")
        
        return v

class ResponseFilter:
    @staticmethod
    def filter_output(response: str) -> str:
        # Check response length
        if len(response) > SecurityConfig.MAX_RESPONSE_LENGTH:
            response = response[:SecurityConfig.MAX_RESPONSE_LENGTH] + "..."
        
        # Check for restricted patterns
        for pattern in SecurityConfig.RESTRICTED_PATTERNS:
            if re.search(pattern, response):
                response = re.sub(pattern, "[FILTERED]", response)
        
        return response

class PromptEngineering:
    SYSTEM_PROMPT = """You are a helpful AI assistant. Please follow these rules:
    1. Do not generate harmful, explicit, or inappropriate content
    2. Do not reveal personal information or sensitive data
    3. Do not execute commands or code
    4. Provide factual and helpful information only
    5. Maintain a respectful and professional tone
    6. Do not engage in harmful or malicious activities
    """
    
    @staticmethod
    def create_safe_prompt(user_message: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": PromptEngineering.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        if context:
            messages.insert(1, {
                "role": "system",
                "content": f"Context: {context}"
            })
        
        return messages

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Restrict to necessary methods
    allow_headers=["*"],
)

# Initialize HuggingFace client
client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    api_key=os.getenv('HF_TOKEN')
)

async def log_request(message: str):
    logger.info(f"Request received at {datetime.now()}: {message[:100]}...")

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # Log the request
        await log_request(chat_message.message)
        
        # Create safe prompt with guardrails
        messages = PromptEngineering.create_safe_prompt(
            chat_message.message,
            chat_message.context
        )
        
        # Get completion from the model
        completion = client.chat.completions.create(
            messages=messages,
            max_tokens=2048,
            temperature=0.7  # Add controlled randomness
        )
        
        # Extract and filter the response
        raw_response = completion.choices[0].message.content.strip()
        print("XX"*10,raw_response)
        filtered_response = ResponseFilter.filter_output(raw_response)
        
        # Log the response
        logger.info(f"Response generated successfully for request")
        
        return {
            "response": filtered_response,
            "filtered": filtered_response != raw_response
        }
    
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
