from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv('secrets.env')

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize HuggingFace client
client = InferenceClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    api_key=os.getenv('HF_TOKEN')
)

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # Prepare the message format for the model
        messages = [
            {
                "role": "user",
                "content": chat_message.message
            }
        ]

        # Get completion from the model
        completion = client.chat.completions.create(
            messages=messages,
            max_tokens=2048
        )

        # Extract and return the response
        response = completion.choices[0].message.content.strip()
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
