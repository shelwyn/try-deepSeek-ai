# Simple LLM Chatbot

A lightweight chatbot implementation using DeepSeek's language model, FastAPI backend, and a simple HTML/JavaScript frontend.

## Project Description

This project implements a simple chatbot that uses:
- Frontend: Single HTML file with vanilla JavaScript
- Backend: FastAPI server that interfaces with HuggingFace's DeepSeek model
- Model: DeepSeek-R1-Distill-Qwen-32B via HuggingFace Inference API

The chatbot features:
- Clean, WhatsApp-like UI
- Real-time typing indicators
- Formatted LLM responses with proper styling for mathematical expressions
- Error handling and response validation
- Cross-Origin Resource Sharing (CORS) enabled

## Prerequisites

- Python 3.8+
- HuggingFace API Token with Inference permissions

## Setup Instructions

1. **Get Your HuggingFace Token**
   - Go to: https://huggingface.co/settings/tokens
   - Click "Create new token"
   - Enable the following permissions under "User permissions/Inference":
     * Make calls to the serverless Inference API
     * Make calls to Inference Endpoints
     * Manage Inference Endpoints
   - Copy your token

2. **Set Up Environment**
   ```bash
   # Create and activate virtual environment (optional but recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install required packages
   pip install fastapi uvicorn huggingface-hub python-dotenv
   ```

3. **Configure Environment Variables**
   - Create a file named `.env` in the project root
   - Add your HuggingFace token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Start the Backend Server**
   ```bash
   python main.py
   # Or alternatively:
   uvicorn main:app --reload
   ```

5. **Launch the Frontend**
   - Open `index.html` in a web browser
   - Or serve it using a simple HTTP server:
   ```bash
   python -m http.server 3000
   ```
   Then visit `http://localhost:3000`

## Project Structure

```
chatbot/
├── main.py           # FastAPI backend server
├── index.html        # Frontend UI
├── .env       # Environment variables (create this)
└── README.md         # This file
```

## Usage

1. Once both backend and frontend are running, open the chatbot in your browser
2. Type your message in the input field
3. Press Enter or click Send
4. The bot will respond with formatted text, including:
   - Thought process in styled blocks
   - Mathematical expressions in dedicated blocks
   - Steps clearly separated and formatted
   - Final answers highlighted

## Error Handling

The chatbot includes error handling for:
- Network connectivity issues
- API token validation
- Model response formatting
- Server errors

If any errors occur, they will be displayed in the chat interface and logged to the console.

## Notes

- The frontend runs on vanilla JavaScript without any external dependencies
- The backend uses FastAPI for efficient async request handling
- CORS is enabled for development purposes
- The model may take a few seconds to respond depending on the query complexity
- Responses are formatted to handle mathematical notation and structured thinking steps

## Security Notes

- Never commit your `secrets.env` file
- The CORS settings in the backend are set to allow all origins for development
- For production, configure proper CORS settings and add appropriate security measures
