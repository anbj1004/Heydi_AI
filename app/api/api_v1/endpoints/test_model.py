from fastapi import APIRouter
from google import genai
from google.genai import types
from pydantic import BaseModel

router = APIRouter()
client = genai.Client()

class LLMRequest(BaseModel):
    content: str
    
@router.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint."}

@router.post("/llm")
def llm_endpoint(request: LLMRequest):
    config = types.GenerateContentConfig(
            system_instruction="You're a summary bot. Create a summary tag for the given content in one or two words. the number of tags should be 1 or 2. structure: (number of tags),(tag1),(tag2),...",
        )
    import time
    start_time = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=request.content
    )
    elapsed_time = time.time() - start_time
    return {"response": response.text}

@router.post("/emotion")
def emotion_endpoint(request: LLMRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You're an emotion analysis bot. Analyze the emotion of the given content and respond with one word representing the emotion (happy, joy, neutral, sad, annoyed, angry).",
        ),
        contents=request.content
    )
    return {"response": response.text}
