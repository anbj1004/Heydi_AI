from google import genai
from google.genai import types

class Topic:
    count: int
    tag1: str
    tag2: str

def extract_topics_from_text(text: str) -> Topic:
    client = genai.Client()
    config = types.GenerateContentConfig(
            system_instruction="You're a summary bot. Create a summary tag for the given content in one or two words in korean. the number of tags should be 1 or 2. structure: (number of tags),(tag1),(tag2),... if there is no tag2, leave it blank. example: 2,기술,인공지능",
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=config,
        contents=text
    )

    topic = Topic()
    s = response.text.strip().split(",")
    topic.count = int(s[0])
    topic.tag1 = s[1]
    topic.tag2 = s[2] if topic.count == 2 else ""

    return topic

if __name__ == "__main__":
    while True:
        text = input("Enter text to extract topics (or 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        topic = extract_topics_from_text(text)
        print(f"Extracted Topics: count={topic.count}, tag1={topic.tag1}, tag2={topic.tag2}")