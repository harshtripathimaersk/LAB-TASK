import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize AzureOpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

def get_text_completion(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-35-turbo", 
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

def main():
    prompt = input("Enter your prompt: ")
    completion = get_text_completion(prompt)
    print("Completion:", completion)

if __name__ == "__main__":
    main()