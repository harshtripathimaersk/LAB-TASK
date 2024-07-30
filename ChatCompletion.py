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

def chat_with_openai(prompt):
    try:
        response = client.chat.completions.create(
        model="gpt-4",  
         messages=[
                {"role": "user", "content": prompt}
            ],
        max_tokens=150
    )
        ai_response = response.choices[0].message.content
        tokens_used=response.usage.total_tokens
        return ai_response, tokens_used
    except Exception as e:
        return str(e),0

def calculate_cost(total_tokens, cost_per_token):
    return total_tokens * cost_per_token

def main():
    conversation_history = ""
    total_tokens = 0
    cost_per_token = 0.0004  # Example cost per token (adjust based on actual pricing)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        conversation_history += f"You: {user_input}\n"
        ai_response, tokens_used = chat_with_openai(conversation_history)
        conversation_history += f"AI: {ai_response}\n"

        if isinstance(ai_response, str) and tokens_used == 0:
                print(f"Error: {ai_response}")
                break
        total_tokens += tokens_used
        
        print(f"AI: {ai_response}")
        print(f"Tokens used in this response: {tokens_used}")
        print(f"Total tokens used so far: {total_tokens}")
        print(f"Estimated cost so far: ${calculate_cost(total_tokens, cost_per_token):.4f}")

if __name__ == "__main__":
    main()
