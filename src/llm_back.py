import os
from llama_cpp import Llama

# Suppress llama.cpp logs
os.environ["GGML_LOG_LEVEL"] = "error"

# Load model once
llm = Llama(
    model_path=r"C:\Users\anshu\Desktop\Mini Project\Helmet_Env\Github\models\llama-2-7b-chat.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8
)

# Store chat history
conversation_history = [
    "The following is a helpful conversation between a user and an AI assistant."
]


def ask_llm(user_input: str) -> str:
    """Chat-style LLM with memory of previous turns, capped to 35 words."""
    global conversation_history

    # Add new user message to history
    conversation_history.append(f"User: {user_input}")

    # Build full prompt
    prompt = "\n".join(conversation_history) + "\nAI:"

    # Generate response
    response = llm(
        prompt,
        max_tokens=256,  # shorter limit since we cap to 35 words
        stop=["User:", "AI:"],
        echo=False
    )
    reply = response["choices"][0]["text"].strip()

    # Enforce 50-word cap
    words = reply.split()
    if len(words) > 50:
        reply = " ".join(words[:50]) + "..."

    # Add AI reply to history
    conversation_history.append(f"AI: {reply}")

    # Trim conversation to avoid slowdown (keep last 10 turns)
    if len(conversation_history) > 25:
        conversation_history = conversation_history[:1] + conversation_history[-20:]

    return reply
