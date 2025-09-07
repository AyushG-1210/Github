from llama_cpp import Llama
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
model = os.path.join(REPO_ROOT, "models", "llama-2-7b-chat.Q4_K_M.gguf")

#load model 
llm = Llama(
    model_path=model,
    n_ctx=2048,
    n_threads=8
)

#store chat history
conv_hist = [
    "The following is a helpful conversation between a user and an AI assistant."
]


def ask_llm(user_input: str) -> str:
    """Chat-style LLM with memory of previous turns, capped to 35 words."""
    global conv_hist

    #add new user message to history
    conv_hist.append(f"User: {user_input}")

    #build full prompt
    prompt = "\n".join(conv_hist) + "\nAI:"

    #generate response
    res = llm(
        prompt,
        max_tokens=256,
        stop=["User:", "AI:"],
        echo=False
    )
    reply = res["choices"][0]["text"].strip()

    #50-word cap
    words = reply.split()
    if len(words) > 50:
        reply = " ".join(words[:50]) + "..."

    #add AI reply to history
    conv_hist.append(f"AI: {reply}")

    #trim conversation 
    if len(conv_hist) > 25:
        conv_hist = conv_hist[:1] + conv_hist[-20:]

    return reply
