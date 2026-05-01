import requests

url = "http://192.168.19.1:1234/v1/chat/completions"
API_URL = "http://192.168.19.1:1234/v1/chat/completions"

def generate_mistral_response(prompt, max_tokens=2000):
    print("[🚀] Generating response from Mistral-7B-Instruct via LM Studio...")
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.9
    }

    response = requests.post(url, headers=headers, json=data)
    
    print("Response status code:", response.status_code)
    print("Response text:", response.text)

    answer = response.json()["choices"][0]["message"]["content"]
    print(f"[✅] Response: {answer[:200]}...")
    return answer

def ask_mistral(query, context_passages):
    combined_context = "\n".join(context_passages[:7])
    prompt = (
        f"You are an expert in education. Based on the following context, answer the question clearly and concisely:\n\n"
        f"Context:\n{combined_context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return generate_mistral_response(prompt)
