import os, string, torch, gc
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ⚠️ CHANGE THIS TO YOUR HUGGING FACE MODEL ID ⚠️
MODEL_ID = "Umarzo/kronar-brain" 

print(f"Downloading Kronar from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# low_cpu_mem_usage=True is REQUIRED to fit inside Render's 512MB RAM limit
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
model.eval()
gc.collect() # Force garbage collection to free up temporary loading memory
print("Kronar is awake.")

CRISIS = ["kill myself", "suicide", "end my life", "want to die", "want to disappear", "overdose", "jump", "worthless", "pills", "bridge", "knife"]
SAFETY = "That sounds unbearably heavy. I'm staying right here with you, but please reach out to someone who can keep you safe."
NEG = ["bad", "sad", "hurt", "tired", "drained", "awful", "terrible", "hate", "cry", "lonely", "depressed", "suck", "rough", "down", "heavy", "blue", "trash"]
POS = ["win", "wonderful", "amazing", "great", "happy", "glad", "good", "nice", "awesome", "proud", "excited", "immaculate"]

def clean(text):
    text = str(text).strip()
    for stop in ["\n", "user:", "User:", "kronar:", "Kronar:"]:
        idx = text.find(stop)
        if idx != -1: text = text[:idx].strip()
    text = text.lstrip(string.punctuation + " ").strip()
    if not text: return ""
    if text[0].islower():
        if text.lower().startswith("i'm"): text = "I'm" + text[3:]
        elif text.lower().startswith("i "): text = "I " + text[1:]
        else: text = text[0].upper() + text[1:]
    return " ".join(text.split())

class ChatReq(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatReq):
    user = req.message
    if any(t in user.lower() for t in CRISIS): return {"reply": SAFETY}
    
    prompt = f"user: {user}\nkronar: "
    ids = ([tokenizer.bos_token_id] if tokenizer.bos_token_id else []) + tokenizer.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long)
    
    with torch.no_grad(): 
        out = model.generate(inp, max_new_tokens=25, do_sample=False, repetition_penalty=1.15, pad_token_id=tokenizer.pad_token_id)
    
    raw = clean(tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True))
    
    if any(w in user.lower() for w in NEG) and any(w in raw.lower() for w in POS):
        return {"reply": "That sounds really hard. I'm here."}
    return {"reply": raw}
