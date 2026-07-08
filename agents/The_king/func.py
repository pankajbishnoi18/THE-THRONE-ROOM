from langchain_ollama import ChatOllama
from .prompts import king_prompt
llm=ChatOllama(model="llama3.2:3b",temperature=0.0)
king_chain=king_prompt|llm