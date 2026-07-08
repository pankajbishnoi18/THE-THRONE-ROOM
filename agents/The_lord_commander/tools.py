from langchain_core.tools import tool
from .RAG_implementation import read_file,retrieve
from pathlib import Path
from langchain_ollama import ChatOllama
from .prompts import breaker_prompt,research_prompt,commander_prompt

situation=""
def get_situation(s):
    global situation
    situation=s
    return situation

@tool
def retrieve_info(search_question: str):
    """
    Search the Royal Archives.

    search_question MUST be the COMPLETE research question.
    Do not shorten, summarize, or convert it into keywords.
    Pass the entire question exactly as written.
    """
    
    try:
        
        content=retrieve(situation,search_question)
    except Exception as e:
        print("ERROR:", type(e).__name__)
        print(e)
        raise


    if content=="empty" :
        return "empty"
    if len(content)==1:
        return f"content1:{content[0]}"
    return f"content1:{content[0]},\ncontent2:{content[1]}"


tools=[retrieve_info]

llm=ChatOllama(model="llama3.2:3b",temperature=0.0)
llm_tool=llm.bind_tools(tools)
researcher_chain=research_prompt | llm_tool
breaker_chain=breaker_prompt | llm
commander_chain=commander_prompt | llm 




