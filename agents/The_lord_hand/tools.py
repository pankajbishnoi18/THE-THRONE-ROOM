from langchain_core.tools import tool
from RAG_implementation import read_file,retrieve
from pathlib import Path
from langchain_ollama import ChatOllama
from prompts import breaker_prompt,research_prompt,hand_prompt


@tool
def retrieve_info(situation:str,query:str):
    """
Search the Royal Archives for information relevant to the current query.
"""
    content=retrieve(situation,query) 

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
hand_chain=hand_prompt | llm 