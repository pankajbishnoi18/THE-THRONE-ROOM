from langchain_core.tools import tool
from function import read_file,retrieve
from pathlib import Path
from langchain_ollama import ChatOllama
from prompts import prompt


@tool
def retrieve_info(query:str):
    """
Search the Royal Archives for information relevant to the current situation.

Use this tool whenever historical records, governance procedures,
political precedents, diplomatic information,
administrative rules or succession laws are required.

Input:
A natural language description of the information you need.

Output:
The most relevant archival documents.
"""
    content=retrieve(query)
    if not content:
        return f"Search status:NO_MATCH,Content:{content},Reason:there are no relevent docs"

    return f"Search status:MATCH_FOUND,Content:{content}"
tools=[retrieve_info]
llm=ChatOllama(model="llama3.2:3b",temperature=0.2)
llm_tool=llm.bind_tools(tools)
chain=prompt | llm_tool





        

