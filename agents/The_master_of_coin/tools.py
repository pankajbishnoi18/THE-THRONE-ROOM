from langchain_core.tools import tool
from RAG_implementation import read_file,retrieve
from pathlib import Path
from langchain_ollama import ChatOllama
from prompts import breaker_prompt,research_prompt,coin_prompt


situation={
    "situation": """

For the past ten days, merchant caravans traveling between Ashvale and Crownhaven have reported repeated attacks near Eagle's Crossing. Although the attackers stole only grain wagons, no merchants were killed, suggesting the raids were carefully planned rather than acts of random banditry.

As a result, several merchants have postponed shipments until the route is declared safe. Grain prices in Crownhaven have risen modestly, while warehouses in Ashvale are beginning to accumulate unsold stock.

The Governor of Ashvale has requested additional patrols along the King's Road. Meanwhile, the Governor of Riverwatch argues that repairing a damaged bridge on the River Road would restore an alternative supply route more quickly than deploying soldiers.


"""
}

@tool
def retrieve_info(search_question: str):
    """
    Search the Royal Archives.

    search_question MUST be the COMPLETE research question.
    Do not shorten, summarize, or convert it into keywords.
    Pass the entire question exactly as written.
    """
    
    try:
        print(search_question)
        content=retrieve(situation["situation"],search_question)
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
coin_chain=coin_prompt | llm 




