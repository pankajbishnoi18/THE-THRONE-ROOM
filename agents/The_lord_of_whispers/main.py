from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages,RemoveMessage
from operator import add
from tools import breaker_chain,retrieve_info,researcher_chain,whisper_chain,situation
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage

import json


class LordWhisperState(TypedDict):
    situation: str
    
    messages: Annotated[list, add_messages]

    research_notes:Annotated[list,add]
    subqueries:list[str]
    current_query:str
    current_query_index:int
    done:bool
    final_answer:str

def breaker_node(state:LordWhisperState):
    resp=breaker_chain.invoke({
    "situation":state["situation"]
   })
    
    if resp.content[0]!="[" or resp.content[-1]!="]":
        raise ValueError(
        f"Breaker did not return a JSON array.\n\nOutput:\n{resp.content}")
    queries=json.loads(resp.content)
    
    return {
        "subqueries":queries,
        "current_query":queries[0],
        "current_query_index":0,
        "done":False
    }



def researcher_node(state: LordWhisperState):
    
    already_has_tool_result = any(
        isinstance(msg, ToolMessage)
        for msg in state["messages"]
    )

    
    response = researcher_chain.invoke({
        "messages": state["messages"],
        "query": state["current_query"]
    })

   
    if response.tool_calls and not already_has_tool_result:
        return {
            "messages": [response]
        }

    
    return {
        "messages": [response],
        "research_notes": [
            {
                "query": state["current_query"],
                "answer": response.content
            }
        ]
    }

tool_node = ToolNode([retrieve_info])

def controller_node(state: LordWhisperState):
    idx = state["current_query_index"] + 1
    clear_messages = [RemoveMessage(id=m.id) for m in state["messages"]]

    if idx >= len(state["subqueries"]):
        return {
            "done": True,
            "messages": clear_messages
        }

    return {
        "done": False,
        "messages": clear_messages,
        "current_query_index": idx,
        "current_query": state["subqueries"][idx]
    }

def route(state: LordWhisperState):
    if state["done"]:
        return "lord_whisper"

    return "researcher"
   

def lord_whisper_node(state:LordWhisperState):
    notes_text = "\n\n".join(
        f"Q: {n['query']}\nA: {n['answer']}" for n in state["research_notes"]
    )
    response=whisper_chain.invoke(
        {   "situation":state["situation"],
            
            "research_notes":notes_text

            })
    return {
            "final_answer":response.content
        }

graph = StateGraph(LordWhisperState)

graph.add_node("breaker", breaker_node)
graph.add_node("researcher", researcher_node)
graph.add_node("tools", tool_node)
graph.add_node("controller", controller_node)
graph.add_node("lord_whisper", lord_whisper_node)

graph.add_edge(START, "breaker")
graph.add_edge("breaker", "researcher")

graph.add_conditional_edges(
    "researcher",
    tools_condition,
    {
        "tools": "tools",
        END: "controller",
    },
)

graph.add_edge("tools", "researcher")


graph.add_conditional_edges(
    "controller",
    route,
    {
        "researcher": "researcher",
        "lord_whisper": "lord_whisper",
    },
)

graph.add_edge("lord_whisper", END)
app = graph.compile()



response=app.invoke(
    situation,
    config={"recursion_limit": 60}
)
print(response["final_answer"])