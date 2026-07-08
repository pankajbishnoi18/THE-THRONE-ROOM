from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages,RemoveMessage
from operator import add
from .tools import breaker_chain,retrieve_info,researcher_chain,hand_chain,get_situation
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage
from json_repair import repair_json
import json


class LordHandState(TypedDict):
    situation: str
    
    messages: Annotated[list, add_messages]

    research_notes:Annotated[list,add]
    subqueries:list[str]
    current_query:str
    current_query_index:int
    done:bool
    final_answer:str

def breaker_node(state:LordHandState):
    resp=breaker_chain.invoke({
    "situation":state["situation"]
   })
    
    queries = repair_json(resp.content, return_objects=True)

    if not isinstance(queries, list):
        raise ValueError(f"Breaker failed to return a list.\nReturned: {queries}")
    return {
        "subqueries":queries,
        "current_query":queries[0],
        "current_query_index":0,
        "done":False
    }



def researcher_node(state: LordHandState):

    response = researcher_chain.invoke({
        "situation":state["situation"],
        "query": state["current_query"],
        "messages": state["messages"]
    })

    update = {
        "messages": [response]
    }

    if response.tool_calls:
        return update

    update["research_notes"] = [
        {
            "query": state["current_query"],
            "answer": response.content
        }
    ]

    return update

tool_node = ToolNode([retrieve_info])

def controller_node(state: LordHandState):
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

def route(state: LordHandState):
    if state["done"]:
        return "lord_hand"

    return "researcher"
   

def lord_hand_node(state:LordHandState):
    notes_text = "\n\n".join(
        f"Q: {n['query']}\nA: {n['answer']}" for n in state["research_notes"]
    )
    response=hand_chain.invoke(
        {   "situation":state["situation"],
            
            "research_notes":notes_text

            })
    return {
            "final_answer":response.content
        }

graph = StateGraph(LordHandState)

graph.add_node("breaker", breaker_node)
graph.add_node("researcher", researcher_node)
graph.add_node("tools", tool_node)
graph.add_node("controller", controller_node)
graph.add_node("lord_hand", lord_hand_node)

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
        "lord_hand": "lord_hand",
    },
)

graph.add_edge("lord_hand", END)
app = graph.compile()




def report(situation:str):
    s=get_situation(situation)
    advice=app.invoke({
        "situation":situation
    },
    config={"recursion_limit": 60})
    return advice
