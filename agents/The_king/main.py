from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from agents.The_lord_commander.main import report as commander_response
from agents.The_lord_hand.main import report as hand_response
from agents.The_lord_of_whispers.main import report  as whisper_response

from agents.The_master_of_coin.main import report as coin_response
from agents.The_master_of_law.main import report as law_response
from .func import king_chain

class KingState(TypedDict):
    situation:str
    lord_hand_advice:str
    lord_commander_advice:str
    lord_of_whispers_advice:str
    master_of_law_advice:str
    master_of_coin_advice:str
    final_decision:str




def lord_hand_node(state:KingState):
    advice=hand_response(state["situation"])
    print("King:What do you say Lord Hand")
    print("------->")
    print(advice["final_answer"])
    print("------->")
    return {
        "lord_hand_advice":advice["final_answer"]
    }


def lord_commander_node(state:KingState):
    advice=commander_response(state["situation"])
    print("King:Lord Commander!!")
    print("------->")
    print(f'Lord Commander:{advice["final_answer"]}')
    print("------->")
    return {
        "lord_commander_advice":advice["final_answer"]
    }


def lord_of_whispers_node(state:KingState):
    advice=whisper_response(state["situation"])
    print("King:I would like to hear your part,Lord of Whispers")
    print("------->")
    print(f'Lord of Whispers:{advice["final_answer"]}')
    print("------->")
    return {
        "lord_of_whispers_advice":advice["final_answer"]
    }


def master_of_law_node(state:KingState):
    advice=law_response(state["situation"])
    print("King:Master of Law,what do you have to say")
    print("------->")
    print(f'Master of Law:{advice["final_answer"]}')
    print("------->")
    return {
        "master_of_law_advice":advice["final_answer"]
    }


def master_of_coin_node(state:KingState):
    advice=coin_response(state["situation"])
    print("------->")
    print("King:Master of Coin, What do you think of the situation")
    print("------->")
    print(f'Master of Coin:{advice["final_answer"]}')
    print("------->")
    return {
        "master_of_coin_advice":advice["final_answer"]
    }
def king_node(state:KingState):
    decision=king_chain.invoke({
        "situation":state["situation"],
        "hand_advice":state["lord_hand_advice"],
        "commander_advice":state["lord_commander_advice"],
        "law_advice":state["master_of_law_advice"],
        "coin_advice":state["master_of_coin_advice"],
        "whisper_advice":state["lord_of_whispers_advice"]


    })

    return {
        "final_decision":decision.content
    }




graph=StateGraph(KingState)
graph.add_node("commander",lord_commander_node)
graph.add_node("hand",lord_hand_node)
graph.add_node("whisper",lord_of_whispers_node)
graph.add_node("coin",master_of_coin_node)
graph.add_node("law",master_of_law_node)
graph.add_node("king",king_node)

graph.add_edge(START,"coin")
graph.add_edge("coin","law")
graph.add_edge("law","whisper")
graph.add_edge("whisper","commander")
graph.add_edge("commander","hand")
graph.add_edge("hand","king")
graph.add_edge("king",END)

app=graph.compile()
situation=input("give a situation:")
response=app.invoke({
        "situation":situation
    },
    config={"recursion_limit": 400})

print(f"King:After hearing the council, the Crown's final decesion is :")
print(f"------> {response['final_decision']}")