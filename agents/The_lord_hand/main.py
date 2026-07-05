from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from tools import chain,retrieve_info
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage
from prompts import prompt


class LordHandState(TypedDict):
    situation: str
    opinions: list
    messages: Annotated[list, add_messages]

def lord_hand_node(state:LordHandState):
    response=chain.invoke(
        {   "situation":state["situation"],
            "opinions":state["opinions"],
            "messages":state["messages"]})
    return {
            "messages":[response]
        }


tool_node=ToolNode([retrieve_info])

graph=StateGraph(LordHandState)
graph.add_node("lord_hand",lord_hand_node)
graph.add_node("tools",tool_node)

graph.add_edge(START,"lord_hand")
graph.add_conditional_edges(
    "lord_hand",
    tools_condition,
    {"tools":"tools",
    END:END}
)
graph.add_edge("tools","lord_hand")
app=graph.compile()
from langchain_core.messages import AIMessage

# opinions = [
#     AIMessage(
#         name="Lord Commander",
#         content=(
#             "We should send the army to collect the taxes. "
#             "This will demonstrate the consequences of refusing to pay tribute."
#         )
#     ),
#     AIMessage(
#         name="Master of Coin",
#         content=(
#             "My King, we cannot afford a war with Ashvale. "
#             "The treasury is already under considerable strain."
#         )
#     ),
# ]
opinions=(
    """
    lord_commander:We should send the army to collect the taxes. This will demonstrate the consequences of refusing to pay tribute.
    master of coin:My King, we cannot afford a war with Ashvale. The treasury is already under considerable strain.
    """
)
result = app.invoke(
    {
        "situation": """
Ashvale has refused to pay tax from the last 2 years 
""",
        "opinions": opinions,
        "messages": []
    
})
final_message=result["messages"][-1].content
print(final_message)
print("_________________")
print(result)

