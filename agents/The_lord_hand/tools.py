from langchain_core.tools import tools
from functions import read_file
from pathlib import Path
@tool
def common_summary_tool(query:str):
    "this tool will access the every common detail about the kingdom"
    path=Path.cwd()/"lore2"/"common"/"common_summary"
    content=read_file(base_path)
    return content
    


