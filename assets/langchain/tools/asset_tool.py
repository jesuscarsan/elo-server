from langchain_core.tools import tool

@tool
def asset_hello_world() -> str:
    """A tool that exists in the assets directory."""
    return "Hello from the assets directory!"
