import asyncio
import os
import shutil
from pathlib import Path
from src.infrastructure.tools.local_tool_manager import LocalToolManager

async def test_manual_verification():
    print("--- Starting Local Tool Manager verification ---")
    
    # 1. Setup
    manager = LocalToolManager()
    
    # 2. Create a tool file manually
    tool_code = """
from langchain_core.tools import tool

@tool
def test_calc_add(a: int, b: int) -> int:
    "Adds two numbers."
    return a + b
"""
    print("Creating test tool file...")
    manager.create_tool_file("test_math.py", tool_code)
    
    # 3. Load tools
    print("Loading tools...")
    tools = manager.load_tools()
    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    
    # 4. Verify specific tool exists
    math_tool = next((t for t in tools if t.name == "test_calc_add"), None)
    if math_tool:
        print(f"Found tool: {math_tool.name}")
        # 5. Execute tool
        print("Executing tool...")
        try:
            result = math_tool.run({"a": 5, "b": 3})
            print(f"Result (5+3): {result}")
            if result == 8:
                print("SUCCESS: Tool execution verified.")
            else:
                print(f"FAILURE: Unexpected result {result}")
        except Exception as e:
            print(f"FAILURE: Tool execution failed: {e}")
    else:
        print("FAILURE: test_calc_add tool not found.")

    # 6. Test Management Tools
    print("\n--- Testing Management Tools ---")
    mgmt_tools = manager.get_management_tools()
    print(f"Management tools: {[t.name for t in mgmt_tools]}")
    
    create_tool = next((t for t in mgmt_tools if t.name == "create_python_tool"), None)
    refresh_tool = next((t for t in mgmt_tools if t.name == "refresh_local_tools"), None)
    
    if create_tool and refresh_tool:
        # Create a new tool via the tool
        new_tool_code = """
from langchain_core.tools import tool

@tool
def test_say_hello(name: str) -> str:
    "Says hello."
    return f"Hello {name}!"
"""
        print("Using create_python_tool...")
        create_tool.run({"filename": "test_hello", "code": new_tool_code})
        
        print("Using refresh_local_tools...")
        refresh_result = refresh_tool.run({})
        print(f"Refresh result: {refresh_result}")
        
        # Verify new tool is loaded in the manager
        # Note: manager.tools is updated by load_tools which is called by refresh_tool
        current_tools = manager.tools
        hello_tool = next((t for t in current_tools if t.name == "test_say_hello"), None)
        
        if hello_tool:
             print("SUCCESS: Dynamic creation and reload verified.")
             print(f"Run hello: {hello_tool.run({'name': 'World'})}")
        else:
             print("FAILURE: New tool not found after refresh.")
             
    else:
        print("FAILURE: Management tools missing.")

    # Cleanup
    # (Optional) remove created files
    # os.remove(workspace_path / "test_math.py")
    # os.remove(workspace_path / "test_hello.py")

if __name__ == "__main__":
    asyncio.run(test_manual_verification())
