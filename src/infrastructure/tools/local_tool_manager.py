import importlib.util
import os
import sys
import inspect
from pathlib import Path
from typing import List, Optional, Any
from langchain_core.tools import BaseTool, StructuredTool, tool

class LocalToolManager:
    def __init__(self, tools_dirs: Optional[List[str]] = None):
        if tools_dirs:
            self.tools_dirs = [Path(d).resolve() for d in tools_dirs]
        else:
            # Default directories: assets and workspace
            self.tools_dirs = []
            
            # 1. Assets (official/git-tracked)
            assets_path = "/app/assets/langchain/tools" if os.path.exists("/app/assets/langchain/tools") else None
            if not assets_path:
                # New path: apps/elo-server/assets/langchain/tools
                assets_path = Path(os.path.dirname(__file__)).resolve().parents[2] / "assets" / "langchain" / "tools"
            
            # 2. Workspace (local/user-defined)
            workspace_path = "/app/workspace/langchain/tools" if os.path.exists("/app/workspace/langchain/tools") else None
            if not workspace_path:
                root_path = Path(os.path.dirname(__file__)).resolve().parents[4]
                workspace_path = root_path / "workspace" / "langchain" / "tools"

            self.tools_dirs = [assets_path, workspace_path]

        # Ensure directories exist (at least the workspace one)
        for d in self.tools_dirs:
            if not d.exists():
                print(f"Creating tools directory: {d}")
                d.mkdir(parents=True, exist_ok=True)
            
            # Add tools dir to sys.path so imports work easily
            if str(d) not in sys.path:
                sys.path.append(str(d))
            
        # The primary directory for creating new tools is the workspace one
        self.primary_tools_dir = self.tools_dirs[-1]

    def load_tools(self) -> List[BaseTool]:
        """
        Scans all tools directories and loads all valid LangChain tools.
        """
        self.tools = []
        for d in self.tools_dirs:
            print(f"Scanning for tools in {d}...")
            if not d.exists():
                continue
                
            for filename in os.listdir(d):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]
                    file_path = d / filename
                    
                    try:
                        print(f"Loading tool module: {module_name} from {file_path}")
                        # Dynamic import
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            sys.modules[module_name] = module
                            spec.loader.exec_module(module)
                            
                            # Inspect module for tools
                            found_tools_in_module = 0
                            for name, obj in inspect.getmembers(module):
                                if isinstance(obj, BaseTool):
                                    self.tools.append(obj)
                                    found_tools_in_module += 1
                                    print(f"  Found tool: {obj.name}")
                                
                            if found_tools_in_module == 0:
                                 print(f"  No tools found in {module_name}")
                                    
                    except Exception as e:
                        print(f"Error loading tool module {module_name}: {e}")
                        import traceback
                        traceback.print_exc()

        return self.tools

    def create_tool_file(self, filename: str, code: str) -> str:
        """
        Creates a new Python file in the tools directory.
        """
        if not filename.endswith(".py"):
            filename += ".py"
            
        file_path = self.primary_tools_dir / filename
        
        # specific validation could go here
        
        with open(file_path, "w") as f:
            f.write(code)
            
        return f"Tool file created at {file_path}. You should refresh tools to load it."

    def get_management_tools(self) -> List[BaseTool]:
        """
        Returns tools that allow the agent to manage (create/refresh) tools.
        """
        
        @tool
        def create_python_tool(filename: str, code: str) -> str:
            """
            Creates a new LangChain tool in Python. 
            The code MUST import 'tool' from 'langchain_core.tools' or 'BaseTool'.
            The code MUST define a function decorated with @tool or a class inheriting from BaseTool.
            Example:
            ```python
            from langchain_core.tools import tool

            @tool
            def my_tool(arg1: str) -> str:
                \"\"\"Description of what the tool does.\"\"\"
                return "result"
            ```
            """
            return self.create_tool_file(filename, code)

        @tool
        def refresh_local_tools() -> str:
            """
            Reloads all local tools from the tools directory. Call this after creating a new tool.
            """
            loaded = self.load_tools()
            tool_names = [t.name for t in loaded]
            return f"Tools refreshed. Loaded {len(loaded)} tools: {', '.join(tool_names)}"

        return [create_python_tool, refresh_local_tools]
