import os
import asyncio
from pathlib import Path
from typing import List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.session import RequestResponder
from langchain_core.tools import StructuredTool

from src.infrastructure.config import load_config

class MCPManager:
    def __init__(self):
        self.config = load_config()
        # Handle Docker and local paths
        if os.path.exists("/app/workspace/mcps"):
            self.workspace_path = Path("/app/workspace/mcps").resolve()
        else:
            self.workspace_path = Path(os.path.dirname(__file__)).parent.parent / "workspace" / "mcps"
            self.workspace_path = self.workspace_path.resolve()
        self._sessions: List[ClientSession] = []
        self._exit_stack = AsyncExitStack()
        self._background_tasks: List[asyncio.Task] = []

    async def start(self):
        """
        Starts the MCP client sessions.
        """
        # 1. Connect to Obsidian MCP
        if self.config.obsidian.api_key:
            repo_path = self.workspace_path / "mcp-obsidian"
            script_path = repo_path / "dist/server.js"
            
            obsidian_params = StdioServerParameters(
                command="node",
                args=[str(script_path), self.config.obsidian.vault_path],
                env={
                    **os.environ,
                    "OBSIDIAN_API_KEY": self.config.obsidian.api_key,
                    "OBSIDIAN_URL": self.config.obsidian.url
                }
            )
            await self._connect_server(obsidian_params, "Obsidian")
        else:
            print("Warning: Obsidian API key not configured. Obsidian tools will not be available.")

        # 2. Connect to Filesystem MCP
        if self.config.filesystem and self.config.filesystem.allowed_paths:
            filesystem_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", *self.config.filesystem.allowed_paths],
                env=os.environ.copy()
            )
            await self._connect_server(filesystem_params, "Filesystem")
        else:
            print("Info: Filesystem MCP not configured or no allowed paths.")

    async def _connect_server(self, params: StdioServerParameters, name: str):
        try:
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
            session = await self._exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            
            # Start background message handler for this session
            task = asyncio.create_task(self._handle_session_messages(session, name))
            self._background_tasks.append(task)
            
            await session.initialize()
            # Store session with its name
            self._sessions.append((session, name))
            print(f"Connected to {name} MCP server.")
        except Exception as e:
            print(f"Failed to connect to {name} MCP server: {e}")

    async def _handle_session_messages(self, session: ClientSession, name: str):
        """
        Handles incoming messages (requests/notifications) from the MCP server.
        Specifically handles roots/list for the Filesystem server.
        """
        try:
            async for message in session.incoming_messages:
                if isinstance(message, RequestResponder):
                    if message.request.root.method == "roots/list":
                        # Provide roots from filesystem config
                        roots = []
                        if self.config.filesystem:
                            for path in self.config.filesystem.allowed_paths:
                                roots.append(types.Root(uri=f"file://{path}", name=os.path.basename(path)))
                        
                        await message.respond(types.ListRootsResult(roots=roots))
                    else:
                        # For other requests, we don't have a default handler yet
                        pass
                elif isinstance(message, Exception):
                    print(f"MCP Session '{name}' error: {message}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in MCP Session '{name}' message handler: {e}")

    async def stop(self):
        """
        Stops all MCP client sessions and background tasks.
        """
        for task in self._background_tasks:
            task.cancel()
        
        await self._exit_stack.aclose()
        self._sessions = []
        self._background_tasks = []
        print("Disconnected from all MCP servers.")

    async def get_tools(self) -> List[StructuredTool]:
        """
        Returns an aggregated list of LangChain tools wrapping all MCP tools.
        Tool names are prefixed with the server name to avoid collisions.
        """
        all_langchain_tools = []

        for session, server_name in self._sessions:
            prefix = server_name.lower().replace(" ", "_")
            try:
                # Add a timeout to avoid hanging the whole application
                mcp_tools = await asyncio.wait_for(session.list_tools(), timeout=10.0)
                
                for tool in mcp_tools.tools:
                    # Capture session and tool_name for the closure
                    current_session = session
                    current_tool_name = tool.name
                    
                    async def tool_func(**kwargs):
                        result = await current_session.call_tool(current_tool_name, arguments=kwargs)
                        # Extract content from result
                        text_content = []
                        if hasattr(result, 'content'):
                            for content in result.content:
                                if hasattr(content, 'text'):
                                     text_content.append(content.text)
                        
                        return "\n".join(text_content)

                    # Prefix tool name if it doesn't already have it
                    namespaced_name = f"{prefix}_{tool.name}"
                    
                    langchain_tool = StructuredTool.from_function(
                        func=None,
                        coroutine=tool_func,
                        name=namespaced_name,
                        description=f"[{server_name}] {tool.description}",
                    )
                    all_langchain_tools.append(langchain_tool)

            except asyncio.TimeoutError:
                print(f"Timeout fetching tools from session {server_name}.")
            except Exception as e:
                print(f"Error fetching tools from session {server_name}: {e}")

        return all_langchain_tools
