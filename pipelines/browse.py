"""
title: Web Browsing Pipeline
version: 1.0
license: MIT
description: A pipeline for web browsing and information extraction using LangChain and Playwright
requirements: langchain-openai,playwright,python-dotenv,langchain-community
"""

import os
import traceback
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.hub import pull
from pipelines.web.base import create_structured_chat_agent
from pipelines.web.toolkit import PlayWrightBrowserToolkit
from pipelines.web.utils import create_async_playwright_browser
import asyncio
from langchain.agents import AgentExecutor
import logging

class Pipeline:
    """Web browsing pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        MODEL: str = os.getenv("MODEL", "gpt-4-turbo")
        HEADLESS: bool = os.getenv("HEADLESS", "true").lower() == "true"

    def __init__(self):
        print("Initializing Pipeline...")
        try:
            self.type = "manifold"
            self.name = "Web Browser Agent"
            self.browser = None
            self.toolkit = None
            self.valves = self.Valves()
            print(f"Initializing LLM with model: {self.valves.MODEL}")
            self.llm = ChatOpenAI(
                model=self.valves.MODEL,
                temperature=0,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True
            )
            self.pipelines = self.get_openai_assistants()
            print("Pipeline initialization complete")
        except Exception as e:
            print(f"Error in Pipeline initialization: {str(e)}")
            print(traceback.format_exc())
            raise

    def get_openai_assistants(self) -> List[dict]:
        """Get the available GPT models"""
        return [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        ]

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.browser:
            await self.browser.close()

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        self.llm = ChatOpenAI(
            model=self.valves.MODEL,
            temperature=0,
            api_key=self.valves.OPENAI_API_KEY,
            streaming=True
        )
        self.pipelines = self.get_openai_assistants()

    def _process_response(self, response: str) -> str:
        """Process and clean the response."""
        if not response:
            return ""

        # Split into lines and process
        lines = response.split('\n')
        processed_lines = []
        
        # Track if we're inside a code block
        in_code_block = False
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and system messages
            if not line or line.lower().startswith(('responded:', 'invoking:', '> entering', '> finished')):
                continue
            
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
            
            if not in_code_block:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines).strip()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"\nStarting browse pipe execution")
        print(f"User request: {user_message}")

        async def generate_response():
            try:
                print("Initializing new browser instance for this request...")
                # Initialize new browser instance
                self.browser = await create_async_playwright_browser(headless=self.valves.HEADLESS)
                self.toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
                
                print("Setting up prompt and tools...")
                prompt = pull("naouufel/structured-chat-agent")
                tools = self.toolkit.get_tools()
                
                print("Creating agent...")
                agent = create_structured_chat_agent(self.llm, tools, prompt)
                
                print("Setting up agent executor...")
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_execution_time=180,
                    max_iterations=50,
                )
                
                prompt_request = f"""
                Instructions:
                    0. If the request is not clear, return and ask the user for clarification, otherwise go to step 1.
                    1. Navigate to the google webpage corresponding to the country of the request language.
                    2. Devise an action plan to fulfill the request using a search engine, navigate to the relevant pages and extract the relevant information.
                    3. Execute the action plan until the request is fulfilled.
                    4. If another action plan is necessary to provide a more detailed answer, go back to step 1.
                    5. Always include the complete, accurate, and relevant link(s) to the page(s) you found in your answer.
                    6. Return your final answer as a message nicely formatted in Markdown.
                    
                Request: {user_message}
                """
                
                if body.get("stream", False):
                    final_response = ""
                    async for chunk in agent_executor.astream(
                        {
                            "input": prompt_request,
                            "chat_history": [],
                        }
                    ):
                        if isinstance(chunk, dict) and "output" in chunk:
                            final_response = str(chunk["output"])
                    return self._process_response(final_response)
                else:
                    response = await agent_executor.ainvoke(
                        {
                            "input": prompt_request,
                            "chat_history": [],
                        }
                    )
                    return self._process_response(str(response.get("output", "")))

            except Exception as e:
                print(f"Error in generate_response: {str(e)}")
                print(traceback.format_exc())
                return f"Error: {str(e)}"
            finally:
                if self.browser:
                    await self.browser.close()

        try:
            # Temporarily disable uvloop
            old_policy = asyncio.get_event_loop_policy()
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(generate_response())
                return result
            finally:
                loop.close()
                # Restore the original event loop policy
                asyncio.set_event_loop_policy(old_policy)

        except Exception as e:
            error_msg = f"Error in pipe execution: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return error_msg