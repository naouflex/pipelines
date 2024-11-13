"""
title: Weekly Summary Generator Pipeline
author: open-webui
version: 1.0
license: MIT
description: A pipeline for generating weekly summaries using LangChain and Playwright
requirements: langchain-openai,playwright,python-dotenv,langchain-community,lxml,bs4
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
from functools import partial
from langchain.agents import AgentExecutor
import logging

class Pipeline:
    """Weekly summary generation pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        MODEL: str = os.getenv("MODEL", "gpt-4-turbo")
        HEADLESS: bool = os.getenv("HEADLESS", "true").lower() == "true"

    def __init__(self):
        print("Initializing Pipeline...")
        try:
            self.type = "manifold"
            self.name = "Inverse Weekly: Weekly Summary Generator"
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
        """Get the available GPT models

        Returns:
            List[dict]: The list of GPT models
        """
        return [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
        ]

    async def init_browser(self, force_new=False):
        """Initialize the browser and toolkit"""
        print("Initializing browser...")
        try:
            if self.browser:
                if not force_new:
                    print("Using existing browser instance")
                    return True
                print("Closing existing browser instance")
                await self.browser.close()
                self.browser = None
                self.toolkit = None
            
            self.browser = await create_async_playwright_browser(headless=self.valves.HEADLESS)
            print("Browser created successfully")
            self.toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
            print("Toolkit initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing browser: {str(e)}")
            print(traceback.format_exc())
            return False

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

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

    async def _execute_agent(self, agent_executor, prompt_request):
        """Execute agent with proper async handling"""
        print("Executing agent...")
        try:
            print(f"Prompt request: {prompt_request[:100]}...")  # Print first 100 chars of prompt
            response = await agent_executor.ainvoke({"input": prompt_request})
            print("Agent execution completed successfully")
            return response["output"]
        except Exception as e:
            print(f"Error executing agent: {str(e)}")
            print(traceback.format_exc())
            return f"Error executing agent: {str(e)}\n{traceback.format_exc()}"

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"\n{'='*50}\nStarting pipe execution")
        print(f"User message: {user_message}")
        print(f"Model ID: {model_id}")
        print(f"Body: {body}")

        async def generate_response():
            try:
                print("Initializing new browser instance for this request...")
                # Always create a new browser instance for each request
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
                    max_iterations=20,
                )
                
                prompt_request = f"""
                Instructions: Produce a weekly article following those instructions\
                    1. Go to https://app.inverse.watch/public/dashboards/iVI1ZdCMMTOg2SwSWOpIKQGOfojXGQd8QDeisa25?org_slug=default&p_week={user_message} and wait 10 seconds for the page to load \
                    2. Use extract_text tool to extract the data from the page.\
                    5. Get examples of articles at https://inverse.watch/weekly-2024-w44 and https://inverse.watch/weekly-2024-w45
                    6. Use extract_text tool to extract the data from the page and identify the sections and formats in those articles
                    7. Produce a summary for week {user_message} with the data you extracted in step 2 and 4.\
                    8. Make sure to adhere to the sections and format of the articles.
                    9. Return your final answer as a message nicely formatted in Markdown.\
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

    async def _stream_response(self, agent_executor, prompt_request):
        """Stream response from agent with proper async handling"""
        try:
            # Execute agent and get response
            response = await agent_executor.ainvoke({"input": prompt_request})
            
            # Process the response
            if isinstance(response, dict) and "output" in response:
                final_response = self._process_response(response["output"])
                
                # Stream the processed response line by line
                if final_response:
                    for line in final_response.split('\n'):
                        if line.strip():  # Only yield non-empty lines
                            yield line + "\n"
                else:
                    yield "No valid response generated."
            else:
                yield "Unexpected response format."
            
        except Exception as e:
            print(f"Error streaming response: {e}")
            print(traceback.format_exc())
            yield f"Error streaming response: {e}"

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
