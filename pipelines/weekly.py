
"""
title: Weekly Summary Generator Pipeline
author: open-webui
version: 1.0
license: MIT
description: A pipeline for generating weekly summaries using LangChain and Playwright
requirements: langchain-openai,playwright,python-dotenv,langchain-community
"""

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.hub import pull
from pipelines.web.base import create_structured_chat_agent
from pipelines.web.toolkit import PlayWrightBrowserToolkit
from pipelines.web.utils import create_async_playwright_browser
import asyncio
from functools import partial

class Pipeline:
    """Weekly summary generation pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        MODEL: str = os.getenv("MODEL", "gpt-4-turbo")
        HEADLESS: bool = os.getenv("HEADLESS", "true").lower() == "true"

    def __init__(self):
        self.type = "manifold"
        self.name = "Weekly Summary Generator"
        self.browser = None
        self.toolkit = None
        self.valves = self.Valves()
        self.pipelines = self.get_models()

    def get_models(self):
        """Get available models for the pipeline"""
        return [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ]

    async def init_browser(self):
        """Initialize the browser and toolkit"""
        try:
            if self.browser:
                await self.browser.close()
            
            self.browser = await create_async_playwright_browser(headless=self.valves.HEADLESS)
            self.toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)
            return True
        except Exception as e:
            print(f"Error initializing browser: {e}")
            return False

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        await self.init_browser()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.toolkit = None

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        await self.init_browser()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"Messages: {messages}")
        print(f"User message: {user_message}")

        try:
            # Handle title generation request
            if body.get("title", False):
                return "Weekly Summary Generator"

            # Create a new event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Initialize browser if not already initialized
            if not self.browser or not self.toolkit:
                loop.run_until_complete(self.init_browser())

            llm = ChatOpenAI(
                model=self.valves.MODEL,
                temperature=0,
                api_key=self.valves.OPENAI_API_KEY,
                streaming=True
            )

            prompt = pull("naouufel/structured-chat-agent")
            tools = self.toolkit.get_tools()
            agent = create_structured_chat_agent(llm, tools, prompt)

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
                5. Get examples of articles at https://inverse.watch/weekly-2024-w01 and https://inverse.watch/weekly-2024-w45
                6. Use extract_text tool to extract the data from the page and identify the sections and formats in those articles
                7. Produce a summary for week {user_message} with the data you extracted in step 2 and 4.\
                8. Make sure to adhere to the sections and format of the articles.
                9. Return your final answer as a string message formatted for discord.\
            """

            if body.get("stream", False):
                return self._stream_response(agent_executor, prompt_request)
            else:
                response = agent_executor.invoke({"input": prompt_request})
                return response["output"]

        except Exception as e:
            print(f"Error generating weekly summary: {e}")
            return f"Error generating weekly summary: {e}"
        finally:
            # Clean up the event loop
            loop.close()

    def _stream_response(self, agent_executor, prompt_request):
        for chunk in agent_executor.stream({"input": prompt_request}):
            if isinstance(chunk, dict):
                if "output" in chunk:
                    yield chunk["output"]
                elif "intermediate_steps" in chunk:
                    for step in chunk["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            _, output = step
                            if output and str(output) != "None":
                                yield str(output)