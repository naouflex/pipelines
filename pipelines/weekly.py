"""A pipeline for generating weekly summaries using LangChain and Playwright"""

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor
from pipelines.web.base import create_structured_chat_agent
from pipelines.web.toolkit import PlayWrightBrowserToolkit
from pipelines.web.utils import create_async_playwright_browser

class Pipeline:
    """Weekly summary generation pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
        MODEL: str = "gpt-4-turbo"
        HEADLESS: bool = True

    def __init__(self):
        self.type = "manifold"
        self.name = "Weekly Summary Generator"
        self.valves = self.Valves()
        self.browser = None
        self.toolkit = None
        self.setup_browser()

    def setup_browser(self):
        """Initialize the browser and toolkit"""
        self.browser = create_async_playwright_browser(headless=self.valves.HEADLESS)
        self.toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=self.browser)

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        self.setup_browser()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.browser:
            await self.browser.close()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        try:
            llm = ChatOpenAI(
                model=self.valves.MODEL,
                temperature=0,
                api_key=self.valves.OPENAI_API_KEY
            )

            prompt = hub.pull("naouufel/structured-chat-agent")
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

            response = agent_executor.invoke({"input": prompt_request})
            yield response["output"]

        except Exception as e:
            print(f"Error generating weekly summary: {e}")
            yield f"Error generating weekly summary: {e}"