"""
title: Text to GraphQL Pipeline
author: open-webui
version: 1.0
license: MIT
description: A pipeline for converting natural language to GraphQL queries using LangChain
requirements: langchain-openai,gql
"""

import os
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from pipelines.text_to_graphql.wrapper import GraphQLAPIWrapper
from pipelines.text_to_graphql.base import create_graphql_agent
from pipelines.text_to_graphql.toolkit import GraphQLDatabaseToolkit

class Pipeline:
    """GraphQL query pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""
        GRAPHQL_ENDPOINT: str = os.getenv("GRAPHQL_ENDPOINT", "https://api.example.com/graphql")
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
        MODEL: str = os.getenv("MODEL", "gpt-4-turbo")

    def __init__(self):
        self.name = "Inverse GraphQL Agent"
        self.wrapper = None
        self.valves = self.Valves()
        

    async def init_graphql_connection(self):
        """Initialize the GraphQL wrapper"""
        try:
            self.wrapper = GraphQLAPIWrapper(
                graphql_endpoint=self.valves.GRAPHQL_ENDPOINT
            )
            return True
        except Exception as e:
            print(f"Error initializing GraphQL connection: {e}")
            return False

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        await self.init_graphql_connection()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.wrapper:
            self.wrapper = None

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        await self.init_graphql_connection()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        # Check for special message types that don't need GraphQL
        if body.get("title", False):
            print("Title Generation")
            return "GraphQL Query Generator"

        try:
            # Create GraphQL agent for actual queries
            llm = ChatOpenAI(
                temperature=0.1,
                model=self.valves.MODEL,
                openai_api_key=self.valves.OPENAI_API_KEY,
                streaming=True
            )
            
            toolkit = GraphQLDatabaseToolkit(graphql_wrapper=self.wrapper, llm=llm)
            agent_executor = create_graphql_agent(
                llm=llm,
                toolkit=toolkit,
                verbose=True,
                agent_type="openai-functions",
                max_iterations=10,
                handle_parsing_errors=True
            )

            if body.get("stream", False):
                # Collect the complete response first
                response_chunks = []
                for chunk in agent_executor.stream({"input": user_message}):
                    if isinstance(chunk, dict):
                        # Collect all relevant parts of the response
                        if "intermediate_steps" in chunk:
                            for step in chunk["intermediate_steps"]:
                                if isinstance(step, tuple) and len(step) == 2:
                                    action, output = step
                                    if output and str(output) != "None":
                                        response_chunks.append(str(output))
                        if "output" in chunk and chunk["output"]:
                            response_chunks.append(str(chunk["output"]))
                        if "content" in chunk and chunk["content"]:
                            response_chunks.append(str(chunk["content"]))

                # Process the complete response
                complete_response = "\n".join(response_chunks)
                final_response = self._process_response(complete_response)

                # Stream the final response
                if final_response:
                    for line in final_response.split('\n'):
                        yield line + "\n"
                else:
                    yield "No valid response generated. Please try rephrasing your question."
            else:
                response = agent_executor.invoke(
                    {"input": user_message},
                    {"return_only_outputs": True}
                )
                return self._process_response(str(response.get("output", "")))
        except Exception as e:
            return f"Error executing query: {str(e)}"

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