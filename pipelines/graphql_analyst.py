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
from langchain_community.utilities import GraphQLAPIWrapper
from pipelines.text_to_graphql.base import create_graphql_agent
from pipelines.text_to_graphql.toolkit import GraphQLDatabaseToolkit

class Pipeline:
    class Valves(BaseModel):
        GRAPHQL_ENDPOINT: str = "https://api.example.com/graphql"
        OPENAI_API_KEY: str = ""
        MODEL: str = "gpt-4-turbo"

    def __init__(self):
        self.name = "Inverse GraphQL Agent"
        self.wrapper = None
        self.valves = self.Valves(
            **{
                "GRAPHQL_ENDPOINT": os.getenv("GRAPHQL_ENDPOINT", "https://api.example.com/graphql"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
                "MODEL": os.getenv("MODEL", "gpt-4-turbo")
            }
        )
        self.init_graphql_connection()

    def init_graphql_connection(self):
        """Initialize the GraphQL wrapper"""
        self.wrapper = GraphQLAPIWrapper(
            graphql_endpoint=self.valves.GRAPHQL_ENDPOINT
        )
        return self.wrapper

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        self.init_graphql_connection()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        self.wrapper = None

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        # Check for special message types
        if body.get("title", False):
            print("Title Generation")
            return "INV GraphQL Explorer"

        # Create GraphQL agent for queries
        llm = ChatOpenAI(
            temperature=0.1,
            model=self.valves.MODEL,
            openai_api_key=self.valves.OPENAI_API_KEY,
            streaming=True
        )
        
        toolkit = GraphQLDatabaseToolkit(
            graphql_wrapper=self.wrapper,
            llm=llm
        )
        
        agent_executor = create_graphql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            agent_type="openai-functions",
            max_iterations=10,
            handle_parsing_errors=True
        )

        try:
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
        
        # Track if we're inside a code block or JSON block
        in_code_block = False
        json_content = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and system messages
            if not line or line.lower().startswith(('responded:', 'invoking:', '> entering', '> finished')):
                continue
                
            # Handle JSON content
            if line.startswith('{') and line.endswith('}'):
                json_content.append(line)
                continue
                
            # Handle code blocks
            if line.startswith('```'):
                in_code_block = not in_code_block
                continue
                
            if not in_code_block:
                processed_lines.append(line)
        
        # Combine processed lines
        result = '\n'.join(processed_lines)
        
        # Add JSON content at the end if present
        if json_content:
            result += '\n\n' + '\n'.join(json_content)
            
        return result.strip()