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
        self.pipelines = self.get_models()

    def get_models(self):
        """Get available models for the pipeline"""
        return [
            {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
            {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ]

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