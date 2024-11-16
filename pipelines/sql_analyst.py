"""
title: Text to SQL Pipeline
author: open-webui
version: 1.0
license: MIT
description: A pipeline for converting natural language to SQL queries using LangChain
requirements: langchain-openai,sqlalchemy,psycopg2-binary
"""

import os
from typing import List, Union, Generator, Iterator, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from pipelines.sql_analyst_tools.base import create_sql_agent
from pipelines.sql_analyst_tools.toolkit import SQLDatabaseToolkit
import time


class Pipeline:
    class Valves(BaseModel):
        DB_CONNECTION_STRING: str = "postgresql://user:pass@localhost:5432/db"
        OPENAI_API_KEY: str = ""
        MODEL: str = "gpt-4-turbo"
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

        # SQL Prompts
        PROMPT_SQL_PREFIX: str = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer."""

        PROMPT_SQL_SUFFIX: str = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""

        PROMPT_SQL_FUNCTIONS_SUFFIX: str = """I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables."""

        PROMPT_QUERY_CHECKER: str = """{query}
Double check the {dialect} query above for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

Output the final SQL query only.

SQL Query: """
    MAX_ITERATIONS: int = 10

    def __init__(self):
        self.name = "Inverse Stats DB Agent"
        self.engine = None
        self.nlsql_response = ""
        self.valves = self.Valves(
            **{
                "DB_CONNECTION_STRING": os.getenv("DB_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/db"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
                "MODEL": os.getenv("MODEL", "gpt-4-turbo"),
            }
        )
        self.init_db_connection()
        self.last_emit_time = 0

    def init_db_connection(self):
        self.engine = create_engine(self.valves.DB_CONNECTION_STRING)
        return self.engine

    async def on_startup(self):
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")
        self.init_db_connection()

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")
        if self.engine:
            self.engine.dispose()

    async def emit_status(
        self,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        level: str,
        message: str,
        done: bool,
    ):
        current_time = time.time()
        if (
            __event_emitter__
            and self.valves.enable_status_indicator
            and (
                current_time - self.last_emit_time >= self.valves.emit_interval or done
            )
        ):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "status": "complete" if done else "in_progress",
                        "level": level,
                        "description": message,
                        "done": done,
                    },
                }
            )
            self.last_emit_time = current_time

    async def pipe(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
    ) -> Optional[dict]:
        await self.emit_status(__event_emitter__, "info", "Initializing SQL Agent", False)
        
        user_message = body.get("messages", [])[-1]["content"]
        model_id = body.get("model", self.valves.MODEL)
        messages = body.get("messages", [])

        # Add debug prints like in OpenAI pipeline
        print(f"pipe:{__name__}")
        print(messages)
        print(user_message)

        # Check for special message types that don't need SQL
        if body.get("title", False):
            print("Title Generation")
            return "INV Price Checker"

        # Only create SQL agent for actual database queries
        llm = ChatOpenAI(
            temperature=0.1,
            model=self.valves.MODEL,
            openai_api_key=self.valves.OPENAI_API_KEY,
            streaming=True
        )

        db = SQLDatabase(self.engine)

        toolkit = SQLDatabaseToolkit(
            db=db, 
            llm=llm,
            valves=self.valves  # Add valves parameter here
        )
        agent_executor = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            valves=self.valves,
            verbose=True,
            agent_type="openai-functions",
            max_iterations=self.MAX_ITERATIONS,
            handle_parsing_errors=True
        )

        try:
            if body.get("stream", False):
                await self.emit_status(__event_emitter__, "info", "Processing streaming query", False)
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
                await self.emit_status(__event_emitter__, "info", "Processing single query", False)
                response = agent_executor.invoke(
                    {"input": user_message},
                    {"return_only_outputs": True}
                )
                processed_response = self._process_response(str(response.get("output", "")))
                
            await self.emit_status(__event_emitter__, "info", "Query complete", True)
            return processed_response
        except Exception as e:
            await self.emit_status(__event_emitter__, "error", f"Error executing query: {str(e)}", True)
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
