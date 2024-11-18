from typing import Optional, Callable, Any
import requests
from datetime import datetime
from langchain_community.utilities.sql_database import SQLDatabase
import os
from pydantic import BaseModel, Field
from fastapi import HTTPException
import logging
from sqlalchemy import create_engine
import re
import time


class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def emit(self, description="Unknown State", status="in_progress", done=False):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": status,
                        "description": description,
                        "done": done,
                    },
                }
            )


def get_send_status(
    __event_emitter__: Optional[Callable[[dict], Any]]
) -> Callable[[str, bool], None]:
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status


class Tools:
    class Valves(BaseModel):
        DB_CONNECTION_STRING: str = Field(
            default=os.getenv(
                "DB_CONNECTION_STRING", "postgresql://user:pass@localhost:5432/db"
            ),
            description="Database connection string",
        )
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
            description="OpenAI API Key",
        )
        OPENAI_API_BASE: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API Base URL",
        )
        MODEL: str = Field(
            default="gpt-4-turbo",
            description="OpenAI model to use",
        )
        MAX_ROWS: int = Field(
            default=10,
            description="Maximum number of rows to return in queries",
        )
        TIMEOUT: int = Field(
            default=30,
            description="Request timeout in seconds",
        )
        temperature: float = Field(
            default=0.1, description="Temperature for the LLM responses"
        )
        dialect: str = Field(default="postgresql", description="SQL dialect being used")
        emit_interval: float = Field(
            default=2.0, description="Interval in seconds between status emissions"
        )
        enable_status_indicator: bool = Field(
            default=True, description="Enable or disable status indicator emissions"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.engine = None
        self.db = None
        self.last_emit_time = 0

    async def init_db_connection(self):
        """
        Initialize Database
        Establishes connection to the database using the provided connection string.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.engine = create_engine(self.valves.DB_CONNECTION_STRING)
            self.db = SQLDatabase.from_uri(self.valves.DB_CONNECTION_STRING)
            return True
        except Exception as e:
            logging.error(f"Error initializing database connection: {e}")
            return False

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Any]],
        level: str,
        message: str,
        done: bool,
    ):
        """
        Unified status emission method
        """
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

    async def query_check(
        self, query: str, __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Check SQL Query
        Analyzes the given SQL query using OpenAI API for errors and potential improvements.

        Args:
            query (str): The SQL query to analyze
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            str: Analysis results or error message
        """
        try:
            await self.emit_status(
                __event_emitter__, "info", "Analyzing query with OpenAI", False
            )

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.valves.MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a SQL expert. Check the following query for errors and potential improvements.",
                    },
                    {"role": "user", "content": f"Check this SQL query:\n{query}"},
                ],
                "temperature": self.valves.temperature,
            }

            response = requests.post(
                f"{self.valves.OPENAI_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.valves.TIMEOUT,
            )
            response.raise_for_status()

            result = response.json()
            await self.emit_status(
                __event_emitter__, "success", "Analysis complete", True
            )
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            await self.emit_status(__event_emitter__, "error", f"Error: {str(e)}", True)
            return f"Error analyzing query: {str(e)}"

    async def query_database(
        self, query: str, __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Execute SQL Query
        Executes the given SQL query and returns the results.
        If there is an error, modify the query accordingly and try again.

        Args:
            query (str): The SQL query to execute
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            str: Query results or error message
        """
        try:
            if self.db is None:
                connection_success = await self.init_db_connection()
                if not connection_success:
                    raise HTTPException(
                        status_code=500,
                        detail="Could not establish database connection",
                    )

            await self.emit_status(__event_emitter__, "info", "Executing query", False)

            # First, check if we need to look up table information
            if not query.strip().lower().startswith('select'):
                # Get list of available tables
                tables = self.db.get_usable_table_names()
                table_info = self.db.get_table_info_no_throw(tables)
                
                # Use OpenAI to construct a proper query
                headers = {
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }

                prompt = f"""Given the following question and available tables, create a SQL query:
                Question: {query}
                Available Tables and Schema:
                {table_info}
                
                Return only the SQL query, no explanations."""

                payload = {
                    "model": self.valves.MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a SQL expert. Generate SQL queries based on natural language questions."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.valves.temperature,
                }

                response = requests.post(
                    f"{self.valves.OPENAI_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.valves.TIMEOUT,
                )
                response.raise_for_status()
                query = response.json()["choices"][0]["message"]["content"].strip()

            if "limit" not in query.lower():
                query = f"{query} LIMIT {self.valves.MAX_ROWS}"

            results = self.db.run_no_throw(query)
            if results is None:
                raise Exception("Query returned no results")

            formatted_results = (
                results.to_string(index=False)
                if hasattr(results, "to_string")
                else str(results)
            )

            await self.emit_status(
                __event_emitter__, "success", "Query executed successfully", True
            )
            return formatted_results

        except Exception as e:
            await self.emit_status(__event_emitter__, "error", f"Error: {str(e)}", True)
            return f"Error executing query: {str(e)}"

    async def get_table_info(
        self,
        table_names: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        """
        Get Table Information
        Retrieves schema and sample data for specified tables.

        Args:
            table_names (str): Comma-separated list of table names
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            str: Table information or error message
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            if self.db is None:
                connection_success = await self.init_db_connection()
                if not connection_success:
                    raise HTTPException(
                        status_code=500,
                        detail="Could not establish database connection",
                    )

            await emitter.emit("Retrieving table information")

            # Split and clean table names
            tables = [t.strip() for t in table_names.split(",")]

            # Get table info using SQLDatabase utility
            table_info = self.db.get_table_info_no_throw(tables)

            await emitter.emit(
                "Table information retrieved successfully", status="complete", done=True
            )
            return table_info

        except Exception as e:
            await emitter.emit(f"Error: {str(e)}", status="error", done=True)
            return f"Error retrieving table information: {str(e)}"

    async def list_tables(
        self, __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        List Tables
        Returns a list of all available tables in the database.

        Args:
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            str: Comma-separated list of table names or error message
        """
        emitter = EventEmitter(__event_emitter__)

        try:
            if self.db is None:
                connection_success = await self.init_db_connection()
                if not connection_success:
                    raise HTTPException(
                        status_code=500,
                        detail="Could not establish database connection",
                    )

            await emitter.emit("Retrieving table list")

            # Get list of tables using SQLDatabase utility
            tables = self.db.get_usable_table_names()
            table_list = ", ".join(tables)

            await emitter.emit(
                "Table list retrieved successfully", status="complete", done=True
            )
            return table_list

        except Exception as e:
            await emitter.emit(f"Error: {str(e)}", status="error", done=True)
            return f"Error retrieving table list: {str(e)}"

    async def validate_query(
        self, query: str, __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> bool:
        """
        Validate SQL Query
        Checks if the query is safe to execute (no DML/DDL statements).

        Args:
            query (str): The SQL query to validate
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            bool: True if query is safe, False otherwise
        """
        emitter = EventEmitter(__event_emitter__)

        # List of dangerous SQL keywords
        dangerous_keywords = [
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "TRUNCATE",
            "ALTER",
            "CREATE",
            "REPLACE",
            "MERGE",
            "UPSERT",
        ]

        query_upper = query.upper()

        try:
            await emitter.emit("Validating query")

            # Check for dangerous keywords
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    await emitter.emit(
                        f"Query contains dangerous keyword: {keyword}",
                        status="error",
                        done=True,
                    )
                    return False

            await emitter.emit(
                "Query validation successful", status="complete", done=True
            )
            return True

        except Exception as e:
            await emitter.emit(f"Error: {str(e)}", status="error", done=True)
            return False

    async def on_startup(self):
        """
        Initialize the database connection when the server starts.
        """
        print(f"on_startup:{__name__}")
        await self.init_db_connection()

    async def on_shutdown(self):
        """
        Clean up database connections when the server stops.
        """
        print(f"on_shutdown:{__name__}")
        if self.engine:
            self.engine.dispose()

    async def on_valves_updated(self):
        """
        Reinitialize the database connection when valves are updated.
        """
        print(f"on_valves_updated:{__name__}")
        await self.init_db_connection()

    async def plan_cte_query(
        self, 
        question: str, 
        tables: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Plan Complex CTE Query
        Uses OpenAI to break down a complex question into a structured CTE query plan.

        Args:
            question (str): The complex question to analyze
            tables (str): Available tables and their schema information
            __event_emitter__ (Optional[Callable]): Event emitter for status updates

        Returns:
            str: Structured CTE query plan or error message
        """
        try:
            await self.emit_status(
                __event_emitter__, 
                "info", 
                "Planning CTE query structure", 
                False
            )

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            prompt = f"""Given the following question and available tables, create a step-by-step CTE query plan:

Question: {question}

Available Tables and Schema:
{tables}

Break this down into:
1. The individual data components needed
2. How these components should be combined using CTEs
3. The final query structure using WITH clauses
4. Any potential optimizations or considerations

Format your response as a clear, structured plan that can be used to build the final query."""

            payload = {
                "model": self.valves.MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a SQL expert specializing in breaking down complex queries into manageable CTEs."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.valves.temperature,
            }

            response = requests.post(
                f"{self.valves.OPENAI_API_BASE}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.valves.TIMEOUT,
            )
            response.raise_for_status()

            result = response.json()
            await self.emit_status(
                __event_emitter__, 
                "success", 
                "CTE query plan generated", 
                True
            )
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            await self.emit_status(
                __event_emitter__, 
                "error", 
                f"Error: {str(e)}", 
                True
            )
            return f"Error planning CTE query: {str(e)}"

    async def is_market_data_query(
        self, 
        query: str, 
        __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> bool:
        """
        Evaluate if a query is related to market data.
        
        Args:
            query (str): The query or question to evaluate
            __event_emitter__ (Optional[Callable]): Event emitter for status updates
            
        Returns:
            bool: True if the query is market data related, False otherwise
        """
        # Common market data related terms
        market_data_keywords = {
            'price', 'prices', 'quote', 'quotes', 'trade', 'trades', 'volume',
            'bid', 'ask', 'open', 'close', 'high', 'low', 'ohlc', 'ohlcv',
            'market', 'ticker', 'symbol', 'exchange', 'stock', 'equity',
            'forex', 'fx', 'cryptocurrency', 'crypto', 'bond', 'future',
            'option', 'derivative', 'volatility', 'vwap', 'twap'
        }
        
        try:
            # Convert query to lowercase for case-insensitive matching
            query_lower = query.lower()
            
            # Check for direct keyword matches
            for keyword in market_data_keywords:
                if keyword in query_lower.split():
                    return True
            
            # If we have a database connection, check table names for market data indicators
            if self.db is not None:
                tables = self.db.get_usable_table_names()
                table_info = self.db.get_table_info_no_throw(tables)
                
                # Check if table info contains market data related columns
                market_data_patterns = [
                    r'\b(price|quote|trade|market|ticker)\w*\b',
                    r'\b(bid|ask|open|close|high|low)\b',
                    r'\b(volume|vwap|twap)\b',
                    r'\b(exchange|symbol)\b'
                ]
                
                for pattern in market_data_patterns:
                    if re.search(pattern, table_info, re.IGNORECASE):
                        return True
            
            return False

        except Exception as e:
            await self.emit_status(
                __event_emitter__, 
                "error", 
                f"Error: {str(e)}", 
                True
            )
            return False