from datetime import datetime
from langchain.pydantic_v1 import BaseModel, Extra, Field, root_validator
import json
from typing import Any, Optional
import requests
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from pipelines.text_to_graphql.wrapper import GraphQLAPIWrapper
from pipelines.text_to_graphql.prompt import GRAPHQL_QUERY_CHECKER
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import gql
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from typing import List

class BaseGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""

    graphql_wrapper: GraphQLAPIWrapper

    name: str = "query_graphql"

    description: str = """
    Base tool for querying a GraphQL API.
    """

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class SchemaGraphQLTool(BaseGraphQLTool):
    """Tool for getting tables information about a GraphQL database."""
    name: str = "graphql_get_tables_info"
    description: str = """
    Input is an empty string, output is the schema and sample rows for those tables.    
    Always use this tool before executing a query with graphql_run_query!
    """
    def _run(
        self,
        input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the schema or an error message."""
        return self.get_introspection_result()
    
    def get_introspection_result(self):
            try:
                introspection_result = requests.post(
                    self.graphql_wrapper.graphql_endpoint, 
                    json={'query': 'query IntrospectionQuery { __schema { types { name fields { name  } } } }'}
                    ).json()

                introspection_result_string = ""

                for type in introspection_result['data']['__schema']['types'] :
                    try:
                        introspection_result_string += type['name'] + " : ["
                        for field in type['fields']:
                            introspection_result_string += field['name'] + ", "
                        introspection_result_string += "]\n"
                    except:
                        pass
                introspection_result_string = introspection_result_string[:-3]
            
                return json.dumps(introspection_result)
            except Exception as e:
                return str("""Error: The subgraph is not available.""")

class GraphQLCheckerTool(BaseGraphQLTool):
    """Use an LLM to check if a query is correct"""

    template: str = GRAPHQL_QUERY_CHECKER
    llm: BaseLanguageModel
    llm_chain: LLMChain = Field(init=False)
    name: str = "graphql_query_syntax_checker"
    description: str = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with graphql_run_query!
    """

    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),
                prompt=PromptTemplate(
                    template=GRAPHQL_QUERY_CHECKER, input_variables=["dialect", "query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["dialect", "query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )

        return values

    def _run(
        self,

        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(
            query=query,
            dialect='graphql',
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            dialect='graphql',
            callbacks=run_manager.get_child() if run_manager else None,
        )

class QueryGraphQLTool(BaseGraphQLTool):
    """Tool for querying a GraphQL database."""

    name: str = "graphql_run_query"
    description: str = """
    Input to this tool is a detailed and correct GraphQL query without single quotes, output is a result from the graph endpoint.\
    If the query is not correct, an error message will be returned.  The input should not include any backticks !\
    If an error is returned, rewrite the query, check the query, and try again.\
    """

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the results or an error message."""
        try:
            return self.graphql_wrapper.run(query)
        except Exception as e:
            return json.dumps(e)

class OutputFormatTool(BaseGraphQLTool):
    """Tool to format the final answer."""
    
    llm: BaseLanguageModel
    llm_chain: LLMChain = Field(init=False)
    name: str = "graphql_format_answer"
    description: str = """
    Input is an answer, output is the final answer with formatted numbers and timestamps as a string.
    Always use this tool at the very end.
    """
    @root_validator(pre=True)
    def initialize_llm_chain(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "llm_chain" not in values:
            values["llm_chain"] = LLMChain(
                llm=values.get("llm"),
                prompt=PromptTemplate(
                    template="""
                            {query}
                            - Convert prices to human readable format.
                            - Add etherscan link for transactions.
                            - Give prices in USD
                            - Format numbers to accounting string when appropriate.
                            - Format dates to human readable format when appropriate.
                            - Convert timestamps to human readable format when appropriate.
                            - Start block and end block are not timestamps so do not convert them.
                            Final Answer:
                            """, input_variables=["query"]
                ),
            )

        if values["llm_chain"].prompt.input_variables != ["query"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query']"
            )

        return values

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the LLM to format your final answer."""
        return self.llm_chain.predict(
            query=query,
            dialect='graphql',
            callbacks=run_manager.get_child() if run_manager else None,
        )

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return await self.llm_chain.apredict(
            query=query,
            dialect='graphql',
            callbacks=run_manager.get_child() if run_manager else None,
        )

    
class TimestampConverterTool(BaseGraphQLTool):
    """Tool for converting a non unix timestamp (not a block) to a human readable format."""
    name: str = "graphql_timestamp_converter"
    description: str = """
    Input is a timestamp (not a block number !) , output is a human readable date.
    Always use this tool if there is a timestamp in the final message.
    """
    def _run(
        self,
        timestamp,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Convert a non unix timestamp to a human readable format."""
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
    
    
    async def _arun(
        self,
        timestamp,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')

class PriceConverterTool(BaseGraphQLTool):
    """Tool for converting a price to a human readable format."""
    
    name: str = "graphql_price_converter"
    description: str = """
    Input is a price in wei, output is the same price divided by 10^18.
    Always use this tool if there is a price in wei in the final message.
    """

    def _run(
        self,
        number,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Convert price to a human readable format wiht 2 decimal places."""
        return "{:,.2f}".format(int(number)/1e18)

    async def _arun(
        self,
        number: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return "{:,.2f}".format(int(number)/1e18)

class TransactionFormatterTool(BaseGraphQLTool):
    """Tool for converting a transaction hash to an etherscan link in the final message"""
    
    name: str = "graphql_transaction_formatter"
    description: str = """
    Input is a transaction id, output is a link to the transaction on etherscan.
    Always use this tool if there is a transaction hash in the final message.
    """

    def _run(
        self,
        transaction_hash: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use python to convert transaction hash to a human readable format."""
        shortened_transaction_hash = transaction_hash[:6] + "..." + transaction_hash[-4:]
        return f"[{shortened_transaction_hash}](https://etherscan.io/tx/{transaction_hash})"

    async def _arun(
        self,
        transaction_hash: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        shortened_transaction_hash = transaction_hash[:6] + "..." + transaction_hash[-4:]
        return f"[{shortened_transaction_hash}](https://etherscan.io/tx/{transaction_hash})"

class GetERC20ContractsTool(BaseGraphQLTool):
    """Tool for getting erc20 contracts info from a GraphQL database."""
    name: str = "graphql_get_contracts_infos"
    description: str = """
    Use this tool at the beginning of every query to get the erc20 contracts info.
    Always use this tool before executing a query with graphql_run_query!
    """
    def _run(
        self,
        input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the schema or an error message."""
        return self.get_erc20_contracts()
    
    def get_erc20_contracts(self):
            try:
                erc20_contracts = requests.post(
                    self.graphql_wrapper.graphql_endpoint, 
                    json={'query': f'query {{ erc20Contracts {{ id name symbol decimals totalSupply}} }}'}
                    ).json()
            
                return json.dumps(erc20_contracts)
            except Exception as e:
                return str("""Error: The subgraph is not available.""")

class GetMarketInfoTool(BaseGraphQLTool):
    """Tool for getting market info from a GraphQL database."""
    name: str = "graphql_get_market_info"
    description: str = """
    Use this tool to obtain the markets information like price, symbol etc.
    Always use this tool before executing a query with graphql_run_query!
    """
    def _run(
        self,
        input: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute the query, return the schema or an error message."""
        try:
            market_info = requests.post(
                self.graphql_wrapper.graphql_endpoint, 
                json={'query': f'query {{ markets {{ id collateral {{ id symbol decimals }} collateralFactorBPS price }} }}'}
                ).json()
        
            return json.dumps(market_info)
        except Exception as e:
            return json.dumps(e)
            
class GetProposalIdAndLink(BaseGraphQLTool):
    """Tool for converting a proposal id to an integer."""
    name: str = "graphql_get_proposal_id_and_link"
    description: str = """
    Always use this tool if there is a proposal id to an integer in your final answer.
    Input is the id of the proposal in hex, output is the id of the proposal in integer.
    Example Input : 0xbeccb6bb0aa4ab551966a7e4b97cec74bb359bf6/0x9f
    Example output : 159
    """

    def _run(
        self,
        proposal_id: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        proposal_id = proposal_id.split("/")[1]
        proposal_id = int(proposal_id,16)
        link_to_proposal = f"[See proposal](https://www.inverse.finance/governance/proposals/mills/{str(proposal_id)})"
        text ="""Proposal id: """ + str(proposal_id) + """ Link: """ + link_to_proposal
        return text


class GetAddressLinkTool(BaseGraphQLTool):
    
    name: str = "graphql_address_link_formatter"
    description: str = """
    Tool to convert an ethereum address to a link in the final message
    Input is an address, output is a link to the address on etherscan.
    Always use this tool if there is an account id or contract id in the final message.
    Example Input : 0x865377367054516e17014ccded1e7d814edc9ce4
    """

    def _run(
        self,
        address: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use python to convert address to a human readable format."""
        shortened_address = address[:6] + "..." + address[-4:]
        return f"[{shortened_address}](https://etherscan.io/address/{address})"

    async def _arun(
        self,
        address: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        shortened_address = address[:6] + "..." + address[-4:]
        return f"[{shortened_address}](https://etherscan.io/address/{address})"
    
class GetBlockLinkTool(BaseGraphQLTool):
    name: str = "graphql_block_link_formatter"
    description: str = """
    Tool for converting a block number to a link in the final message
    Input is a block number, output is a link to the block on etherscan.
    Always use this tool if there is a block number in the final message.
    """

    def _run(
        self,
        block_number: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use python to convert block number to a human readable format."""
        return f"[{block_number}](https://etherscan.io/block/{block_number})"

    async def _arun(
        self,
        block_number: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        return f"[{block_number}](https://etherscan.io/block/{block_number})"
   
