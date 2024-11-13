from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from pipelines.text_to_graphql.wrapper import GraphQLAPIWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field

from langchain_community.agent_toolkits.base import BaseToolkit
from typing import List

from pipelines.text_to_graphql.tool import (
    GraphQLCheckerTool,
    GetAddressLinkTool,
    GetBlockLinkTool,
    GetERC20ContractsTool,
    GetMarketInfoTool,
    GetProposalIdAndLink,
    OutputFormatTool,
    PriceConverterTool,
    QueryGraphQLTool,
    SchemaGraphQLTool,
    TimestampConverterTool,
    TransactionFormatterTool,
)


class GraphQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with GraphQL databases."""

    graphql_wrapper: GraphQLAPIWrapper = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    @property
    def dialect(self) -> str:
        return 'graphql'

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return  [
        SchemaGraphQLTool(graphql_wrapper=self.graphql_wrapper),
        GraphQLCheckerTool(graphql_wrapper=self.graphql_wrapper,llm=self.llm),
        QueryGraphQLTool(graphql_wrapper=self.graphql_wrapper),
        OutputFormatTool(graphql_wrapper=self.graphql_wrapper,llm=self.llm),
        TimestampConverterTool(graphql_wrapper=self.graphql_wrapper),
        PriceConverterTool(graphql_wrapper=self.graphql_wrapper),
        TransactionFormatterTool(graphql_wrapper=self.graphql_wrapper),
        GetERC20ContractsTool(graphql_wrapper=self.graphql_wrapper),
        GetMarketInfoTool(graphql_wrapper=self.graphql_wrapper),
        GetProposalIdAndLink(graphql_wrapper=self.graphql_wrapper),
        GetAddressLinkTool(graphql_wrapper=self.graphql_wrapper),
        GetBlockLinkTool(graphql_wrapper=self.graphql_wrapper),
    ]