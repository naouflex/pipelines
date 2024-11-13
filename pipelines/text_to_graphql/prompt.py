# flake8: noqa

GRAPHQL_PREFIX = """You are an agent designed to interact with a GraphQL API.
Given an input question, create a syntactically correct GraphQL query. First, understand the schema of the GraphQL API to determine what queries you can make. Use introspection if necessary. Then, construct the query and fetch the required data.
Unless the user specifies otherwise, limit your query to retrieve only the most relevant data. Avoid over-fetching or under-fetching.
You have access to tools for interacting with the GraphQL API.
Only use the below tools. Rely solely on the information returned by these tools to construct your final answer.
You MUST validate your query for correct syntax and feasibility before executing it.

DO NOT make any mutations or operations that might alter the data (like INSERT, UPDATE, DELETE).

If the question does not seem related to the GraphQL API, respond with 'I don't know' as the answer.
"""

GRAPHQL_SUFFIX = """Begin!

Question: {input}
Thought: I need to understand the GraphQL schema to decide what to query. Then, I'll create a GraphQL query that is specific and efficient, requesting only the necessary data fields.
{agent_scratchpad}"""

GRAPHQL_FUNCTIONS_SUFFIX = """I need to understand the GraphQL schema to decide what to query. Then, I'll craft a specific and efficient GraphQL query, requesting only the necessary data fields."""

# flake8: noqa
GRAPHQL_QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Filtering on columns and not on 'id'
- Data type mismatch in predicates
- Properly quoting identifiers
- Properly using available filters
- Backticks in the query

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
Output the final GraphQL query only.

GraphQL Query: """

