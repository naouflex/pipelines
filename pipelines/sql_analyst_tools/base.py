"""SQL agent."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools import BaseTool

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.agent_types import AgentType

def create_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SQLDatabaseToolkit,
    valves: Any,
    agent_type: Optional[AgentType] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    extra_tools: Sequence[BaseTool] = (),
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a SQL agent from an LLM and tools."""
    from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
    from langchain.agents.agent_types import AgentType
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
    from langchain.chains.llm import LLMChain

    agent_type = agent_type or AgentType.ZERO_SHOT_REACT_DESCRIPTION
    tools = toolkit.get_tools() + list(extra_tools)
    
    prefix = valves.PROMPT_SQL_PREFIX.format(dialect=toolkit.dialect, top_k=top_k)
    agent: BaseSingleActionAgent

    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt = ZeroShotAgent.create_prompt(
            tools,
            prefix=prefix,
            suffix=valves.PROMPT_SQL_SUFFIX,
            input_variables=input_variables,
        )
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)

    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        messages = [
            SystemMessage(content=prefix),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=valves.PROMPT_SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        input_variables = ["input", "agent_scratchpad"]
        _prompt = ChatPromptTemplate(input_variables=input_variables, messages=messages)

        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )

