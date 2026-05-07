"""
Base Agent

Every specialist agent owns its own ChatXAI instance configured for its
specific model. Subclasses declare MODEL and SYSTEM_PROMPT as class attributes.
Instances are created once and cached in app.extensions by the coordinator.
"""

import logging
from abc import ABC, abstractmethod
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Subclasses MUST declare:
      - MODEL       : the xAI model string (e.g. "grok-3", "grok-3-mini")
      - SYSTEM_PROMPT : the agent's identity and scope

    Python enforces this — instantiating BaseAgent directly, or any subclass
    that omits MODEL or SYSTEM_PROMPT, raises TypeError immediately.

    Architecture:
      - Each agent owns its own ChatXAI instance (different models per task)
      - System prompt defines the agent's identity and scope
      - Untrusted data is passed in the human turn inside <untrusted_data> tags
      - with_structured_output() enforces typed Pydantic output — injected text
        stays in string fields and cannot propagate as executable instructions
    """

    @property
    @abstractmethod
    def MODEL(self) -> str:
        """xAI model identifier. Use grok-3-mini for simple tasks, grok-3 for complex ones."""

    @property
    @abstractmethod
    def SYSTEM_PROMPT(self) -> str:
        """Agent identity and scope. Keep it narrowly focused on one task."""

    def __init__(self, api_key: str):
        self.llm = ChatXAI(model=self.MODEL, temperature=0, api_key=api_key)
        self._prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        logger.info("Initialized %s using model=%s", self.__class__.__name__, self.MODEL)

    def _invoke_structured(self, schema_class, input_text: str):
        """Format prompt → call structured LLM → return typed Pydantic object."""
        chain = self._prompt | self.llm.with_structured_output(schema_class)
        return chain.invoke({"input": input_text})
