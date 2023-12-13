from typing import Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser


class FinishedOutputParser(BaseOutputParser[Tuple[str, bool]]):
    """Output parser that checks if the output is finished."""

    finished_value: str = "</s>"
    """Value that indicates the output is finished."""

    def parse(self, text: str) -> Tuple[str, bool]:
        cleaned = text.strip()
        finished = self.finished_value in cleaned
        return cleaned.replace(self.finished_value, ""), finished


PROMPT_TEMPLATE = """\
参考context信息回答用户的问题。 \
如果context不为空，则使用context内容验证回答,回答结束使用</s>标识。
>>> CONTEXT: {context}
>>> USER INPUT: {user_input}
>>> RESPONSE: {response}\
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["user_input", "context", "response"],
)


QUESTION_GENERATOR_PROMPT_TEMPLATE = """\
Given a user input and an existing partial response as context, \
ask a question to which the answer is the given term/entity/phrase:

>>> USER INPUT: {user_input}
>>> EXISTING PARTIAL RESPONSE: {current_response}

The question to which the answer is the term/entity/phrase "{uncertain_span}" is:"""

QUESTION_GENERATOR_PROMPT = PromptTemplate(
    template=QUESTION_GENERATOR_PROMPT_TEMPLATE,
    input_variables=["user_input", "current_response", "uncertain_span"],
)
