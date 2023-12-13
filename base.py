from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from tools.FLARE.ChinesePrompts import (
    PROMPT,
    QUESTION_GENERATOR_PROMPT,
    FinishedOutputParser,
)
from langchain.chains.llm import LLMChain
from langchain.llms import BaseLLM
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate, BaseRetriever, Generation, LLMResult
from langchain.schema.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.language_model import BaseLanguageModel

"""
LLMResult(
generations=[[Generation(text='xxx', generation_info={'finish_reason': 'stop', 'logprobs': <OpenAIObject at 0x11ef98130> JSON: {
"tokens": [
    "\n",
    "\n",
    "bytes:\\xe8",
    "bytes:\\x82",
    "bytes:]

"token_logprobs": [
    -0.00070218404,
    -0.00089621654,
    -0.00010926496,
    -0.00048095852,
    -2.319992e-05,
    -5.5577775e-06]
}
"""


# MyLLM
class MyResponseLLM(BaseLLM):
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        return None

    @property
    def _llm_type(self) -> str:
        return "MyLLM"


class _ResponseChain(LLMChain):
    """Base class for chains that generate responses."""

    prompt: BasePromptTemplate = PROMPT

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    def generate_tokens_and_log_probs(
            self,
            _input: Dict[str, Any],
            *,
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[Sequence[str], Sequence[float]]:
        llm_result = self.generate([_input], run_manager=run_manager)
        return self._extract_tokens_and_log_probs(llm_result.generations[0])

    @abstractmethod
    def _extract_tokens_and_log_probs(
            self, generations: List[Generation]
    ) -> Tuple[Sequence[str], Sequence[float]]:
        """Extract tokens and log probs from response."""


class _TransformerResponseChain(_ResponseChain):
    """Chain that generates responses from user input and context."""

    llm: MyResponseLLM = Field(default_factory=lambda: MyResponseLLM())

    def _extract_tokens_and_log_probs(
            self, generations: List[Generation]
    ) -> Tuple[Sequence[str], Sequence[float]]:
        tokens = []
        log_probs = []
        for gen in generations:
            if gen.generation_info is None:
                raise ValueError
            tokens.extend(gen.generation_info["logprobs"]["tokens"])
            log_probs.extend(gen.generation_info["logprobs"]["token_logprobs"])
        return tokens, log_probs


class QuestionGeneratorChain(LLMChain):
    """Chain that generates questions from uncertain spans."""

    prompt: BasePromptTemplate = QUESTION_GENERATOR_PROMPT
    """Prompt template for the chain."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain."""
        return ["user_input", "context", "response"]


def _low_confidence_spans(
        tokens: Sequence[str],
        token_probs: Sequence[float],
        min_prob: float,
        min_token_gap: int,
        num_pad_tokens: int,
) -> List[str]:
    # 1.筛选所有低于阈值min_prob的token
    # _low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    _low_idx = np.where(np.array(token_probs) < min_prob)[0]

    # 2.筛选token为至少 \w(包含一个字母、数字、下划线）以及\u4e00-\u9fa5 中文
    low_idx = [i for i in _low_idx if re.search(r"[\w\u4e00-\u9fa5]", tokens[i])]

    if len(low_idx) == 0:
        return []

    # 3.先将获得连续低置信度token的范围，放到list中
    spans = [[low_idx[0], low_idx[0] + num_pad_tokens + 1]]

    # 4.遍历后续的结果，控制两个低置信度区间的距离
    for i, idx in enumerate(low_idx[1:]):
        end = idx + num_pad_tokens + 1
        if idx - low_idx[i] < min_token_gap:
            spans[-1][1] = end
        else:
            spans.append([idx, end])
    # 5.合并每个低置信度区间的token作为一个新的最小单位，返回低置信度区间
    return ["".join(tokens[start:end]) for start, end in spans]


class FlareChain(Chain):
    """Chain that combines a retriever, a question generator,
    and a response generator."""

    question_generator_chain: QuestionGeneratorChain
    """Chain that generates questions from uncertain spans."""
    response_chain: _ResponseChain = Field(default_factory=_TransformerResponseChain)
    """Chain that generates responses from user input and context."""
    output_parser: FinishedOutputParser = Field(default_factory=FinishedOutputParser)
    """Parser that determines whether the chain is finished."""
    retriever: BaseRetriever
    """Retriever that retrieves relevant documents from a user input."""
    min_prob: float = 0.2
    """Minimum probability for a token to be considered low confidence."""
    min_token_gap: int = 5
    """Minimum number of tokens between two low confidence spans."""
    num_pad_tokens: int = 2
    """Number of tokens to pad around a low confidence span."""
    max_iter: int = 10
    """Maximum number of iterations."""
    """Whether to start with retrieval."""

    @property
    def input_keys(self) -> List[str]:
        """Input keys for the chain."""
        return ["user_input"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys for the chain."""
        return ["response"]

    def _do_generation(
            self,
            questions: List[str],
            user_input: str,
            response: str,
            _run_manager: CallbackManagerForChainRun,
    ) -> Tuple[str, bool]:
        callbacks = _run_manager.get_child()
        docs = []
        # 把所有相关问题都拿去检索
        for question in questions:
            docs.extend(self.retriever.get_relevant_documents(question))

        # 拼接检索结果
        context = "\n\n".join(d.page_content for d in docs)
        # 根据query，相关问题答案让LLM去回答
        result = self.response_chain.predict(
            user_input=user_input,
            context=context,
            response=response,
            callbacks=callbacks,
        )
        # 获取结果 和 结束标识
        marginal, finished = self.output_parser.parse(result)
        return marginal, finished

    """
        根据低置信度sapn生成相关问题，将相关问题拿去检索
    """

    def _do_retrieval(
            self,
            low_confidence_spans: List[str],
            _run_manager: CallbackManagerForChainRun,
            user_input: str,
            response: str,
            initial_response: str,
    ) -> Tuple[str, bool]:

        question_gen_inputs = [
            {
                "user_input": user_input,
                "current_response": initial_response,
                "uncertain_span": span,
            }
            for span in low_confidence_spans
        ]

        callbacks = _run_manager.get_child()
        # 基于低置信度的span去生成相关的问题
        question_gen_outputs = self.question_generator_chain.apply(
            question_gen_inputs, callbacks=callbacks
        )
        # 提取问题
        questions = [
            output[self.question_generator_chain.output_keys[0]]
            for output in question_gen_outputs
        ]

        _run_manager.on_text(
            f"Generated Questions: {questions}", color="yellow", end="\n"
        )

        return self._do_generation(questions, user_input, response, _run_manager)

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        user_input = inputs[self.input_keys[0]]

        response = ""

        for i in range(self.max_iter):
            _run_manager.on_text(
                f"Current Response: {response}", color="blue", end="\n"
            )
            _input = {"user_input": user_input, "context": "", "response": response}
            # 获得生成结果token和对应概率
            tokens, log_probs = self.response_chain.generate_tokens_and_log_probs(
                _input, run_manager=_run_manager
            )
            # 根据token、概率、阈值和间隔获得连续的 低置信度span
            low_confidence_spans = _low_confidence_spans(
                tokens,
                log_probs,
                self.min_prob,
                self.min_token_gap,
                self.num_pad_tokens,
            )
            # 合并初始的输出
            initial_response = response.strip() + " " + "".join(tokens)

            # 如果没有低置信度的span直接输出
            if not low_confidence_spans:
                response = initial_response
                final_response, finished = self.output_parser.parse(response)
                if finished:
                    return {self.output_keys[0]: final_response}
                continue

            # 检索
            marginal, finished = self._do_retrieval(
                low_confidence_spans,
                _run_manager,
                user_input,
                response,
                initial_response,
            )
            #
            response = response.strip() + " " + marginal
            if finished:
                break
        return {self.output_keys[0]: response}

    @classmethod
    def from_llm(
            cls, llm: BaseLanguageModel, assistantLLM: BaseLanguageModel, max_generation_len: int = 32, **kwargs: Any
    ) -> FlareChain:
        """Creates a FlareChain from a language model.

        Args:
            llm: 根据检索信息回答问题
            assistantLLM:生成低置信度为答案的span的相关问题
            max_generation_len: Maximum length of the generated response.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            FlareChain class with the given language model.
            :param llm:
            :param max_generation_len:
            :param assistantLLM:
        """
        question_gen_chain = QuestionGeneratorChain(llm=llm)
        response_chain = _TransformerResponseChain(llm=assistantLLM)

        return cls(
            question_generator_chain=question_gen_chain,
            response_chain=response_chain,
            **kwargs,
        )
