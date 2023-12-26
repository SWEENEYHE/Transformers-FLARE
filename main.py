'''



'''
import os
from tools.FLARE.base import FlareChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import BaseLLM
import requests
from typing import Any, List, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document, BaseMessage, PromptValue, LLMResult, Generation
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.globals import set_verbose

set_verbose(True)
os.environ["SERPER_API_KEY"] = ""


# MyRetriver
'''
    可以实现自己的检索器
'''
class MyRetriver(BaseRetriever):
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        return None

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()


class SerperSearchRetriever(BaseRetriever):
    search: GoogleSerperAPIWrapper = None

    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        return [Document(page_content=self.search.run(query))]

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()


retriever = SerperSearchRetriever(search=GoogleSerperAPIWrapper())


# MyLLM
'''
    自定义模型
'''
class MyResponseLLM(BaseLLM):
    SYSTEM_PORMPT = "你会回答用户问题\n"
    max_length = 2048
    def __init__(self, url):
        self.url = url
    def httpLLM(self, prompt):
        print("url: ", self.url)
        print("prompt : ", prompt)
        response = requests.post(url=self.url,
                                 json={"query": self.SYSTEM_PORMPT + prompt + "回答：",
                                       "max_new_tokens": self.max_length})
        assert response.status_code == 200, f"{response.status_code}  http connection unstable "
        return response.json()

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = self.httpLLM(prompt)
            generation = Generation(
                text=response['text'],
                generation_info=dict(
                    logprobs=dict(
                        tokens=response['tokens'],
                        token_logprobs=response['token_probs']
                    )
                )
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "MyLLM"


if __name__ == "__main__":
    url = "http:xxx.xxx.xxx.xxx:9090/queries_chat"
    myLLM = MyResponseLLM(url = url)

    flare = FlareChain.from_llm(
        llm=myLLM,
        assistantLLM=myLLM,
        retriever=retriever,
        max_generation_len=1024,
        min_prob=0.5,
    )
    myLLM2 = MyResponseLLM(url = url)
    question = "为什么蘑菇星球没有火车？"

    result1 = myLLM2.predict(question)
    print(result1)

    print("=======")

    result2 = flare.run(question)
    print(result2)
