# Chinese-FLARE
An implementation version of Forward-Looking Active Retrieval augmented generation (FLARE) compatible with **Chinese and Transformers**.
The original project is [here](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb) and the paper project is [here](https://github.com/jzbjyb/FLARE)

本项目是基于[langchain-cookbook FLARE](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb)
的适配中文和Transformers库模型的实现，由于langchain cookbook中仅适配了openAI的接口，为适配Transformers库及中文实现该项目。这里是[原始论文和代码](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb)。
# 1.项目结构
为方便部署和调试，将模型分离成独立的服务，FLARE逻辑独立，两者基于http进行连接：  
(1) model-server.py : 模型部署服务  
(2) base.py: FLARE主要流程  
(3) ChinesePrompts.py: FLARE中使用的prompt  
(4) main.py: 运行FLARE  

# 2.自定义
## 2.1 自定义检索器
```python
class MyRetriver(BaseRetriever):
    def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        #实现检索逻辑
        return [Document(page_content="")]

    async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError()
```
## 2.2 自定义模型
```python
class MyResponseLLM(BaseLLM):
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = ""
            generation = Generation(
                text=response['text'],
                # 返回token和对应的概率token_probs
                generation_info=dict(
                    logprobs=dict(
                        tokens=response['tokens'],
                        token_logprobs=response['token_probs']
                    )
                )
            )
            generations.append([generation])
        return LLMResult(generations=generations)
```

## 2.3 回答模型与问题生成模型
在FLARE中，需要对概率低的连续token（称之为span）  
i)生成相关的问题或  
ii)直接mask掉这些span去检索，这里实现了第i)种，因此需要使用一个assistant模型生成问题 和一个 回答模型 llm，两个模型可以使用同一个 

```python
    flare = FlareChain.from_llm(
        llm=myLLM,
        assistantLLM=myLLM,
        retriever=retriever,
        max_generation_len=1024,
        min_prob=0.5,
    )
```


