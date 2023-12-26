# Transformers-FLARE
An implementation version of Forward-Looking Active Retrieval augmented generation (FLARE) compatible with **Transformers and Chinese**.
The original project is [here](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb) and the paper project is [here](https://github.com/jzbjyb/FLARE)

本项目是基于[langchain-cookbook FLARE](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb)
的适配中文和Transformers库模型的实现，由于langchain cookbook中仅适配了openAI的接口，为适配Transformers库及中文实现该项目。这里是[原始论文和代码](https://github.com/langchain-ai/langchain/blob/master/cookbook/forward_looking_retrieval_augmented_generation.ipynb)。
# 0.Background 背景
 Large models based on autoregressive token generation can generate many illusions. To address this issue, it is often necessary to inject external knowledge based on the RAG (Retrieval-Augmented Generation) approach. However, introducing external information without consideration may actually weaken the performance of large models. To tackle this, FLARE proposes RAG on low-confidence portions during token generation, where low confidence indicates model uncertainty or illusions. Specifically, two measures are taken:

i) Explicitly: Directly masking tokens with low confidence and using this portion as a query for retrieval.
ii) Implicitly: Constructing questions based on the low-confidence portion and using the questions as queries for retrieval.  
    大模型基于自回归接龙的token生成会产生许多幻觉，为了解决这个问题，通常需要基于RAG注入外部知识。然而，不加考虑地引入外部信息可能反而会削弱大模型的表现。为此，FLARE提出基于token生成时的概率对低置信度部分（低置信度说明模型不确定结果即幻觉）进行RAG。具体而言两条措施：
    i) 显式地：直接mask掉低置信度token，将这部分作为query进行检索
    ii) 隐式地：基于低置信度的部分去构造问题，将问题作为query进行检索
    
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
ii)直接mask掉这些span去检索
```python
    flare = FlareChain.from_llm(
        llm=myLLM,
        assistantLLM=myLLM,
        retriever=retriever,
        max_generation_len=1024,
        min_prob=0.5,
    )
```
## 2.4 注意事项
 部分模型（Baichuan）generate方法可能与chat方法不一致，需要额外处理（如在首尾添加一些special token，user_token_id, assistant_token_id等）才能符合原始训练等组织形式，本项目在服务端对Baichuan做了适配，其他模型可能需要参考进行修改
```python
# 按照chat接口实现，前面增加user_id，最后增加assistant_id,按照按照百川模型训练规则进行生成，确保输出结果流畅
#'<reserved_106>'
user_id = tokenizer.decode(model.generation_config.user_token_id)
#'<reserved_107>'
assistant_id = tokenizer.decode(model.generation_config.assistant_token_id)
query = user_id + query + assistant_id
```
