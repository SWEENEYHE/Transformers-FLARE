import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from fastapi import FastAPI

MODEL_PATH = "Baichuan2-13B-Chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", trust_remote_code=True)

app = FastAPI() # 创建 api 对象

class Data(BaseModel):
    query:str
    max_new_tokens:int


@app.post("/queries")
def get_per_token_prob(data:Data):
    print("query post: ",data.query)
    inputs = tokenizer([data.query], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=data.max_new_tokens, return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=0.5)

    # 根据logits计算概率
    # 只截取output部分
    output_sequence = outputs.sequences[:, inputs['input_ids'].shape[-1]:][0]
    response = tokenizer.decode(output_sequence)
    # [bsz=1, seq_len, vocab_size]
    probs_all = torch.stack(outputs.scores, dim= 1).softmax(-1)
    # [seq_len]
    probs = torch.max(probs_all.squeeze(0),dim=1).values.cpu().tolist()
    tokens = [tokenizer.decode(token) for token in output_sequence]

    result = {"text":response, "tokens": tokens, "token_probs":probs}
    print("response: ", response)
    return result

@app.post("/queries_chat")
def get_per_token_prob_chat(data:Data):
    print("query post: ",data.query)
    # 按照chat接口实现，前面增加user_id，最后增加assistant_id,按照按照百川模型训练规则进行生成，确保输出结果流畅
    #'<reserved_106>'
    user_id = tokenizer.decode(model.generation_config.user_token_id)
    #'<reserved_107>'
    assistant_id = tokenizer.decode(model.generation_config.assistant_token_id)

    query = user_id + data.query + assistant_id
    inputs = tokenizer([query], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=data.max_new_tokens, return_dict_in_generate=True, output_scores=True, do_sample=True, temperature=0.5)
    #outputs = model.generate(**inputs, max_new_tokens=data.max_new_tokens, return_dict_in_generate=True, output_scores=True)

    # 根据logits计算概率
    # 只截取output部分
    output_sequence = outputs.sequences[:, inputs['input_ids'].shape[-1]:][0]
    response = tokenizer.decode(output_sequence)
    # [bsz=1, seq_len, vocab_size]
    probs_all = torch.stack(outputs.scores, dim= 1).softmax(-1)
    # [seq_len]
    probs = torch.max(probs_all.squeeze(0),dim=1).values.cpu().tolist()
    tokens = [tokenizer.decode(token) for token in output_sequence]

    result = {"text":response, "tokens": tokens, "token_probs":probs}
    print("response: ", response)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9900)
