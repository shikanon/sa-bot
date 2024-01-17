"""
agent tools.
author: shikanon
create: 2023/10/6
"""
import re
import doubao
from typing import Any
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage, HumanMessage

fn = doubao.ModelFunctionClass(
    name="CallHumanCustomerService",
    description="""当需要转人工服务的时候使用此函数。当你觉得无法很好帮助客户解决问题，
    需要借助专业人士的力量，你可以使用此函数方法转到的人工客服，这个函数的输入question是你基于上下文总结归纳的用户问题和描述""",
    parameters={
        "properties": {
            "question": {"description": "用户咨询的问题和问题相关的上下文信息", "type":"string"}
            },
            "required": ["question"],
            "type": "object",
    },
)

chat = doubao.ChatSkylark(
    model="skylark2-pro-4k",
    model_version="1.100",
    model_endpoint="mse-20231227193502-58xhk",
    top_k=1,
    functions=[fn.todict()]
    )

system_prompt = """## Character
你是一位智能游戏客服，礼貌且善于洞察客户情绪。你的主要职责是帮助客户解决他们在游戏中遇到的问题。

## Skills
### Skill 1: 洞察客户情绪
1. 通过用户的反馈和描述，尝试理解客户的情绪。
2. 你的回复应适配客户的情绪。如客户表现出沮丧，应给予关心和理解；若客户高兴，你应和他们一同分享喜悦。

### Skill 2: 解决客户问题
1. 当用户反馈问题时，应详细询问问题的情况，如何操作会产生这个问题，以了解问题的全貌。
2. 在得到足够信息后，提供详细的解决步骤来帮助他们解决问题。

### Skill 3: 提供礼貌的服务
1. 无论任何情况下，都应保持礼貌，展示专业和尊重。
2. 遇到问题时，应先道歉，然后提出解决方案。

## Constraints
- 你的回复只应与游戏相关的问题和答案有关。
- 不论客人的情绪如何，都应礼貌地回复，不允许表现出任何不专业的行为。
- 在尽量短的时间里为客户提供有效的解决方案，使他们满意。
"""

prompt = """
Q: 我的充值金额出现了问题，我充值了50块却只显示10块钱，快帮我看看
A: 好的，请问您方便提供下您的账号和订单号吗？
Q: {question}
A:
"""

question="我的账号有问题需要立刻帮我转人工"
messages = [
    SystemMessage(content=system_prompt),
    HumanMessagePromptTemplate.from_template(
        template=prompt,
    ).format(question=question),
]

# req = chat(messages)
# if "function_call" in req.additional_kwargs:
#     fn_name = req.additional_kwargs["function_call"]
#     print(fn_name)
# print(req)


from sentence_transformers import SentenceTransformer
from elasticsearch7 import Elasticsearch

#初始化ES Client
# es = Elasticsearch( "https://admin:Pwd@12345@elasticsearch-lexoxskha9dichht-f1zvzj7w.escloud.volces.com:9200", verify_certs=False, ssl_show_warn=False,)

#加载Embedding 模型
# from volcengine.maas import MaasService, MaasException, ChatRole
# maas = MaasService('ml-maas-api.bytedance.net', 'cn-beijing')
# req = {
#     "model": {
#         "name": "bge-large-zh",
#         "version": "1.0", # use default version if not specified.
#     },
#     "input": [
#         "天很蓝",
#         "海很深"
#     ]
# }
# resp = maas.embeddings(req)
# print(resp)

import knowledge
kg = knowledge.MaaSKnowledgeEmbedding("bge-large-zh","1.0")
resps = kg.encode(["天蓝","海深"])
for resp in resps:
    print(resp)

# kg.save(system_prompt)