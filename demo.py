import  doubao

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from functools import wraps


multi_choice_prompt = """请针对 >>> 和 <<< 中间的用户问题，选择一个适合的工具去回答他的问题，只要用A、B、C的选项字母告诉我答案。
如果你觉得都不适合，就选D。

>>> {question} <<<

你能使用的工具如下：
A. 一个为用户进行商品导购和推荐的工具
B. 一个能够查询最近下单的订单信息，获得最新的订单情况的工具
C. 一个能够查询商家的退换货政策、运费、物流时长、支付渠道的工具
D. 都不适合

请按以下格式进行回答`A`、`B`、`C`、`D`。
"""

multi_choice_tools_prompt = """
请针对 >>> 和 <<< 中间的用户问题，选择一个适合的工具去回答他的问题，工具的名称已经给出。
如果你觉得都不适合，就回复“no_tools: 以上工具都不适用”。

>>> {question} <<<

你能使用以下四个工具：
- recommend_product: 一个为用户进行商品导购和推荐的工具
- search_order: 一个能够查询最近下单的订单信息，获得最新的订单情况的工具
- search_merchant_policies: 一个能够查询商家的退换货政策、运费、物流时长、支付渠道的工具
- no_tools: 以上工具都不适用

请按以下格式进行回答:
{{
    "recommend_product": "一个为用户进行商品导购和推荐的工具"
}}
"""


agent_prompt = """Answer the following questions as best you can. You have access to the following tools:

Search Order:
一个能够查询订单信息，获得最新的订单情况的工具，参数是输入订单id
Recommend product: 一个能够基于商品及用户
信息为用户进行商品推荐导购的工具，参数是输入要推荐的商品类型

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: Search Order, Recommend product

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

ALWAYS use the following format:

Question: the input question you
must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation:
the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the
final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.'
{question}
"""

def print_decorator(text: str):
    def decorator(func):
        @wraps(func)
        def wrapfunction(*args, **kwargs):
            print("\n\n===start===\n")
            print(text)
            return func(*args, **kwargs)
        return wrapfunction
    return decorator

@print_decorator("模拟agent判断选择：")
def test_choice():
    # 模拟agent选择题
    chat = doubao.ChatSkylark(model="skylark-chat",top_k=1)
    question="我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
    messages = [
        HumanMessagePromptTemplate.from_template(
            template=multi_choice_prompt,
        ).format(question=question),
    ]
    req = chat(messages)
    print("问题: %s\n"%question)
    print(req.content)

@print_decorator("模拟agent对工具的选择：")
def test_choice_tools():
    # 模拟选择工具
    chat = doubao.ChatSkylark(model="skylark-chat",top_k=1)
    choice_chain = LLMChain(llm=chat,prompt=PromptTemplate(template=multi_choice_tools_prompt,input_variables=["question"]),output_key="answer")
    question_1 = "我有一张订单，一直没收到，可以帮我查询下吗"
    result = choice_chain(question_1)
    print("问题: %s\n"%question_1)
    print(result["answer"])
    question_2 = "请问你们家的货可以送到四川吗，物流大概要多久？"
    result = choice_chain(question_2)
    print("问题: %s\n"%question_2)
    print(result["answer"])
    question_3 = "你们家什么款式最畅销？"
    result = choice_chain(question_3)
    print("问题: %s\n"%question_3)
    print(result["answer"])

# 模拟agent prompt生成答案
@print_decorator("模拟agent规划、选择：")
def test_agent_prompt():
    chat = doubao.ChatSkylark(model="skylark-chat",top_k=1)
    question="我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
    messages = [
        HumanMessagePromptTemplate.from_template(
            template=agent_prompt,
        ).format(question=question),
    ]
    result = chat(messages)
    # agent的解析
    import re
    import json
    text = result.content
    pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL) # 匹配```符合中符合json的
    found = pattern.search(text)
    action = found.group(1)
    response = json.loads(action.strip())
    print("问题: %s\n"%question)
    print(response) #json解析后已经满足json格式

# 模拟电商订单
def search_order(input: str)->str:
    print("你需要调用search_order，一个能够查询订单信息，获得最新的订单情况的工具。")
    return "{order}，订单状态：已发货".format(order=input)

# 模拟商品推荐
def recommend_product(input: str)->str:
    print("你需要调用recommend_product，一个能够基于商品及用户信息为用户进行商品推荐导购的工具。")
    return "黑色连衣裙"

@print_decorator("模拟agent全流程(thought、action、observation)：")
def test_agent():
    tools = [
        Tool(
            name="Search Order",
            func=search_order,
            description="""一个能够查询订单信息，获得最新的订单情况的工具，参数是输入订单id"""
        ),
        Tool(
            name="Recommend product",
            func=recommend_product,
            description="一个能够基于商品及用户信息为用户进行商品推荐导购的工具，参数是输入要推荐的商品类型"
        )
    ]

    chat = doubao.ChatSkylark(model="skylark-chat",top_k=1)
    agent_tools = initialize_agent(tools=tools, llm=chat, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True,agent_kwargs={"handle_parsing_errors": True})
    user_question = "我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
    print("用户问题：%s "%user_question)
    result = agent_tools.run(user_question)
    print(result)


if __name__ == "__main__":
    test_choice()
    test_choice_tools()
    test_agent_prompt()
    test_agent()