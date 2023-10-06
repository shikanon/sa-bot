# 智能体实验室

- 实现将豆包(云雀大模型)接入langchain体系
- 基于langchain测试skylark-chat的prompt agent


## Case

### Agent

Agent 主要利用大模型的推理(reasoning)、模仿(few-shot learning)和规划能力(Planning)，再结合函数调用来实现工具使用(Tools use)，在介绍 Agent 之前，我们先来通过几个简单的例子来学习 Agent 的运行逻辑，同时也测试下 skylark 大模型 Agent 能力。

**利用大模型判断做选择**

我们可以利用大模型从多个选择中选出正确的出来，比如按下面的问题输入大模型：
```python
multi_choice_prompt = """请针对 >>> 和 <<< 中间的用户问题，选择一个适合的工具去回答他的问题，只要用A、B、C的选项字母告诉我答案。
如果你觉得都不适合，就选D。

>>> {question} <<<

你能使用的工具如下：
A. 一个能够查询商品信息为用户进行商品导购的工具
B. 一个能够查询最近下单的订单信息，获得最新的订单情况的工具
C. 一个能够商家的退换货政策、运费、物流时长、支付渠道的工具
D. 都不适合

请按以下格式进行回答`A`、`B`、`C`、`D`。
"""

chat = doubao.ChatSkylark(model="skylark-chat",temperature=0,top_p=0,top_k=1)
question="我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
messages = [
    HumanMessagePromptTemplate.from_template(
        template=multi_choice_prompt,
    ).format(question=question),
]
req = chat(messages)
print("问题: %s"%question)
print(req.content)
```

这个例子可以通过在本地运行`python demo.py`来得到结果。
结果如下：
```
根据提供的信息，最适合的工具是 A. 一个为用户进行商品导购和推荐的工具。因为用户的问题是关于选择适合的衣服，需要推荐和导购。B、C 选项的工具虽然也有用，但并不是最直接解决用户问题的工具。因此，选择 A 选项。回答为`A`。
```

在这里我们构造了一个选择题给到大模型，让大模型从多个选项中选出适合的工具。


**让大模型通过判断正确选择函数工具并输出**

上面例子测试了大模型的推理和选择判断能力，下面我们将上面的 A,B,C,D 换成我们的函数名称，并要求其按照固定格式输出，prompt如下：
```markdown
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
```

测试`skylark-chat`：
```
问题: 我有一张订单，一直没收到，可以帮我查询下吗

{
    "search_order": "一个能够查询最近下单的订单信息，获得最新的订单情况的工具"
}
```
这里可以看到，针对问题按预设的结果输出了所需要的工具，并做了格式，对格式化的json数据就可以被程序所处理。

```
问题: 请问你们家的货可以送到四川吗，物流大概要多久？

根据用户的问题，需要查询商家的退换货政策、运费、物流时长等信息。而给出的四个工具中，search_merchant_policies 能够查询商 家的退换货政策、运费、物流时长、支付渠道等信息，与用户需求相符。

因此，回复内容为：

{
    "search_merchant_policies": "一个能够查询商家的退换货政策、运费、物流时长、支付渠道的工具"
}
```

这里可以看到，针对一些问题，`skylark-chat` 有时不是直接回复结果，而是会在前面解释一通，这是因为`skylark-chat`训练数据用到大量的 CoT 的方式来提升准确率。针对这种结果可以通过正则表达式提取json数据给到程序使用。

**agent模板和解析**

这里 agent 模板使用了经典的`chat zero shot react`，分为"Thought","Action","Observation" 三部分。这里直接看 prompt 代码：

```python
agent_prompt = """Answer the following questions as best you can. You have access to the following tools:

Search Order:
一个能够查询订单信息，获得最新的订单情况的工具，参数是输入订单id
Recommend product: 一个能够基于商品及用户
信息为用户进行商品推荐导购的工具，参数是输入要推荐的商品类型

The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: Search Order, Recommend product

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

\`\`\`
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
\`\`\`

ALWAYS use the following format:

Question: the input question you
must answer
Thought: you should always think about what to do
Action:
\`\`\`
$JSON_BLOB
\`\`\`
Observation:
the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the
final answer
Final Answer: the final answer to the original input question

Begin! Reminder to always use the exact characters `Final Answer` when responding.'
{question}
"""

chat = doubao.ChatSkylark(model="skylark-chat",temperature=0,top_p=1,top_k=1)
question="我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗"
messages = [
    HumanMessagePromptTemplate.from_template(
        template=agent_prompt,
    ).format(question=question),
]
result = chat(messages)
# agent的解析
text = result.content
pattern = re.compile(r"^.*?`{3}(?:json)?\n(.*?)`{3}.*?$", re.DOTALL) 
found = pattern.search(text)
action = found.group(1)
response = json.loads(action.strip())
print("问题: %s\n"%question)
print(response) #json解析后已经满足json格式
```

运行可以看到正常解析出了符合要求的json格式：
```
{'action': 'Recommend product', 'action_input': {'user_demographic': {'age': 25, 'gender': 'Male', 'location': 'New York'}, 'preferences': {'style': 'Casual', 'color': 'Blue'}}}
```

**将skylark放入langchain中测试agent**

编写工具函数：
```python
# 模拟电商订单
def search_order(input: str)->str:
    print("调用search_order：一个能够查询订单信息，获得最新的订单情况的工具:")
    return "{order}，订单状态：已发货".format(order=input)

# 模拟商品推荐
def recommend_product(input: str)->str:
    print("调用recommend_product：一个能够基于商品及用户信息为用户进行商品推荐导购的工具:")
    return "黑色连衣裙"
```

接入langchain的agent：
```python
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

chat = doubao.ChatSkylark(model="skylark-chat",temperature=0,top_p=0,top_k=1)
agent_tools = initialize_agent(tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
result = agent_tools.run("我想卖一件衣服，但不知道哪款适合我，有什么好推荐吗")
print(result)
```

查看结果：
```markdown
我需要找到一个工具来推荐适合我的衣服。根据给定的工具，我可以使用“Recommend product”来获得推荐。

Action: Recommend product
Action Input: 衣服类型
Observation: 调用recommend_product：一个能够基于商品及用户信息为用户进行商品推荐导购的工具

Observation:  黑色连衣裙
Thought: 根据推荐的结果，我选择了黑色连衣裙。

Final Answer: 黑色连衣裙
```