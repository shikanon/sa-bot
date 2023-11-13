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

class PythonInterpreter:

    def get_name(self) -> str:
        return "Python Interpreter"
    
    def get_func(self) -> Any:
        def func(input_text: str) -> str:
            print(input_text)
            code = ""
            if '```python' in input_text:
                pattern = re.compile(r"^.*?```(?:python)?(.*?)```.*?$", re.DOTALL)
                found = pattern.search(input_text)
                code = found.group(1)
            elif '```' in input_text:
                pattern = re.compile(r"^.*?`{3}(.*?)`{3}.*?$", re.DOTALL)
                found = pattern.search(input_text)
                code = found.group(1)
            elif '`' in input_text:
                pattern = re.compile(r"^.*?`(.*?)`.*?$", re.DOTALL)
                found = pattern.search(input_text)
                code = found.group(1)
            if code != "":
                try:
                    function_compile = compile(code,'<string>','exec')
                    global_variable = {}
                    exec(__source=function_compile,__globals=global_variable)
                    if isinstance(global_variable["Solution"],function):
                        result = global_variable["Solution"]()
                        return str(result)
                except Exception as e:
                    return "Help! Unexpected error: \n%s"%str(e)
            return "你的输出格式不符合要求，需要遵守以下规则，否则你无法使用 Python Interpreter 这个工具：\
                (1) 你的代码必须放在```{{code}}```当中 \
                (2) 你的代码必须使用 Solution 作为函数名称，可以参考我给你的样例"
        return func

    def get_descripte(self) -> str:
        description = """Python Interpreter 是一个用来执行 Python 代码的工具。你在使用他的时候必须给他输入一个函数字符串，
函数名必须得是`Solution`，代码对应的是你一步步的思考过程。

问题：求两数之和，给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值 target 的那两个整数，并返回它们的数组下标。输入：nums = [2,7,11,15], target = 9, 输出：

代码如下：
```python
def Solution():
    nums = [2,7,11,15]
    target = 9
    return twoSum(nums, target)
  
def twoSum(self, nums, target):
    dic = dict()
    for idx,num in enumerate(nums):
        if target - num in dic:
            return [dic[target - num],idx]        
        dic[num] = idx
```
"""
        return description


if __name__ == "__main__":
    python = PythonInterpreter()
    tools = [
        Tool(
            name=python.get_name(),
            func=python.get_func(),
            description=python.get_descripte()
        )
    ]

    chat = doubao.ChatSkylark(model="skylark-chat",temperature=0.01,top_k=1)
    print(chat)

    agent_tools = initialize_agent(
        tools=tools, llm=chat, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        agent_kwargs={"handle_parsing_errors": True}, verbose=True)
    question = """若 $z=-1+\sqrt{3}i$, 则 $\frac{z}{{z\overline{z}-1}}=\left(\ \ \right)$，用 Python 求解"""
    result = agent_tools.run(question)
    print(result)
    question = "哪个数字是第10个斐波那契数？"
    result = agent_tools.run(question)
    print(result)
    # from langchain.agents.agent_toolkits import create_python_agent
    # from langchain.tools.python.tool import PythonREPLTool
    # from langchain.python import PythonREPL
    # agent_executor = create_python_agent(
    #     llm=chat,
    #     tool=PythonREPLTool(),
    #     verbose=True,
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # )
    # result = agent_executor.run("What is the 10th fibonacci number?")
    # print(result)