# coding:utf-8
import requests
import chatbot_test

url = "http://127.0.0.1:8000/ask"
data = {
    "question": "我的账号有问题需要立刻帮我转人工", 
    "history_messages": [("我的充值金额出现了问题，我充值了50块却只显示10块钱，快帮我看看","好的，请问您方便提供下您的账号和订单号吗？")]}
response = requests.post(url, json=data)
print(response.json())

url = "http://127.0.0.1:8000/uploadfile/"
files = {'file': open('Readme.md', 'rb')}
response = requests.post(url, files=files)
print(response.json())

url = "http://127.0.0.1:8000/save-faq/"
faq_id = "345"
data = {
    "faq_id": faq_id,
    "question": "大模型的评测体系有哪些",
    "answer": """360评测 - 横向进行跨学科、跨能力维度的评测，用于快速衡量模型是否具有广泛的世界知识和各类问题解决能力。
基础能力评测 - 为更专业解决某种场景的问题，模型需要在某些类别中体现更加突出的能力。因此方舟还提供不同侧重的，基于能力维度的模型评测选项。
语言创作 - 理解与生成文本的能力，与人类语言考试的读、写对应
推理数学 - 逻辑推理与数学计算，及延伸的对复杂规则的学习能力
知识能力 - 记忆与理解各行各业知识，如常识、生活、社会文化等""",
    "faq_type": "knowledge",
}
response = requests.post(url, json=data)
print(response.json())

response = requests.delete(f"http://127.0.0.1:8000/delete-faq-by-id/{faq_id}")
assert response.json() == {"status": "success"}

url = "http://127.0.0.1:8000/debug-ask"
data = {"question": "使用 es-knn 作为向量数据库",
        "history_messages": []}
response = requests.post(url, json=data)
print(response.json())
