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

url = "http://127.0.0.1:8000/debug-ask"
data = {"question": "使用 es-knn 作为向量数据库",
        "history_messages": []}
response = requests.post(url, json=data)
print(response.json())
