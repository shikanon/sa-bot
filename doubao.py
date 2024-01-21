
"""
doubao chat wrapper.
author: shikanon
create: 2023/10/2
"""
from __future__ import annotations
from langchain.pydantic_v1 import Field, root_validator
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

from retrying import retry
from volcengine.maas import MaasService, MaasException, ChatRole
from langchain.adapters.openai import convert_dict_to_message, convert_message_to_dict

from langchain.chat_models.base import BaseChatModel
from langchain.pydantic_v1 import Field, root_validator
from langchain.schema import ChatGeneration, ChatResult
from langchain.schema import AIMessage, HumanMessage, ChatMessage, SystemMessage
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from langchain.schema.output import ChatGenerationChunk
from langchain.utils import get_from_dict_or_env, get_pydantic_field_names

class ModelFunctionClass:
    def __init__(self, name=None, description=None, parameters=None):
        """
        name: 回调函数的名称，会用于做function call 的意图识别
        description: 回调函数的功能描述，和name一样用于做function call 的意图识别
        parameters: 字典，函数的参数，参数如下：
        "parameters": 
        {
            "properties": 
            {
                "query": {"description": "表示用戶输入查询实体", "type":"string"}},
                "required": ["query"],"type": "object",
            },
            # 调用调用示用示例示例
            "examples": ['{"query": "最近3天看过的文档"}'],
        }
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def todict(self) -> Dict:
        if self.name is None or self.description is None:
            raise ValueError("invaild vale, name and description cannot be null")
        result = {
            "name": self.name,
            "description": self.description,
        }
        if self.parameters is not None:
            result["parameters"] = self.parameters
        return result

class ChatSkylark(BaseChatModel):
    model_name: str = Field(default="skylark-chat", alias="model")
    model_version: Optional[str] = None 
    model_endpoint: Optional[str] = None
    """VOLC_ACCESSKEY"""
    model_ak: Optional[str] = None
    """VOLC_SECRETKEY"""
    model_sk: Optional[str] = None
    # 输出文本的最大tokens限制
    max_tokens: Optional[int] = None
    """temperature:用于控制生成文本的随机性和创造性，Temperature值越大随机性越大，取值范围0~1"""
    temperature: float = 0.7
    """top_p:用于控制输出tokens的多样性，TopP值越大输出的tokens类型越丰富，取值范围0~1"""
    top_p: float = 0.9
    """top_k:选择预测值最大的k个token进行采样，取值范围0-1000，0表示不生效"""
    top_k: Optional[int] = None
    maas: MaasService = MaasService('maas-api.ml-platform-cn-beijing.volces.com', 'cn-beijing')
    streaming: bool = False
    """plugins: 支持头条搜索插件，[browsing]"""
    plugins: Optional[str] = None
    """functions: 自定义第三方函数回调"""
    functions: Optional[List] = None

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "skylark-chat"
    
    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> Dict[str, Any]:
        messages_list = []
        for m in messages:
            if isinstance(m, SystemMessage):
                message_dict = {"role": ChatRole.SYSTEM, "content": m.content}
            elif isinstance(m, HumanMessage):
                message_dict = {"role": ChatRole.USER, "content": m.content}
            elif isinstance(m, AIMessage):
                message_dict = {"role": ChatRole.ASSISTANT, "content": m.content}
            else:
                message_dict = {"role": ChatRole.USER, "content": m.content}
            messages_list.append(message_dict)
        req = {
            "model": {
                "name": self.model_name,
            },
            "parameters": {
                # 输出文本的最大tokens限制
                "max_new_tokens": self.max_tokens,
                # 用于控制生成文本的随机性和创造性，Temperature值越大随机性越大，取值范围0~1
                "temperature": self.temperature,
                # 用于控制输出tokens的多样性，TopP值越大输出的tokens类型越丰富，取值范围0~1
                "top_p": self.top_p,
                # 选择预测值最大的k个token进行采样，取值范围0-1000，0表示不生效
                "top_k": self.top_k,
            },
            "messages": messages_list
        }
        if self.functions is not None and len(self.functions) >0 :
            req["functions"] = self.functions
        if self.model_version is not None:
            req["model"]["version"] = self.model_version
        if self.model_endpoint is not None:
            req["model"]["endpoint_id"] = self.model_endpoint
        return req

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        res = response["choice"]
        message = convert_dict_to_message(res["message"])
        gen = ChatGeneration(
            message=message,
            generation_info=dict(finish_reason=res.get("finish_reason")),
        )
        generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def completion_with_retry(
        self, run_manager: Optional[AsyncCallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:

        @retry(stop_max_attempt_number=3)
        def _completion_with_retry(**kwargs: Any) -> Any:
            return self.maas.chat(**kwargs)

        return _completion_with_retry(**kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        param = self._create_message_dicts(messages)
        for res in self.maas.stream_chat(param):
            if "choice" in res:
                content = res["choice"]["message"]["content"]
                messagechunk = AIMessageChunk(content=content,role="assistant")
                generation = ChatGenerationChunk(
                    text=content,
                    message=messagechunk,
                    generation_info={"finish_reason": "finished"},
                )
                yield generation

    async def _astream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        param = self._create_message_dicts(messages)
        for res in self.maas.stream_chat(param):
            if "choice" in res:
                content = res["choice"]["message"]["content"]
                messagechunk = AIMessageChunk(content=content,role="assistant")
                generation = ChatGenerationChunk(
                    text=content,
                    message=messagechunk,
                )
                yield generation

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(
                messages=messages, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        # 将统一的message转换为maas处理的参数
        params = self._create_message_dicts(messages)
        response = self.completion_with_retry(req=params)
        # 由于豆包还未实现stop命令，手动实现stop命令
        content = response.choice.message.content
        if stop:
            for s in stop:
                if s in content:
                    content = content.split(s)[0] + s
        response_dict = {
            "choice": {
                "message": {
                    "role": response.choice.message.role,
                    "content": content,
                    "function_call": response.choice.message.function_call,
                },
                "finish_reason": response.choice.finish_reason,
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        return self._create_chat_result(response_dict)
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            generation: Optional[ChatGenerationChunk] = None
            async for chunk in await self._astream(
                messages=messages, **kwargs
            ):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            yield ChatResult(generations=[generation])
        # 将统一的message转换为maas处理的参数
        params = self._create_message_dicts(messages)
        response = self.completion_with_retry(req=params)
        # 由于豆包还未实现stop命令，手动实现stop命令
        content = response.choice.message.content
        if stop:
            for s in stop:
                if s in content:
                    content = content.split(s)[0] + s
        response_dict = {
            "choice": {
                "message": {
                    "role": response.choice.message.role,
                    "content": content,
                    "function_call": response.choice.message.function_call,
                },
                "finish_reason": response.choice.finish_reason,
            },
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        yield self._create_chat_result(response_dict)