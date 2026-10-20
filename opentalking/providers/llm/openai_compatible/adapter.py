from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx


class OpenAICompatibleLLMClient:
    """Async streaming client for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "qwen-turbo",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def chat_stream(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        if not self.base_url:
            raise RuntimeError("LLM base_url is not configured")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "stream": True}
        if self.model == "prune":
            qa_dict = {
                "你好": "你好呀，很高兴见到你",
                "hi": "你好呀，很高兴见到你",
                "今天天气怎么样": "我这边没法直接获取实时天气，您可以打开手机天气应用看一下。出门前留意温度和降雨提示，早晚温差大的话记得带件外套。",
                "天气如何":"我这边没法直接获取实时天气，您可以打开手机天气应用看一下。出门前留意温度和降雨提示，早晚温差大的话记得带件外套。", 
                "今天天气":"我这边没法直接获取实时天气，您可以打开手机天气应用看一下。出门前留意温度和降雨提示，早晚温差大的话记得带件外套。", 
                "外面天气怎么样":"我这边没法直接获取实时天气，您可以打开手机天气应用看一下。出门前留意温度和降雨提示，早晚温差大的话记得带件外套。", 
                "天气好吗":"我这边没法直接获取实时天气，您可以打开手机天气应用看一下。出门前留意温度和降雨提示，早晚温差大的话记得带件外套。",
                "周末安排": "如果想让身体放松，可以去公园走走或者爬个小山；如果想安静一点，在家看部电影、做顿饭也不错。关键是别把周末排得太满，留点休息时间。",
            }
            input_msg = messages[-1]["content"]
            if input_msg in qa_dict:
                content = qa_dict[input_msg]
            else:
                content = qa_dict["hi"]
            yield content
        else:
            timeout = httpx.Timeout(connect=60.0, read=120.0, write=30.0, pool=30.0)
            url = f"{self.base_url}/chat/completions"

            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    async for raw_line in response.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            return
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
