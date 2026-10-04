"""Ollama 和 DeepSeek 的最小非流式适配；不保存密钥或推理过程。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    response_model: str
    finish_reason: str
    latency_seconds: float
    prompt_tokens: int
    output_tokens: int
    cache_hit_tokens: int | None = None
    system_fingerprint: str | None = None
    load_seconds: float | None = None
    generation_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMProvider(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult: ...


class ProviderError(RuntimeError):
    """对外只暴露安全错误摘要，不打印请求头或服务端响应正文。"""


class HTTPProvider:
    def __init__(self, client: httpx.Client) -> None:
        self.client = client

    def request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.client.request(method, url, **kwargs)
            if response.status_code != 200:
                raise ProviderError(f"LLM HTTP {response.status_code}；请检查服务、模型、权限和余额")
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("response")
            return data
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(f"LLM 请求失败：{type(error).__name__}（敏感详情已隐藏）") from None


class OllamaProvider(HTTPProvider):
    def __init__(self, spec: dict[str, Any], settings: dict[str, Any], *, client=None):
        host = os.environ.get(spec["host_env"]) or spec["default_host"]
        host = host if "://" in host else "http://" + host
        parsed = urlparse(host)
        # 这一实验明确比较本地模型；禁止把本地证据意外发到远端 Ollama。
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
            raise ValueError("Phase 6 Ollama 必须是本机回环地址")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama 地址不能包含凭据、查询或片段")
        if parsed.hostname == "0.0.0.0":
            host = host.replace("0.0.0.0", "127.0.0.1")
        super().__init__(client or httpx.Client(timeout=settings["timeout_seconds"], trust_env=False))
        self.host, self.spec, self.settings = host.rstrip("/"), spec, settings

    def identity(self) -> dict[str, Any]:
        tags = self.request("GET", self.host + "/api/tags")
        selected = next((m for m in tags["models"] if m["name"] == self.spec["model"]), None)
        if selected is None or selected["digest"] != self.spec["digest"]:
            raise ValueError("本地模型不存在或 digest 与配置不同；不要静默更换模型")
        info = self.request("POST", self.host + "/api/show", json={"model": self.spec["model"]})
        if "thinking" not in info.get("capabilities", []):
            raise ValueError("当前配置要求能显式关闭 thinking 的模型")
        return {
            "model": selected["name"], "digest": selected["digest"],
            "model_bytes": selected["size"], "details": info["details"],
            "template_sha256": fingerprint(info.get("template", "")),
            "ollama_version": self.request("GET", self.host + "/api/version")["version"],
        }

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        started = time.perf_counter()
        data = self.request("POST", self.host + "/api/chat", json={
            "model": self.spec["model"], "messages": messages,
            "stream": False, "think": False, "keep_alive": self.spec["keep_alive"],
            **({"truncate": False, "shift": False} if self.settings.get("strict_context") else {}),
            "options": {
                "temperature": self.settings["temperature"],
                "num_predict": self.settings["max_output_tokens"],
                "num_ctx": self.spec["num_ctx"], "seed": self.spec["seed"],
                "top_p": 1, "top_k": 0, "min_p": 0,
                "repeat_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
            },
        })
        if not data.get("done") or data["message"].get("thinking"):
            raise ProviderError("Ollama 返回未完成响应或未关闭 thinking")
        return GenerationResult(
            answer=data["message"]["content"], response_model=data["model"],
            finish_reason=data.get("done_reason", "unknown"),
            latency_seconds=time.perf_counter() - started,
            prompt_tokens=data["prompt_eval_count"], output_tokens=data["eval_count"],
            load_seconds=data.get("load_duration", 0) / 1e9,
            generation_seconds=data.get("eval_duration", 0) / 1e9,
        )


class DeepSeekProvider(HTTPProvider):
    def __init__(self, spec: dict[str, Any], settings: dict[str, Any], *, client=None):
        # 固定官方 HTTPS 地址，避免自定义 endpoint 将密钥发送给第三方。
        if spec["base_url"] != "https://api.deepseek.com":
            raise ValueError("DeepSeek 配置必须使用官方 HTTPS 地址")
        key = os.environ.get(spec["api_key_env"], "").strip()
        if not key:
            raise ProviderError(f"缺少环境变量 {spec['api_key_env']}；请在本机安全配置")
        super().__init__(client or httpx.Client(timeout=settings["timeout_seconds"]))
        self.headers = {"Authorization": "Bearer " + key}
        self.spec, self.settings = spec, settings

    def identity(self) -> dict[str, Any]:
        data = self.request("GET", self.spec["base_url"] + "/models", headers=self.headers)
        if self.spec["model"] not in {model["id"] for model in data["data"]}:
            raise ValueError("配置的 DeepSeek 模型未被 API 列出")
        return {
            "model": self.spec["model"], "documented_version": self.spec["documented_version"],
            "immutable_revision_available": False,
        }

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        started = time.perf_counter()
        data = self.request("POST", self.spec["base_url"] + "/chat/completions",
                            headers=self.headers, json={
            "model": self.spec["model"], "messages": messages, "stream": False,
            "thinking": {"type": "disabled"},
            "temperature": self.settings["temperature"],
            "max_tokens": self.settings["max_output_tokens"],
            **({"response_format": self.settings["response_format"]}
               if "response_format" in self.settings else {}),
        })
        choice, usage = data["choices"][0], data["usage"]
        if choice["message"].get("reasoning_content"):
            raise ProviderError("DeepSeek 未关闭 thinking")
        return GenerationResult(
            answer=choice["message"]["content"] or "", response_model=data["model"],
            finish_reason=choice["finish_reason"], latency_seconds=time.perf_counter() - started,
            prompt_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
            cache_hit_tokens=usage.get("prompt_cache_hit_tokens"),
            system_fingerprint=data.get("system_fingerprint"),
        )


def build_provider(spec: dict[str, Any], settings: dict[str, Any]) -> LLMProvider:
    classes = {"ollama": OllamaProvider, "deepseek": DeepSeekProvider}
    return classes[spec["kind"]](spec, settings)
