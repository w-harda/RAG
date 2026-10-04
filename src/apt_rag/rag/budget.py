"""真实 tokenizer 预检 + 输出预留；云端模板只能估算，usage 必须再次核验。"""

import hashlib
import json
from pathlib import Path


class ContextOverflow(ValueError):
    pass


def checked_file(path: Path, expected: str) -> Path:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Tokenizer 文件指纹不匹配")
    return path


class TokenBudget:
    def __init__(self, root, config, spec, output_tokens, *, provider=None):
        from tokenizers import Tokenizer

        self.config, self.spec, self.output_tokens, self.provider = config, spec, output_tokens, provider
        self.token_spec = config["tokenizers"][spec["id"]]
        self.native_verified = False
        if spec["kind"] == "ollama":
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(self.token_spec["model_name"], "tokenizer.json",
                                       revision=self.token_spec["revision"], local_files_only=True))
        else:
            path = root / self.token_spec["directory"] / "tokenizer.json"
            checked_file(path.with_name("tokenizer_config.json"), self.token_spec["config_sha256"])
        checked_file(path, self.token_spec["tokenizer_sha256"])
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.backend_json = json.loads(path.read_text(encoding="utf-8"))

    def measure(self, messages):
        if [m["role"] for m in messages] != ["system", "user"]:
            raise ValueError("本阶段只接受两条固定 system/user messages")
        if self.spec["kind"] == "ollama":
            if not self.native_verified:
                identity = self.provider.identity()
                if identity["ollama_version"] != self.config["ollama_version"]:
                    raise ValueError("Ollama 版本改变；需复核严格长度/模板接口")
                info = self.provider.request("POST", self.provider.host + "/api/show",
                                             json={"model": self.spec["model"], "verbose": True})["model_info"]
                vocab = self.tokenizer.get_vocab()
                native = info["tokenizer.ggml.tokens"]
                merges = [" ".join(m) if isinstance(m, list) else m for m in self.backend_json["model"]["merges"]]
                if (info["tokenizer.ggml.pre"] != "qwen35" or
                        any(index >= len(native) or native[index] != token for token, index in vocab.items()) or
                        merges != info["tokenizer.ggml.merges"] or
                        native[len(vocab):] != [f"[PAD{i}]" for i in range(len(native) - len(vocab))]):
                    raise ValueError("本地 GGUF tokenizer 与固定 Qwen tokenizer 不一致")
                self.native_verified = True
            # 版本固定、官方源码和真实服务验证过的 render-only 接口；不执行生成。
            data = self.provider.request("POST", self.provider.host + "/api/chat", json={
                "model": self.spec["model"], "messages": messages, "think": False, "stream": False,
                "truncate": False, "shift": False, "_debug_render_only": True,
                "keep_alive": self.spec["keep_alive"], "options": {"num_ctx": self.spec["num_ctx"]},
            })
            rendered = data.get("_debug_info", {}).get("rendered_template")
            if not isinstance(rendered, str) or any(m["content"] not in rendered for m in messages):
                raise ValueError("不能验证服务完整模板；禁止回退到字符估算")
            kind = "native_render_verified_vocab_token_count"
        else:
            # 官方演示包中两消息模板；不假装与云端实际模板/权重版本完全相同。
            rendered = ("<｜begin▁of▁sentence｜>" + messages[0]["content"] + "<｜User｜>"
                        + messages[1]["content"] + "<｜Assistant｜>")
            kind = "official_v4_demo_chat_estimate"
        count = len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)
        limit, margin = self.config["context_window_tokens"], self.config["token_safety_margin"]
        if count + margin + self.output_tokens > limit:
            raise ContextOverflow(f"Context 超限：输入 {count} + 安全预留 {margin} + 输出 {self.output_tokens} > {limit}；未截断")
        return {"input_tokens": count, "count_kind": kind, "context_window_tokens": limit,
                "safety_margin": margin, "max_output_tokens": self.output_tokens,
                "render_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                "tokenizer_sha256": self.token_spec["tokenizer_sha256"]}

    def verify_usage(self, result, budget):
        if (result.prompt_tokens < 1 or result.output_tokens < 0 or
                result.output_tokens > self.output_tokens or
                result.prompt_tokens > budget["input_tokens"] + budget["safety_margin"] or
                result.prompt_tokens + self.output_tokens > budget["context_window_tokens"]):
            raise ContextOverflow("服务实际 usage 超出预检/输出预算；不接受该答案")
