#!/usr/bin/env python3
"""SST-2 指令学习评测：不训练模型，只靠一段指令让大模型判断影评情感，统计准确率。

对应导师材料《1-使用大模型进行指令学习》，有三处必要调整：
  1. 用验证集 validation（872 条，有标签）。GLUE 的 test 集标签是隐藏的（全是 -1），
     材料里 split="test[:20]" 只能看输出，算不出准确率。
  2. 标签 0/1 换成材料里的 -1/1（0 = 负面 -> -1，1 = 正面 -> 1）。
  3. 走聊天接口时末尾不加 <think>。聊天接口会自动套上模型自己的对话模板，
     思考与否用 --think 开关控制；材料里的 <think> 是直接续写原始文本时的写法。

三种后端：
  deepseek  DeepSeek 官方 API，走 OpenAI 接口（需要 pip install openai）
  ollama    本机 Ollama，走它的原生接口 /api/chat，可以开关思考（只用标准库）
  openai    其他 OpenAI 兼容服务：llama.cpp 的 llama-server、LM Studio、vLLM 等

常用命令（Windows PowerShell）：
  $env:DEEPSEEK_API_KEY = "sk-..."                    # 只对当前窗口有效
  python sst2_instruction_eval.py --demo              # 发材料里那句例句，看完整返回
  python sst2_instruction_eval.py --n 20              # 前 20 条
  python sst2_instruction_eval.py                     # 全部 872 条
  python sst2_instruction_eval.py --think on --n 20   # 打开思考模式
  python sst2_instruction_eval.py --provider ollama --model qwen3.5:4b --n 20
  python sst2_instruction_eval.py --provider ollama --model qwen3.5:0.8b qwen3.5:2b qwen3.5:4b
  python sst2_instruction_eval.py --report            # 把 results 里的结果汇成一张表

每组实验在 results 文件夹下生成三个文件：
  <名字>.jsonl         每条样本的完整返回（含思考过程），同时是断点：同样的命令重跑会接着跑
  <名字>.csv           每条一行：真实标签、预测、是否正确、解析方式、耗时、token 数、回答
  <名字>_summary.json  准确率、无效输出数、混淆矩阵、token 总数
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# ======================= 1. 指令模板（英文原文照抄材料） =======================
HEADER = ("Below is an instruction that describes a task, paired with an input that provides "
          "further context. Write a response that appropriately completes the request.")
THINK_STEP = ("Before answering, think carefully about the question and create a step-by-step "
              "chain of thoughts to ensure a logical and accurate response.")
TASK = ("Analyze the given text from an online review and determine the sentiment polarity. "
        "Return a single number of either -1 and 1, with -1 being negative and 1 being the "
        "positive sentiment.")

PROMPTS = {
    # 材料第 2 页发给 ChatGPT 的版本：只要一个数字，不要解释
    "basic": (HEADER + "\n\n### Instruction:\n"
              "Analyze the given text from an online review and determine the sentiment polarity. "
              "Return a single number of either -1 and 1, with 1 being positive and -1 being the "
              "negative sentiment.\nNo further explanation or justification is required.\n\n"
              "### Question:\n{sentence} What's the sentiment polarity of the text?\n\n"
              "### Response:"),
    # 材料第 5 页循环里的版本：要求先一步步思考
    "cot": (HEADER + "\n" + THINK_STEP + "\n\n### Instruction:\n" + TASK +
            "\n\n### Input:\n{sentence}\n\n### Response:"),
    # 材料第 4 页的版本：再加一句角色设定
    "role_cot": (HEADER + "\n" + THINK_STEP + "\n\n### Instruction:\n"
                 "You are an expert of consumer comment analysis. " + TASK +
                 "\n\n### Input:\n{sentence}\n\n### Response:"),
}

DEMO_SENTENCE = ("this film 's relationship to actual tension is the same as what christmas-tree "
                 "flocking in a spray can is to actual snow : a poor -- if durable -- imitation .")

PROVIDERS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash",
                 "workers": 8, "timeout": 180},
    "ollama": {"base_url": "http://localhost:11434", "model": "qwen3.5:4b",
               "workers": 1, "timeout": 600},
    "openai": {"base_url": "http://localhost:8080/v1", "model": None,
               "workers": 1, "timeout": 600},
}

# ======================= 2. 从回答里取出 -1 或 1 =======================
_ONLY_NUM = re.compile(r"^[+-]?1(?:\.0+)?$")
_BOXED = re.compile(r"\\boxed\{\s*([+-]?)1(?:\.0+)?\s*\}")
# 答案标记：answer / sentiment polarity / response 等，可带 value、score，
# 以及 “of the text”“for this review” 这类修饰（Qwen 和 DeepSeek 实际输出里都出现过）。
_ANSWER_MARKER = (r"(?:final\s+(?:answer|label|prediction)|answer|response|"
                  r"(?:sentiment|polarity)(?:\s+polarity)?(?:\s+(?:value|score))?|"
                  r"prediction|label|output|result|value|number|conclusion)"
                  r"(?:\s+(?:of|for|on|in|to)\s+(?:the|this)\s+(?:provided\s+|given\s+)?"
                  r"(?:text|review|sentence|input|scale|instruction|question|task))?")
# “is” “is:” “should be” “of” “:” “=” 都算明确作答，后面可以跟 “the number”
_LINK = (r"\s*(?:(?:is|should\s+be|would\s+be|of)\s*:?|:|=)\s*"
         r"(?:the\s+)?(?:single\s+)?(?:number\s+|value\s+)?")
_LABEL_WORD = r"(positive|negative|正面|积极|负面|消极)"
_NUM = r"([+-]?1(?:\.0+)?)"
# 数字后面括号里的标签词，可以带 sentiment，如 “-1 (Negative sentiment)”（Mistral 的写法）
_PAREN_WORD = r"\(\s*" + _LABEL_WORD + r"(?:\s+sentiment)?\s*\)"
# 数字后面跟一句同向的说明，如 “-1, indicating a negative sentiment”（Llama、Mistral 的写法）
_DEGREE = (r"(?:strongly|very|highly|mostly|overall|generally|clearly|predominantly|largely|"
           r"somewhat|slightly|mildly)")
_INDICATING = (r"\s*,\s*(?:indicating|which\s+(?:indicates|means))\s+(?:an?\s+)?(?:" + _DEGREE +
               r"\s+)?" + _LABEL_WORD + r"\s+sentiment")
_EXPLICIT_NUM = re.compile(
    r"\b" + _ANSWER_MARKER + _LINK + _NUM + r"(?:\s*" + _PAREN_WORD + r")?(?:" + _INDICATING + r")?"
    r"\s*[.!。]?\s*$", re.I)
_WORD_ONLY = re.compile(r"positive|negative|正面|积极|负面|消极", re.I)
_EXPLICIT_WORD = re.compile(
    r"\b" + _ANSWER_MARKER + _LINK + _LABEL_WORD + r"(?:\s*\(" + _NUM + r"\))?\s*[.!。]?\s*$", re.I)
# 末行只有数字再跟一个冒号，如 “1:” 或 “1: 1”（Qwen 模仿 “### Response:” 时会这样写）
_NUM_COLON = re.compile(r"^([+-]?1)(?:\.0+)?\s*:(?:\s*([+-]?1)(?:\.0+)?)?$")
# 末行是 “Return the number 1.” “Return `1`.” 或 “1 (Positive)” “-1 (Negative sentiment)”
_RETURN = re.compile(r"\breturn\s+(?:the\s+)?(?:single\s+)?(?:number\s+|value\s+)?" + _NUM + r"\s*[.!。]?\s*$", re.I)
_NUM_WORD = re.compile(r"^" + _NUM + r"\s*" + _PAREN_WORD + r"$", re.I)
_TAG_LINE = re.compile(r"</?[A-Za-z_][\w-]*>?")  # <final_answer> 这类单独成行的标签
_NEGATIVE_WORDS = ("negative", "负面", "消极")
# 数学负号、长短破折号、全角符号统一成普通的 - 和 1
_DASHES = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-", "\uff0d": "-", "\uff11": "1"})


def split_think(text: str) -> tuple[str, str]:
    """把 <think>...</think> 里的思考和最终回答分开（有的服务把思考混在正文里返回）。"""
    text = text or ""
    if "</think>" in text:
        thinking, answer = text.rsplit("</think>", 1)
        return thinking.replace("<think>", "").strip(), answer.strip()
    if "<think>" in text:  # 思考没写完就被截断了，没有回答
        before, thinking = text.split("<think>", 1)
        return thinking.strip(), before.strip()
    return "", text.strip()


def parse_label(text: str) -> tuple[int | None, str]:
    """返回 (预测, 解析方式)。预测为 None 表示回答里找不到 -1/1，按答错计。

    exact       回答只有一个数字
    first_line  第一行只有一个数字，后面是解释
    boxed       \\boxed{-1} 这种写法
    last_number 先推理后作答，末行数字或明确的答案标记
    word        单独的情感词，或明确的 sentiment/answer 标记

    不从任意推理文字中猜标签；只提到选项、否定词或含糊表达按无效输出计。
    """
    answer = split_think(text)[1].translate(_DASHES)
    if not answer:
        return None, "empty"
    boxed = list(_BOXED.finditer(answer))
    m = boxed[-1] if boxed else None
    if m and not answer[m.end():].strip(" \t\r\n.!。*`\"'$"):  # $ 是 LaTeX 公式的结尾，如 $\boxed{1}$
        return (-1 if m.group(1) == "-" else 1), "boxed"
    lines = [ln.replace("**", "").replace("`", "").strip(" \t*#>\"'") for ln in answer.splitlines()]
    lines = [ln for ln in lines if ln and not _TAG_LINE.fullmatch(ln)]
    first = lines[0].rstrip(" .。") if lines else ""
    if _ONLY_NUM.match(first):
        # 首行作答后又明确改成另一个标签，不能继续把首行当作最终答案。
        first_value = -1 if first.startswith("-") else 1
        for line in lines[1:]:
            value, _ = line_label(line.rstrip(" .。"))
            if value == "conflict" or (value is not None and value != first_value):
                return None, "ambiguous"
        return first_value, ("exact" if len(lines) == 1 else "first_line")
    value, method = line_label(lines[-1].rstrip(" .。") if lines else "")
    if value == "conflict":
        return None, "ambiguous"
    if value is not None:
        return value, method
    return None, "invalid"


def _sign(number: str) -> int:
    return -1 if number.startswith("-") else 1


def _word_value(word: str) -> int:
    return -1 if word.lower() in _NEGATIVE_WORDS else 1


def line_label(line: str) -> tuple[int | str | None, str]:
    """从一行里取明确的答案：(-1/1, 方式)；没有答案是 (None, "")；数字和括号里的词矛盾是 ("conflict", "")。"""
    if _ONLY_NUM.fullmatch(line):
        return _sign(line), "last_number"
    m = _NUM_COLON.match(line)
    if m:
        if m.group(2) is not None and _sign(m.group(2)) != _sign(m.group(1)):
            return "conflict", ""
        return _sign(m.group(1)), "last_number"
    m = _EXPLICIT_NUM.search(line) or _NUM_WORD.match(line)
    if m:
        if any(word is not None and _word_value(word) != _sign(m.group(1)) for word in m.groups()[1:]):
            return "conflict", ""
        return _sign(m.group(1)), "last_number"
    m = _RETURN.search(line)
    if m:
        return _sign(m.group(1)), "last_number"
    m = _WORD_ONLY.fullmatch(line)
    if m:
        return _word_value(m.group(0)), "word"
    m = _EXPLICIT_WORD.search(line)
    if m:
        if m.group(2) is not None and _sign(m.group(2)) != _word_value(m.group(1)):
            return "conflict", ""
        return _word_value(m.group(1)), "word"
    return None, ""


# ======================= 3. 读数据 =======================
def load_rows(args) -> tuple[list[dict], str]:
    """返回 [{"idx", "sentence", "label"(-1/1)}, ...] 和数据来源说明。"""
    if args.data_file:
        path = Path(args.data_file)
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
        raw = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not {"sentence", "label"}.issubset(reader.fieldnames or []):
                sys.exit(f"{path} 必须包含 sentence 和 label 两列")
            for i, r in enumerate(reader):
                lab = (r.get("label") or "").strip()
                if not re.fullmatch(r"[+-]?\d+", lab) or int(lab) not in (-1, 0, 1):
                    sys.exit(f"{path} 第 {reader.line_num} 行的 label 必须是整数 0/1 或 -1/1，读到 {lab!r}")
                sentence = r.get("sentence")
                if sentence is None or not sentence.strip():
                    sys.exit(f"{path} 第 {reader.line_num} 行的 sentence 不能为空")
                raw.append((i, sentence, int(lab)))
        values = {lab for _, _, lab in raw}
        if 0 in values and -1 in values:
            sys.exit(f"{path} 混用了 0/1 和 -1/1 标签，请统一使用其中一种格式")
        zero_one = 0 in values  # 0/1 格式（GLUE 原始 tsv）要换成 -1/1；本来就是 -1/1 的不动
        rows = [{"idx": i, "sentence": s.strip(),
                 "label": (-1 if lab == 0 else 1) if zero_one else lab}
                for i, s, lab in raw]
        source = str(path)
    else:
        if args.hf_mirror:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            from datasets import load_dataset
        except ImportError:
            sys.exit("缺少 datasets 库：pip install datasets（或者用 --data-file 指定本地 tsv）")
        try:
            ds = load_dataset("nyu-mll/glue", "sst2", split=args.split)
        except Exception as e:  # 网络问题最常见
            sys.exit(f"下载 SST-2 失败：{e}\n国内网络可以加 --hf-mirror（走 hf-mirror.com 镜像），"
                     "或者下载 GLUE 的 SST-2/dev.tsv 后用 --data-file 指定。")
        rows = [{"idx": int(r["idx"]), "sentence": r["sentence"].strip(),
                 "label": -1 if r["label"] == 0 else 1}
                for r in ds if r["label"] != -1]  # -1 = 隐藏标签
        if not rows:
            sys.exit(f"{args.split} 集没有标签（GLUE 的 test 标签是隐藏的），请用 --split validation")
        source = f"GLUE SST-2 {args.split}"
    if args.n > 0:
        rows = rows[:args.n]
    return rows, source


# ======================= 4. 三种后端 =======================
class FatalError(Exception):
    """配置类错误（key 错、余额不足、模型不存在、服务没开……），重试没有意义，直接停。"""


STATUS_HINTS = {
    401: "API key 不对或没生效",
    402: "余额不足，去 platform.deepseek.com 充值",
    403: "没有访问权限",
    404: "地址或模型名不对，检查 --base-url 和 --model",
    422: "请求参数不被接受，检查 --model、--max-tokens 等",
}
_warned: set[str] = set()
_warn_lock = threading.Lock()


def warn_once(msg: str) -> None:
    with _warn_lock:
        if msg not in _warned:
            _warned.add(msg)
            print(f"[提示] {msg}", flush=True)


class OpenAIClient:
    """DeepSeek 和其他 OpenAI 兼容服务：用官方 openai 库发请求。"""

    def __init__(self, provider, base_url, api_key, model, think, max_tokens, temperature, timeout):
        try:
            import openai
        except ImportError:
            sys.exit("缺少 openai 库：pip install openai")
        self.openai = openai
        # max_retries：遇到限流(429)、服务器错误(5xx)、断网时，库会自己等一会儿再试
        local_http = {}
        if urllib.parse.urlparse(base_url).hostname in ("localhost", "127.0.0.1", "::1"):
            local_http["http_client"] = openai.DefaultHttpxClient(trust_env=False)
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout,
                                    max_retries=4, **local_http)
        self.provider, self.model, self.think = provider, model, think
        self.max_tokens, self.temperature = max_tokens, temperature
        self.send_think = think != "default"

    def _think_body(self) -> dict | None:
        if not self.send_think:
            return None
        on = self.think == "on"
        if self.provider == "deepseek":  # DeepSeek 默认开思考，要关必须明确写 disabled
            return {"thinking": {"type": "enabled" if on else "disabled"}}
        return {"chat_template_kwargs": {"enable_thinking": on}}  # llama-server / vLLM 的 Qwen 开关

    def _create(self, kwargs: dict):
        try:
            return self.client.chat.completions.create(**kwargs)
        except self.openai.APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code in STATUS_HINTS:
                raise FatalError(f"HTTP {code}，{STATUS_HINTS[code]}。原始信息：{str(e)[:300]}") from e
            raise

    def __call__(self, prompt: str) -> dict:
        kwargs = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": self.max_tokens, "temperature": self.temperature}
        extra = self._think_body()
        if extra:
            kwargs["extra_body"] = extra  # openai 库会把它原样并进请求体
        try:
            resp = self._create(kwargs)
        except self.openai.BadRequestError:
            if not (extra and self.provider == "openai"):
                raise
            kwargs.pop("extra_body")
            resp = self._create(kwargs)  # 有的服务不认识思考开关：去掉再试一次
            self.send_think = False
            warn_once("这个服务不接受思考开关 chat_template_kwargs，已去掉，按模型默认方式运行")
        choice = resp.choices[0]
        msg = choice.message
        # DeepSeek 把思考放在 reasoning_content；有的兼容服务叫 reasoning
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        usage = getattr(resp, "usage", None)
        return {"content": msg.content or "", "reasoning": reasoning,
                "finish": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None)}


class OllamaClient:
    """本机 Ollama 的原生接口 /api/chat。用 "think": true/false 开关思考。"""

    def __init__(self, base_url, model, think, max_tokens, temperature, timeout):
        self.url = base_url.rstrip("/") + "/api/chat"
        host = urllib.parse.urlparse(self.url).hostname or ""
        # 访问本机时不走系统代理（开着代理软件时常见的连不上问题）
        handlers = [urllib.request.ProxyHandler({})] if host in ("localhost", "127.0.0.1", "::1") else []
        self.opener = urllib.request.build_opener(*handlers)
        self.model, self.think, self.timeout = model, think, timeout
        self.options = {"temperature": temperature, "num_predict": max_tokens}
        self.send_think = think != "default"

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def __call__(self, prompt: str) -> dict:
        for attempt in range(5):
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                    "stream": False, "options": self.options}
            if self.send_think:
                body["think"] = self.think == "on"
            try:
                data = self._post(body)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                if e.code == 400 and "think" in detail.lower() and self.send_think:
                    self.send_think = False  # 这个模型不支持思考开关：去掉再试
                    warn_once(f"{self.model} 不支持思考开关，已去掉 think 参数（{detail}）")
                    continue
                if e.code == 404:
                    raise FatalError(f"Ollama 里没有模型 {self.model}，先运行：ollama pull {self.model}"
                                     f"（{detail}）") from e
                if e.code >= 500 and attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            except urllib.error.URLError as e:
                if isinstance(e.reason, ConnectionRefusedError) or "refused" in str(e.reason).lower():
                    raise FatalError(f"连不上 {self.url}：Ollama 没在运行？打开 Ollama 应用，"
                                     "或在另一个窗口运行 ollama serve") from e
                if attempt < 3:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (socket.timeout, TimeoutError):
                if attempt < 3:
                    continue
                raise
        else:
            raise RuntimeError("重试多次仍然失败")
        msg = data.get("message") or {}
        return {"content": msg.get("content") or "", "reasoning": msg.get("thinking") or "",
                "finish": data.get("done_reason"),
                "prompt_tokens": data.get("prompt_eval_count"),
                "completion_tokens": data.get("eval_count")}


def resolve_key(args, provider: str) -> str | None:
    if provider == "ollama":
        return None
    if args.api_key:
        return args.api_key
    if provider == "openai":  # 本地服务一般不校验 key；用别家服务时设 LLM_API_KEY
        return os.environ.get("LLM_API_KEY") or "none"
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        sys.exit("没找到 DeepSeek 的 API key。PowerShell 里先运行：\n"
                 '  $env:DEEPSEEK_API_KEY = "sk-你的key"\n'
                 "（cmd 用 set DEEPSEEK_API_KEY=sk-你的key；想长期保存用 setx，之后要开新窗口）")
    return key


def make_client(args, provider, model, think, max_tokens, temperature):
    cfg = PROVIDERS[provider]
    base_url = args.base_url or cfg["base_url"]
    timeout = args.timeout or cfg["timeout"]
    if provider == "ollama":
        return OllamaClient(base_url, model, think, max_tokens, temperature, timeout)
    return OpenAIClient(provider, base_url, resolve_key(args, provider), model, think,
                        max_tokens, temperature, timeout)


# ======================= 5. 跑一组实验 =======================
def auto_max_tokens(prompt_name: str, think: str) -> int:
    if think != "off":
        return 4096  # 思考过程也算输出 token，留足空间
    return 64 if prompt_name == "basic" else 1024  # cot 模板会在正文里写推理步骤


def auto_temperature(think: str) -> float:
    return 0.0 if think == "off" else 0.6  # 思考模式下 0 容易原地打转（DeepSeek 思考模式会忽略它）


def run_name(provider: str, model: str, prompt_name: str, think: str, n: int) -> str:
    raw = f"{provider}_{model}_{prompt_name}_think-{think}_{'all' if n <= 0 else f'n{n}'}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)


def experiment_fingerprint(provider: str, model: str, prompt_name: str, think: str,
                           base_url: str, max_tokens: int, temperature: float,
                           rows: list[dict]) -> tuple[str, str]:
    """设置与有序样本共同标识实验，避免旧预测混进新配置或新标签中。"""
    def digest(value) -> str:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    data_hash = digest([[r["idx"], r["sentence"], r["label"]] for r in rows])
    config = {"schema": 2, "provider": provider, "model": model, "prompt_name": prompt_name,
              "prompt_text": PROMPTS[prompt_name], "think": think,
              "base_url": base_url.rstrip("/"), "max_tokens": max_tokens,
              "temperature": temperature, "data_fingerprint": data_hash}
    return digest(config), data_hash


def load_checkpoint(path: Path) -> dict[int, dict]:
    done: dict[int, dict] = {}
    if path.exists():
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            if not line.strip():
                offset += len(line)
                continue
            try:
                rec = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                if index != len(lines) - 1:
                    raise ValueError(f"断点文件 {path} 第 {index + 1} 行损坏，请保留原文件并检查") from e
                # 上次中断时只写了半行。先移除坏尾部，再追加新记录，防止两行粘在一起。
                path.write_bytes(raw[:offset])
                warn_once(f"已修复 {path.name} 末尾未写完的记录，该条会重新请求")
                raw = raw[:offset]
                break
            if not isinstance(rec, dict) or not isinstance(rec.get("idx"), int):
                raise ValueError(f"断点文件 {path} 第 {index + 1} 行不是有效的样本记录")
            if rec.get("error") is None:  # 失败的条目重跑时再试
                done[rec["idx"]] = rec
            offset += len(line)
        if raw and not raw.endswith(b"\n"):
            # 即使末条 JSON 是完整的，缺换行也会让下次追加损坏这条与下一条。
            with path.open("ab") as f:
                f.write(b"\n")
    return done


def show_record(rec: dict) -> None:
    print(f"\n    --- 样本 idx={rec['idx']}，真实标签 {rec['label']}")
    print(f"    句子：{rec['sentence']}")
    if rec["reasoning"]:
        r = " ".join(rec["reasoning"].split())
        print(f"    思考：{r[:300]}{'……' if len(r) > 300 else ''}")
    if rec["error"]:
        print(f"    出错：{rec['error']}")
    else:
        answer = " ".join(split_think(rec["content"])[1].split())
        print(f"    回答：{answer[:300]}{'……' if len(answer) > 300 else ''}")
        print(f"    解析：{rec['pred']}（{rec['parse']}），{'正确' if rec['correct'] else '错误'}", flush=True)


def summarize(recs: list[dict], meta: dict) -> dict:
    n = len(recs)
    correct = sum(r["correct"] for r in recs)
    invalid = sum(r["pred"] is None for r in recs)
    confusion = {}
    for lab in (-1, 1):
        sub = [r for r in recs if r["label"] == lab]
        confusion[f"label={lab}"] = {"pred=-1": sum(r["pred"] == -1 for r in sub),
                                     "pred=1": sum(r["pred"] == 1 for r in sub),
                                     "invalid": sum(r["pred"] is None for r in sub)}
    return {**meta, "n": n,
            "accuracy": round(correct / n, 4) if n else None,
            "correct": correct, "invalid": invalid,
            "accuracy_on_valid": round(correct / (n - invalid), 4) if n > invalid else None,
            "errors": sum(r["error"] is not None for r in recs),
            "truncated": sum(r.get("finish") == "length" for r in recs),
            "parse_counts": dict(Counter(r["parse"] for r in recs)),
            "confusion": confusion,
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in recs),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in recs),
            "reasoning_chars": sum(len(r["reasoning"] or "") for r in recs),
            "avg_seconds": round(sum(r["seconds"] for r in recs) / n, 3) if n else None}


def write_csv(path: Path, recs: list[dict]) -> None:
    cols = ["idx", "label", "pred", "correct", "parse", "finish", "seconds", "prompt_tokens",
            "completion_tokens", "reasoning_chars", "answer", "error", "sentence"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig：Excel 直接打开不乱码
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            answer = " ".join(split_think(r["content"])[1].split())
            w.writerow([r["idx"], r["label"], "" if r["pred"] is None else r["pred"], int(r["correct"]),
                        r["parse"], r["finish"] or "", r["seconds"], r["prompt_tokens"] or "",
                        r["completion_tokens"] or "", len(r["reasoning"] or ""), answer[:500],
                        r["error"] or "", r["sentence"]])


def print_summary(s: dict, csv_path: Path) -> None:
    if not s["n"]:
        print("    没有结果")
        return
    c = s["confusion"]
    print(f"    准确率 {s['accuracy']:.4f}（{s['correct']}/{s['n']}），无效输出 {s['invalid']}，"
          f"请求失败 {s['errors']}，被截断 {s['truncated']}")
    print(f"    真实 -1：预测 -1 {c['label=-1']['pred=-1']} 条，预测 1 {c['label=-1']['pred=1']} 条，"
          f"无效 {c['label=-1']['invalid']} 条")
    print(f"    真实  1：预测 -1 {c['label=1']['pred=-1']} 条，预测 1 {c['label=1']['pred=1']} 条，"
          f"无效 {c['label=1']['invalid']} 条")
    print(f"    tokens：输入 {s['prompt_tokens']:,}，输出 {s['completion_tokens']:,}（含思考）；"
          f"单条请求平均 {s['avg_seconds']:.2f} 秒")
    print(f"    明细：{csv_path}")
    if s["think"] == "off" and s["reasoning_chars"] > 0:
        print("    [注意] 设了 --think off，返回里仍有思考内容：这个模型或服务可能关不掉思考")
    if s["think"] == "on" and s["reasoning_chars"] == 0:
        print("    [注意] 设了 --think on，却没收到思考内容：这个模型可能不支持思考")
    if s["truncated"]:
        print(f"    [注意] {s['truncated']} 条写到 max_tokens 上限被截断，可加大 --max-tokens 并加 --fresh 重跑")
    if s["errors"]:
        print(f"    [注意] {s['errors']} 条请求失败，原命令再跑一次只会重试这些")
    if s["invalid"] > s["errors"]:
        print(f"    [注意] {s['invalid'] - s['errors']} 条回答里没找到 -1/1，CSV 的 answer 列可以看原文")


def run_experiment(args, rows: list[dict], source: str, provider: str, model: str,
                   prompt_name: str, think: str) -> None:
    max_tokens = args.max_tokens or auto_max_tokens(prompt_name, think)
    temperature = args.temperature if args.temperature is not None else auto_temperature(think)
    workers = args.workers or PROVIDERS[provider]["workers"]
    base_url = args.base_url or PROVIDERS[provider]["base_url"]
    fingerprint, data_fingerprint = experiment_fingerprint(
        provider, model, prompt_name, think, base_url, max_tokens, temperature, rows)
    name = run_name(provider, model, prompt_name, think, args.n) + "_" + fingerprint[:12]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{name}.jsonl"
    if args.fresh and log_path.exists():
        log_path.unlink()
    row_of = {r["idx"]: r for r in rows}
    results = {i: rec for i, rec in load_checkpoint(log_path).items()
               if i in row_of and row_of[i]["sentence"] == rec.get("sentence")
               and row_of[i]["label"] == rec.get("label")
               and rec.get("fingerprint") == fingerprint}
    todo = [r for r in rows if r["idx"] not in results]
    csv_path = out_dir / f"{name}.csv"
    summary_path = out_dir / f"{name}_summary.json"
    if args.fresh or todo:
        # 汇总只表示完整覆盖当前样本的实验。重跑被中断时不能留下上次的旧汇总。
        csv_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    resumed = len(rows) - len(todo)
    print(f"\n=== {name}")
    print(f"    共 {len(rows)} 条，断点里已有 {resumed} 条；"
          f"并发 {workers}，max_tokens {max_tokens}，temperature {temperature}", flush=True)
    client = make_client(args, provider, model, think, max_tokens, temperature)
    template = PROMPTS[prompt_name]

    def work(row: dict) -> dict:
        t0 = time.time()
        out, error = {}, None
        try:
            out = client(template.format(sentence=row["sentence"]))
        except FatalError:
            raise
        except Exception as e:  # 单条失败不影响别的条，记下来，重跑时再试
            error = f"{type(e).__name__}: {e}"[:500]
        content = out.get("content", "")
        pred, how = parse_label(content) if error is None else (None, "error")
        return {"idx": row["idx"], "label": row["label"], "pred": pred,
                "fingerprint": fingerprint,
                "correct": pred == row["label"], "parse": how, "finish": out.get("finish"),
                "seconds": round(time.time() - t0, 2),
                "prompt_tokens": out.get("prompt_tokens"),
                "completion_tokens": out.get("completion_tokens"),
                "reasoning": out.get("reasoning") or split_think(content)[0],
                "content": content, "error": error, "sentence": row["sentence"]}

    t_start = last_print = time.time()
    new = new_errors = shown = printed = 0
    step = max(5, len(todo) // 10)  # 大约打印 10 次进度；慢的时候至少每 30 秒一次
    pool = ThreadPoolExecutor(max_workers=workers)
    pending = {pool.submit(work, r) for r in todo}
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            while pending:
                # 带超时地等，Windows 上按 Ctrl+C 才能及时停下
                finished, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in finished:
                    rec = fut.result()  # FatalError 会从这里抛出
                    log.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log.flush()
                    results[rec["idx"]] = rec
                    new += 1
                    new_errors += rec["error"] is not None
                    if shown < args.show:
                        shown += 1
                        show_record(rec)
                    if resumed == 0 and new == min(5, len(todo)) and new_errors == new:
                        raise FatalError(f"前 {new} 条全部失败，先检查配置。第一条错误：{rec['error']}")
                if finished and (new - printed >= step or not pending or time.time() - last_print > 30):
                    if printed == 0 and shown:
                        print()
                    printed, last_print = new, time.time()
                    done = list(results.values())
                    acc = sum(r["correct"] for r in done) / len(done)
                    per = (time.time() - t_start) / new
                    eta = f"  还要约 {(len(todo) - new) * per / 60:.1f} 分钟" if pending else ""
                    print(f"    [{len(done):>4}/{len(rows)}] 准确率 {acc:.4f}  "
                          f"无效 {sum(r['pred'] is None for r in done)}  {per:.2f} 秒/条{eta}", flush=True)
    except FatalError as e:
        print(f"\n停止：{e}", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(1)
    except KeyboardInterrupt:
        print(f"\n已中断。完成的 {len(results)} 条已存进 {log_path}，同样的命令再跑会接着跑。", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
        os._exit(130)
    pool.shutdown()

    recs = [results[r["idx"]] for r in rows if r["idx"] in results]
    meta = {"run": name, "provider": provider, "model": model,
            "base_url": base_url, "fingerprint": fingerprint,
            "data_fingerprint": data_fingerprint, "target_n": len(rows),
            "completed": len(recs) == len(rows),
            "prompt": prompt_name, "think": think, "data": source,
            "max_tokens": max_tokens, "temperature": temperature,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    summary = summarize(recs, meta)
    write_csv(csv_path, recs)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary, csv_path)


# ======================= 6. 汇总表、试发一条、命令行 =======================
def report(out_dir: str) -> None:
    files = sorted(Path(out_dir).glob("*_summary.json"))
    if not files:
        print(f"{out_dir} 里还没有结果")
        if Path(out_dir).exists():
            (Path(out_dir) / "report.md").write_text("当前没有可汇总的已完成实验。\n", encoding="utf-8")
        return
    rows = []
    for p in files:
        summary = json.loads(p.read_text(encoding="utf-8"))
        log_path = p.with_name(p.name.removesuffix("_summary.json") + ".jsonl")
        if not summary.get("completed", True) or (log_path.exists()
                and log_path.stat().st_mtime_ns > p.stat().st_mtime_ns):
            print(f"[提示] 跳过尚未完成或已过期的汇总 {p.name}，原命令重跑后会更新")
            continue
        rows.append(summary)
    rows.sort(key=lambda s: (s["provider"], s["model"], s["prompt"], s["think"], s["n"]))
    lines = ["| 后端 | 模型 | 模板 | 思考 | 数据 | 生成设置 | 样本数 | 准确率 | 无效 | 平均秒/条 | 输出 tokens | 解析统计 |",
             "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for s in rows:
        acc = "—" if s["accuracy"] is None else f"{s['accuracy']:.4f}"
        sec = "—" if s["avg_seconds"] is None else f"{s['avg_seconds']:.2f}"
        counts = ", ".join(f"{key}={value}" for key, value in sorted(s.get("parse_counts", {}).items()))
        data = str(s.get("data", "—")).replace("|", "\\|").replace("\n", " ")
        config = f"temperature={s.get('temperature', '—')}, max_tokens={s.get('max_tokens', '—')}"
        lines.append(f"| {s['provider']} | {s['model']} | {s['prompt']} | {s['think']} | {data} | {config} | {s['n']} | "
                     f"{acc} | {s['invalid']} | {sec} | {s['completion_tokens']:,} | {counts} |")
    text = "\n".join(lines) + "\n"
    out = Path(out_dir) / "report.md"
    out.write_text(text, encoding="utf-8")
    print(text + f"\n已保存到 {out}")


def demo(args, provider: str, model: str, prompt_name: str, think: str) -> None:
    max_tokens = args.max_tokens or auto_max_tokens(prompt_name, think)
    temperature = args.temperature if args.temperature is not None else auto_temperature(think)
    client = make_client(args, provider, model, think, max_tokens, temperature)
    prompt = PROMPTS[prompt_name].format(sentence=DEMO_SENTENCE)
    print("=" * 70)
    print(f"{provider} / {model} / 模板 {prompt_name} / 思考 {think}")
    print("----- 发出去的指令 -----")
    print(prompt)
    t0 = time.time()
    try:
        out = client(prompt)
    except FatalError as e:
        sys.exit(f"停止：{e}")
    seconds = time.time() - t0
    reasoning = out["reasoning"] or split_think(out["content"])[0]
    if reasoning:
        print("----- 思考过程 -----")
        print(reasoning.strip())
    print("----- 模型的回答 -----")
    print(split_think(out["content"])[1])
    pred, how = parse_label(out["content"])
    print(f"----- 解析出 {pred}（{how}），正确答案是 -1；用时 {seconds:.1f} 秒，"
          f"输入 {out['prompt_tokens']} / 输出 {out['completion_tokens']} tokens\n", flush=True)


def main() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")  # 模型回答里有表情等特殊字符时不至于报错
    except Exception:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=list(PROVIDERS), default="deepseek", help="后端，默认 deepseek")
    p.add_argument("--model", nargs="+", help="模型名，可以写多个依次跑；默认 deepseek-flash / qwen3.5:4b")
    p.add_argument("--prompt", nargs="+", choices=list(PROMPTS), default=["basic"],
                   help="指令模板，可以写多个；默认 basic")
    p.add_argument("--think", nargs="+", choices=["off", "on", "default"], default=["off"],
                   help="思考模式：off 关 / on 开 / default 不传开关、用模型默认；默认 off")
    p.add_argument("--n", type=int, default=0, help="只跑前 n 条；0 表示全部（验证集 872 条）")
    p.add_argument("--split", default="validation", help="数据划分，默认 validation")
    p.add_argument("--data-file", help="本地数据文件（tsv/csv，含 sentence 和 label 两列）")
    p.add_argument("--hf-mirror", action="store_true", help="用 hf-mirror.com 镜像下载数据集")
    p.add_argument("--base-url", help="服务地址，一般不用改")
    p.add_argument("--api-key", help="API key；建议改用环境变量 DEEPSEEK_API_KEY")
    p.add_argument("--max-tokens", type=int, help="最多生成多少 token；默认按模板和思考开关自动选")
    p.add_argument("--temperature", type=float, help="默认：关思考 0，开思考 0.6")
    p.add_argument("--workers", type=int, help="同时发几个请求；默认 deepseek 8，本地 1")
    p.add_argument("--timeout", type=float, help="单个请求最多等几秒")
    p.add_argument("--show", type=int, default=3, help="完整打印前几条的返回，默认 3")
    p.add_argument("--out-dir", default="results", help="结果文件夹，默认 results")
    p.add_argument("--fresh", action="store_true", help="不用断点，从头重跑")
    p.add_argument("--demo", action="store_true", help="只发材料里的那句例句，看完整返回")
    p.add_argument("--dry-run", action="store_true", help="只检查数据和指令，不发请求、不花钱")
    p.add_argument("--report", action="store_true", help="把结果文件夹里的所有实验汇成一张表")
    args = p.parse_args()

    if args.report:
        report(args.out_dir)
        return
    provider = args.provider
    models = args.model or ([PROVIDERS[provider]["model"]] if PROVIDERS[provider]["model"] else None)
    if not models:
        sys.exit("--provider openai 需要用 --model 写模型名（llama-server 随便写一个即可，"
                 "LM Studio 写界面上显示的 API identifier）")
    combos = [(m, pr, th) for m in models for pr in args.prompt for th in args.think]

    if args.demo:
        for m, pr, th in combos:
            demo(args, provider, m, pr, th)
        return

    if not args.dry_run:
        resolve_key(args, provider)  # 先检查 key，免得下载完数据才报错
    rows, source = load_rows(args)
    if not rows:
        sys.exit("没有读到数据")
    if args.dry_run:
        pos = sum(r["label"] == 1 for r in rows)
        print(f"数据：{source}，{len(rows)} 条（正面 1：{pos} 条，负面 -1：{len(rows) - pos} 条）")
        for m, pr, th in combos:
            print(f"  将运行 {run_name(provider, m, pr, th, args.n)}，"
                  f"max_tokens {args.max_tokens or auto_max_tokens(pr, th)}")
        print(f"\n第一条样本（真实标签 {rows[0]['label']}）发出去的完整指令：\n")
        print(PROMPTS[combos[0][1]].format(sentence=rows[0]["sentence"]))
        if provider == "deepseek":
            ok = bool(args.api_key or os.environ.get("DEEPSEEK_API_KEY"))
            print(f"\nDEEPSEEK_API_KEY：{'已设置' if ok else '未设置'}")
        return

    for m, pr, th in combos:
        run_experiment(args, rows, source, provider, m, pr, th)
    if len(combos) > 1:
        print()
        report(args.out_dir)


if __name__ == "__main__":
    main()
