"""Use a locally hosted model.

Expects an OpenAI-compatible API server to be running on port 8000, e.g. launched with:

```
vllm serve /path/to/huggingface/model
```
"""

import json
import os
import random
import re
from collections.abc import Collection, Sequence

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, get_text_content_as_str, text_content_block_from_string


class InvalidModelOutputError(Exception): ...


def reformat_message(message: ChatCompletionMessageParam):
    if message["role"] == "user" or message["role"] == "assistant":
        content = ""
        if "content" in message and message["content"] is not None:
            for message_content in message["content"]:
                if isinstance(message_content, str):
                    content += message_content
                elif "content" in message_content:
                    content += message_content["content"]
                else:
                    content += str(message_content)
                content += "\n\n"
            content = content.strip()
    else:
        content = message["content"]
    return content


def chat_completion_request(
    client: openai.OpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    temperature: float | None = 1.0,
    top_p: float | None = 0.9,
) -> str:
    try:
        reformatted_messages = []
        for message in messages:
            content = reformat_message(message)
            reformatted_messages.append({"role": message["role"], "content": content})
        extra_body = {}
        if os.getenv("LOCAL_LLM_DISABLE_THINKING", "0") == "1" or "qwen3" in model.lower():
            extra_body["think"] = False
        response = (
            client.chat.completions.create(
                model=model,
                messages=reformatted_messages,
                temperature=temperature,
                top_p=top_p,
                seed=random.randint(0, 1000000),
                extra_body=extra_body or None,
            )
            .choices[0]
            .message.content
        )
    except Exception as e:
        print(f"[debug] error: {e}")
        response = ""
    if response is None:
        raise InvalidModelOutputError("No response from model")
    return response


_tool_calling_prompt = """# Instructions
You are a helpful assistant. You will be given a task and have access to a set of possible functions which you may call in order to generate a final answer to the question. Functions must be called one at a time, but you may continue to call additional functions if you need to before providing your final answer.

## Currently Available Functions
{functions}

## Function Calling
You may output your reasoning prior to calling a function.

If you choose to call a particular function, include the function call in the following format as part of your response:

```
<function=function_name>{{"param1": "val1", ...}}</function>
```

where the JSON inside <function=...> contains exactly the parameters for that function. Pass the arguments in correct format, i.e., strings should be enclosed in quotes, lists should be enclosed in square brackets, integers should have no quotes, etc. If there are no parameters, then you should still include an empty JSON object `{{}}` inside the <function=...> tag.

### Function Results
After calling a function, stop generating text and wait for the results of the function call in the next message. The next message will use provide the output of the function call as a JSON object. Based on the output of the function call, you may call additional functions or provide a final answer."""


def _make_system_prompt(system_message: str, tools: Collection[Function]) -> str:
    if len(tools) == 0:
        return system_message

    functions = ""
    for tool in tools:
        tool_dict = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        }
        functions += json.dumps(tool_dict, indent=2)
        functions += "\n\n"

    prompt = _tool_calling_prompt.format(functions=functions)

    if system_message:
        prompt += "\n\n" + "## Additional Instructions" + "\n\n" + system_message

    return prompt


def _extract_balanced_json(s: str, start_idx: int = 0) -> str | None:
    """Find the first balanced {...} block starting at or after start_idx.
    Handles strings with escaped quotes properly.
    Returns the JSON substring (inclusive of braces), or None if not found."""
    depth = 0
    jstart = -1
    in_str = False
    esc = False
    for i in range(start_idx, len(s)):
        c = s[i]
        if esc:
            esc = False
            continue
        if in_str:
            if c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                jstart = i
            depth += 1
        elif c == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and jstart != -1:
                    return s[jstart : i + 1]
    return None


def _parse_model_output(completion: str) -> ChatAssistantMessage:
    """Parse model output for function calls.

    Handles two output styles:
      1. Llama-3.x style: ``<function=name>{...}</function>`` (paired tags)
      2. Qwen3 / markdown-wrapped style: ``<function=name>{...}`` (no close tag,
         often inside ```...``` code block, no </function>)
    """
    default_message = ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=[]
    )
    open_tag_pattern = re.compile(r"<function\s*=\s*([^>]+)>")
    open_match = open_tag_pattern.search(completion)
    if not open_match:
        return default_message

    function_name = open_match.group(1).strip()
    start_idx = open_match.end()

    # 1. Try paired-tag style first
    close_tag = "</function>"
    end_idx = completion.find(close_tag, start_idx)
    if end_idx != -1:
        raw_json = completion[start_idx:end_idx].strip()
        # Strip trailing markdown fences if any
        raw_json = re.sub(r"\s*```+\s*$", "", raw_json).strip()
    else:
        # 2. No </function> close tag — fall back to balanced-brace JSON extraction
        raw_json = _extract_balanced_json(completion, start_idx) or completion[start_idx:].strip()
        # Belt-and-suspenders: strip surrounding markdown fences if present
        raw_json = re.sub(r"\s*```+\s*$", "", raw_json).strip()

    try:
        params_dict = json.loads(raw_json)
        tool_calls = [FunctionCall(function=function_name, args=params_dict)]
    except json.JSONDecodeError:
        print(f"[debug] broken JSON: {raw_json!r}")
        return default_message
    except ValidationError:
        print(f"[debug] validation error (probably not type dict): {raw_json!r}")
        return default_message

    return ChatAssistantMessage(
        role="assistant", content=[text_content_block_from_string(completion.strip())], tool_calls=tool_calls
    )


class LocalLLM(BasePipelineElement):
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        temperature: float | None = 0.0,
        top_p: float | None = 0.9,
        tool_delimiter: str | None = "tool",
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.tool_delimiter = tool_delimiter

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        messages_ = []
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system" and content is not None:
                content = _make_system_prompt(get_text_content_as_str(content), runtime.functions.values())
            if role == "tool":
                role = self.tool_delimiter
                if "error" in m and m["error"] is not None:
                    content = json.dumps({"error": m["error"]})
                else:
                    func_result = m["content"]
                    if func_result == "None":
                        func_result = "Success"
                    content = json.dumps({"result": func_result})
            messages_.append({"role": role, "content": content})

        completion = chat_completion_request(
            self.client,
            model=self.model,
            messages=messages_,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        output = _parse_model_output(completion)
        return query, runtime, env, [*messages, output], extra_args
