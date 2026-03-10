import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


Message = Dict[str, str]


@dataclass(frozen=True)
class TemplateInfo:
    template: str
    family: str
    supported: bool
    size_b: Optional[float]
    size_label: str


def get_template_for_model(model_name: str) -> str:
    return get_model_template_info(model_name).template


def get_model_template_info(model_name: str) -> TemplateInfo:
    normalized = (model_name or "").lower()
    template = "generic"
    family = "generic"

    if "openchat" in normalized:
        template = "openchat"
        family = "openchat"
    elif "llama-3" in normalized or "llama 3" in normalized:
        template = "llama3"
        family = "llama3"
    elif "mistral" in normalized:
        template = "mistral"
        family = "mistral"
    elif "qwen" in normalized:
        template = "qwen"
        family = "qwen"
    elif "deepseek-coder" in normalized:
        template = "deepseek_chatml"
        family = "deepseek-coder"
    elif "deepseek" in normalized:
        template = "deepseek"
        family = "deepseek"
    elif "phi" in normalized:
        template = "phi3"
        family = "phi3"
    elif "gemma" in normalized:
        template = "gemma"
        family = "gemma"
    elif "wizardcoder" in normalized or "wizardlm" in normalized:
        template = "wizardcoder"
        family = "wizardcoder"
    elif "smollm3" in normalized or "smollm" in normalized:
        template = "smollm3"
        family = "smollm3"

    size_b = _detect_model_size_b(model_name)
    return TemplateInfo(
        template=template,
        family=family,
        supported=template != "generic",
        size_b=size_b,
        size_label=_format_size_label(size_b),
    )


def build_chat_prompt(
    model_name: str,
    system_prompt: str,
    user_message: str,
    conversation_history: Optional[List[Message]] = None,
) -> str:
    messages = _normalize_messages(system_prompt, conversation_history or [], user_message)
    return build_chat_prompt_from_messages(model_name, messages)


def build_chat_prompt_from_messages(
    model_name: str,
    messages: List[Message],
) -> str:
    template_info = get_model_template_info(model_name)
    template_name = template_info.template
    logger.info(f"[PromptRouter] Using template: {template_name}")

    builders = {
        "llama3": _build_llama3_prompt,
        "mistral": _build_mistral_prompt,
        "qwen": _build_qwen_prompt,
        "deepseek_chatml": _build_qwen_prompt,
        "deepseek": _build_deepseek_prompt,
        "phi3": _build_phi3_prompt,
        "openchat": _build_openchat_prompt,
        "gemma": _build_gemma_prompt,
        "wizardcoder": _build_wizardcoder_prompt,
        "smollm3": _build_smollm3_prompt,
        "generic": _build_generic_prompt,
    }
    return builders[template_name](messages)


def _normalize_messages(
    system_prompt: str,
    conversation_history: List[Message],
    user_message: str,
) -> List[Message]:
    messages: List[Message] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    for message in conversation_history:
        role = (message.get("role") or "").strip().lower()
        content = (message.get("content") or "").strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def _build_llama3_prompt(messages: List[Message]) -> str:
    prompt_parts: List[str] = ["<|begin_of_text|>"]
    for message in messages:
        prompt_parts.append(
            f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
            f"{message['content']}<|eot_id|>"
        )
    prompt_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(prompt_parts)


def _build_mistral_prompt(messages: List[Message]) -> str:
    system_prompt, exchanges = _split_system_and_turns(messages)
    if not exchanges:
        return f"<s>[INST] {system_prompt} [/INST]"

    prompt_parts: List[str] = []
    first_user = True
    for turn in exchanges:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            if first_user:
                content = _merge_system_into_user(system_prompt, content)
                prompt_parts.append(f"<s>[INST]\n{content}\n[/INST]")
                first_user = False
            else:
                prompt_parts.append(f"[INST]\n{content}\n[/INST]")
        elif role == "assistant":
            prompt_parts.append(f"{content}</s>")
    return "".join(prompt_parts)


def _build_qwen_prompt(messages: List[Message]) -> str:
    prompt_parts: List[str] = []
    has_system = any(message["role"] == "system" for message in messages)
    if not has_system:
        prompt_parts.append("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n")
    for message in messages:
        prompt_parts.append(
            f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        )
    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


def _build_deepseek_prompt(messages: List[Message]) -> str:
    system_prompt, exchanges = _split_system_and_turns(messages)
    prompt_parts = [system_prompt.strip(), "\n\n"] if system_prompt.strip() else []

    for turn in exchanges:
        if turn["role"] == "user":
            prompt_parts.extend(["### Instruction:\n", turn["content"], "\n"])
        elif turn["role"] == "assistant":
            prompt_parts.extend(["### Response:\n", turn["content"], "\n<|EOT|>\n"])

    if not exchanges or exchanges[-1]["role"] != "user":
        prompt_parts.extend(["### Instruction:\n", "\n"])
    prompt_parts.append("### Response:\n")
    return "".join(prompt_parts)


def _build_phi3_prompt(messages: List[Message]) -> str:
    prompt_parts: List[str] = []
    for message in messages:
        prompt_parts.append(
            f"<|{message['role']}|>\n{message['content']}<|end|>\n"
        )
    prompt_parts.append("<|assistant|>\n")
    return "".join(prompt_parts)


def _build_openchat_prompt(messages: List[Message]) -> str:
    system_prompt, turns = _split_system_and_turns(messages)
    if turns and turns[0]["role"] == "user" and system_prompt.strip():
        turns = turns.copy()
        turns[0] = {"role": "user", "content": _merge_system_into_user(system_prompt, turns[0]["content"])}
        system_prompt = ""

    prompt_parts: List[str] = ["<s>"]
    for turn in turns:
        if turn["role"] == "user":
            prompt_parts.extend(
                ["GPT4 Correct User: ", turn["content"], "<|end_of_turn|>"]
            )
        elif turn["role"] == "assistant":
            prompt_parts.extend(
                ["GPT4 Correct Assistant: ", turn["content"], "<|end_of_turn|>"]
            )
    prompt_parts.append("GPT4 Correct Assistant:")
    return "".join(prompt_parts)


def _build_gemma_prompt(messages: List[Message]) -> str:
    system_prompt, turns = _split_system_and_turns(messages)
    if turns and turns[0]["role"] == "user":
        turns = turns.copy()
        turns[0] = {"role": "user", "content": _merge_system_into_user(system_prompt, turns[0]["content"])}
        system_prompt = ""

    prompt_parts: List[str] = []
    for turn in turns:
        role = "user" if turn["role"] == "user" else "model"
        prompt_parts.extend(
            [f"<start_of_turn>{role}\n", turn["content"], "<end_of_turn>\n"]
        )
    if system_prompt.strip():
        prompt_parts.extend(
            [
                "<start_of_turn>user\n",
                system_prompt.strip(),
                "<end_of_turn>\n",
            ]
        )
    prompt_parts.append("<start_of_turn>model\n")
    return "".join(prompt_parts)


def _build_wizardcoder_prompt(messages: List[Message]) -> str:
    instruction = _build_instruction_transcript(messages)
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def _build_smollm3_prompt(messages: List[Message]) -> str:
    system_prompt, turns = _split_system_and_turns(messages)
    system_block = system_prompt.strip()
    if "/think" not in system_block and "/no_think" not in system_block:
        system_block = f"/no_think\n{system_block}".strip()

    prompt_parts = [
        "<|im_start|>system\n",
        system_block if system_block else "/no_think\nYou are a helpful AI assistant.",
        "\n<|im_end|>\n",
    ]
    for turn in turns:
        prompt_parts.extend(
            [
                f"<|im_start|>{turn['role']}\n",
                turn["content"],
                "\n<|im_end|>\n",
            ]
        )
    prompt_parts.append("<|im_start|>assistant\n")
    return "".join(prompt_parts)


def _build_generic_prompt(messages: List[Message]) -> str:
    prompt_parts: List[str] = []
    for message in messages:
        prompt_parts.append(f"{message['role'].upper()}:\n{message['content']}\n\n")
    prompt_parts.append("ASSISTANT:\n")
    return "".join(prompt_parts)


def _split_system_and_turns(messages: List[Message]) -> tuple[str, List[Message]]:
    system_prompt = ""
    turns: List[Message] = []
    for message in messages:
        if message["role"] == "system" and not system_prompt:
            system_prompt = message["content"]
            continue
        turns.append(message)
    return system_prompt, turns


def _merge_system_into_user(system_prompt: str, user_content: str) -> str:
    if not system_prompt.strip():
        return user_content
    return f"{system_prompt.strip()}\n\n{user_content.strip()}"


def _build_instruction_transcript(messages: List[Message]) -> str:
    system_prompt, turns = _split_system_and_turns(messages)
    if not turns:
        return system_prompt.strip()

    current_user = ""
    history_lines: List[str] = []
    for turn in turns[:-1]:
        label = "User" if turn["role"] == "user" else "Assistant"
        history_lines.append(f"{label}: {turn['content']}")
    if turns[-1]["role"] == "user":
        current_user = turns[-1]["content"]

    sections: List[str] = []
    if system_prompt.strip():
        sections.extend(["System Instructions:\n", system_prompt.strip(), "\n\n"])
    if history_lines:
        sections.extend(["Conversation History:\n", "\n".join(history_lines), "\n\n"])
    sections.extend(["Current Request:\n", current_user.strip()])
    return "".join(sections).strip()


def _detect_model_size_b(model_name: str) -> Optional[float]:
    normalized = (model_name or "").lower()
    if "openchat-3.5" in normalized:
        return 7.0
    if "phi-3-mini" in normalized or "phi3-mini" in normalized:
        return 3.8

    match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", normalized, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _format_size_label(size_b: Optional[float]) -> str:
    if size_b is None:
        return "unknown"
    if size_b.is_integer():
        return f"{int(size_b)}B"
    return f"{size_b:g}B"
