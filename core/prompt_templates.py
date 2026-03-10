from typing import Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


Message = Dict[str, str]


def get_template_for_model(model_name: str) -> str:
    normalized = (model_name or "").lower()
    if "llama-3" in normalized or "llama 3" in normalized:
        return "llama3"
    if "mistral" in normalized:
        return "mistral"
    if "qwen" in normalized:
        return "qwen"
    if "deepseek" in normalized:
        return "deepseek"
    if "phi" in normalized:
        return "phi3"
    return "generic"


def build_chat_prompt(
    model_name: str,
    system_prompt: str,
    user_message: str,
    conversation_history: Optional[List[Message]] = None,
) -> str:
    template_name = get_template_for_model(model_name)
    logger.info(f"[PromptRouter] Using template: {template_name}")
    messages = _normalize_messages(system_prompt, conversation_history or [], user_message)

    builders = {
        "llama3": _build_llama3_prompt,
        "mistral": _build_mistral_prompt,
        "qwen": _build_qwen_prompt,
        "deepseek": _build_deepseek_prompt,
        "phi3": _build_phi3_prompt,
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
    turns = _split_system_and_turns(messages)
    system_prompt = turns["system"]
    exchanges = turns["turns"]
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
    turns = _split_system_and_turns(messages)
    system_prompt = turns["system"]
    exchanges = turns["turns"]
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


def _build_generic_prompt(messages: List[Message]) -> str:
    prompt_parts: List[str] = []
    for message in messages:
        prompt_parts.append(f"{message['role'].upper()}:\n{message['content']}\n\n")
    prompt_parts.append("ASSISTANT:\n")
    return "".join(prompt_parts)


def _split_system_and_turns(messages: List[Message]) -> Dict[str, List[Message] | str]:
    system_prompt = ""
    turns: List[Message] = []
    for message in messages:
        if message["role"] == "system" and not system_prompt:
            system_prompt = message["content"]
            continue
        turns.append(message)
    return {"system": system_prompt, "turns": turns}


def _merge_system_into_user(system_prompt: str, user_content: str) -> str:
    if not system_prompt.strip():
        return user_content
    return f"{system_prompt.strip()}\n\n{user_content.strip()}"
