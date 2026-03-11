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
        if "coder" in normalized:
            family = "qwen-coder"
        else:
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


def get_stop_sequences_for_model(model_name: str) -> Optional[List[str]]:
    template_name = get_model_template_info(model_name).template
    stops = {
        "llama3": ["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"],
        "mistral": ["</s>", "[INST]"],
        "qwen": ["<|im_end|>", "<|im_start|>user", "<|im_start|>system"],
        "deepseek_chatml": ["<|im_end|>", "<|im_start|>user", "<|im_start|>system"],
        "deepseek": ["<|EOT|>", "### Instruction:"],
        "phi3": ["<|end|>", "<|user|>", "<|system|>"],
        "openchat": ["<|end_of_turn|>", "GPT4 Correct User:"],
        "gemma": ["<end_of_turn>", "<start_of_turn>user"],
        "wizardcoder": ["### Instruction:", "### Response:"],
        "smollm3": ["<|im_end|>", "<|im_start|>user", "<|im_start|>system"],
    }
    return stops.get(template_name)


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
    """
    Mistral Instruct format.
    Merge the cleaned system prompt into the first user turn and emit the
    tokenizer-native `[INST] ... [/INST] assistant </s>` boundaries.
    Do not prepend `<s>` manually because many GGUF Mistral builds already
    auto-insert BOS at tokenization time.
    """
    system_prompt, exchanges = _split_system_and_turns(messages)

    if not exchanges:
        clean = _strip_xml_from_system(system_prompt)
        content = clean.strip() if clean else ""
        return f"[INST] {content} [/INST]" if content else "[INST] [/INST]"

    clean_system = _strip_xml_from_system(system_prompt)
    prompt_parts: List[str] = []
    first_user = True

    for turn in exchanges:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            if first_user and clean_system:
                content = f"{clean_system}\n\n{content.strip()}"
            normalized = content.strip()
            prompt_parts.append(f"[INST] {normalized} [/INST]")
            first_user = False
        elif role == "assistant":
            prompt_parts.append(f" {content.strip()} </s>")

    return "".join(prompt_parts).strip()


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
    """
    DeepSeek legacy instruction format.
    System prompt is prepended as raw text after XML stripping.
    """
    system_prompt, exchanges = _split_system_and_turns(messages)
    clean_system = _strip_xml_from_system(system_prompt)

    prompt_parts: List[str] = []
    if clean_system.strip():
        prompt_parts.extend([clean_system.strip(), "\n\n"])

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
    """
    OpenChat 3.5 GPT4 Correct format.
    Keep system as a clean prose preamble instead of merging XML into the first user turn.
    """
    system_prompt, turns = _split_system_and_turns(messages)
    clean_system = _strip_xml_from_system(system_prompt)

    prompt_parts: List[str] = ["<s>"]
    if clean_system:
        prompt_parts.extend([clean_system, "\n\n"])

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
    """
    Gemma 2 format. Merge a cleaned system prompt into the first user turn.
    """
    system_prompt, turns = _split_system_and_turns(messages)

    if not turns:
        if system_prompt.strip():
            return (
                f"<start_of_turn>user\n{system_prompt.strip()}<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        return "<start_of_turn>model\n"

    clean_system = _strip_xml_from_system(system_prompt)

    prompt_parts: List[str] = []
    first_user_processed = False

    for turn in turns:
        role = "user" if turn["role"] == "user" else "model"
        content = turn["content"]

        if role == "user" and not first_user_processed:
            if clean_system:
                content = f"{clean_system}\n\n{content.strip()}"
            first_user_processed = True

        prompt_parts.extend(
            [
                f"<start_of_turn>{role}\n",
                content,
                "<end_of_turn>\n",
            ]
        )

    prompt_parts.append("<start_of_turn>model\n")
    return "".join(prompt_parts)


def _build_wizardcoder_prompt(messages: List[Message]) -> str:
    """
    WizardCoder only gets the current task plus short clean context.
    """
    system_prompt, turns = _split_system_and_turns(messages)

    current_request = ""
    history_pairs: List[str] = []
    is_code_task = False

    for index, turn in enumerate(turns):
        if turn["role"] == "user" and index == len(turns) - 1:
            current_request = turn["content"].strip()
            is_code_task = _is_code_like_text(current_request)
        elif turn["role"] == "user":
            history_pairs.append(f"User: {turn['content'].strip()}")
        elif turn["role"] == "assistant":
            history_pairs.append(f"Assistant: {turn['content'].strip()}")

    clean_system = _strip_xml_from_system(system_prompt)
    concise_system = _truncate_words(clean_system, 100)

    instruction_parts: List[str] = []
    if concise_system:
        instruction_parts.append(concise_system)
        instruction_parts.append("")

    if history_pairs and is_code_task:
        recent = history_pairs[-6:]
        instruction_parts.append("Previous conversation:")
        instruction_parts.extend(recent)
        instruction_parts.append("")

    instruction_parts.append(current_request)
    instruction = "\n".join(instruction_parts).strip()

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def _build_smollm3_prompt(messages: List[Message]) -> str:
    """
    SmolLM3 format with dynamic thinking mode.
    """
    system_prompt, turns = _split_system_and_turns(messages)
    system_block = system_prompt.strip()

    if "/think" in system_block or "/no_think" in system_block:
        pass
    else:
        think_directive = "/think" if _smollm3_is_complex_query(turns) else "/no_think"
        system_block = f"{think_directive}\n{system_block}".strip()

    if not system_block:
        system_block = "/no_think\nYou are a helpful AI assistant."

    prompt_parts = [
        "<|im_start|>system\n",
        system_block,
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


def _strip_xml_from_system(system_prompt: str) -> str:
    """
    Remove XML wrappers while preserving useful memory content as plain text.
    """
    if not system_prompt:
        return ""

    def replace_block(match: re.Match) -> str:
        tag = match.group(1).lower()
        body = match.group(2)
        rendered = _render_plaintext_system_block(tag, body)
        return f"\n{rendered}\n" if rendered else "\n"

    xml_block_pattern = re.compile(
        r"<(knowledge|answer_policy|conversation_rules|personal_memory_guard|"
        r"preference_resolution|retrieved_facts|semantic_claims|episodic_context|"
        r"project_context|document_context|research_context|identity)>\s*"
        r"(.*?)"
        r"\s*</\1>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = xml_block_pattern.sub(replace_block, system_prompt)
    cleaned = re.sub(r"IMPORTANT:.*?(?=\n|$)", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"Local Knowledge Confidence:.*?(?=\n|$)", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"Web Research Allowed:.*?(?=\n|$)", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"System Time:.*?(?=\n|$)", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _truncate_words(text: str, limit: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    return " ".join(words[:limit]).strip()


def _render_plaintext_system_block(tag: str, body: str) -> str:
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return ""

    if tag in {
        "knowledge",
        "retrieved_facts",
        "semantic_claims",
        "episodic_context",
        "project_context",
        "document_context",
        "research_context",
        "identity",
    }:
        facts = []
        for line in lines:
            line = re.sub(r"^\[id=[^\]]+\]\s*", "", line)
            line = re.sub(r"\s+\|\s+confidence=.*$", "", line)
            line = re.sub(r"\s+\|\s+score=.*$", "", line)
            line = re.sub(r"\s+\|\s+source=.*$", "", line)
            cleaned = line.strip(" -")
            if cleaned:
                facts.append(cleaned)
        return "\n".join(facts)

    return "\n".join(lines)


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


def _smollm3_is_complex_query(turns: List[Message]) -> bool:
    if not turns:
        return False
    last_user = next(
        (turn["content"].lower() for turn in reversed(turns) if turn["role"] == "user"),
        "",
    )
    complex_markers = (
        "explain",
        "why",
        "how does",
        "reasoning",
        "step by step",
        "calculate",
        "prove",
        "derive",
        "analyze",
        "compare",
        "difference between",
        "advantage",
        "disadvantage",
        "trade-off",
        "math",
        "equation",
        "logic",
        "algorithm complexity",
    )
    return any(marker in last_user for marker in complex_markers)


def _is_code_like_text(text: str) -> bool:
    lowered = (text or "").lower().strip()
    code_markers = (
        "write a",
        "write the",
        "create a",
        "create the",
        "implement",
        "build a",
        "build the",
        "code a",
        "code the",
        "function",
        "class ",
        "script",
        "algorithm",
        "program",
        "module",
        "snippet",
        "debug",
        "fix the bug",
        "fix this",
        "refactor",
        "optimize the",
        "unit test",
        "generate code",
        "snake game",
        "pygame",
        "flask",
        "fastapi",
        "sql query",
        "regex",
        "decorator",
        "recursion",
    )
    return any(marker in lowered for marker in code_markers)


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
