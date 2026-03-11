# LLM Prompt Templates Backup

This document serves as a backup for the prompt templates and configurations of the local LLMs tested. Specifically generated for the models requested: Gemma, Llama 3, Mistral, and OpenChat.

---

## [2] gemma-2-9b-it-Q4_K_M.gguf

- **Template ID**: `gemma`
- **Family**: `gemma`
- **Size**: 9B (from filename)
- **Stop Sequences**: `["<end_of_turn>", "<start_of_turn>user"]`

### Structure
The system prompt is merged into the first user turn.

```text
<start_of_turn>user
{System Prompt (if provided) followed by double newline}

{User Message}<end_of_turn>
<start_of_turn>model
{Assistant Response}<end_of_turn>
```
*(The final prompt passed to the model append `<start_of_turn>model\n` to await the model's generation).*

---

## [3] llama-3-8b-instruct.Q4_K_M.gguf

- **Template ID**: `llama3`
- **Family**: `llama3`
- **Size**: 8B (from filename)
- **Stop Sequences**: `["<|eot_id|>", "<|start_header_id|>user<|end_header_id|>"]`

### Structure
Each turn has its own role block wrapped with `<|start_header_id|>` and `<|end_header_id|>`.

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{System Prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{User Message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{Assistant Response}<|eot_id|>
```
*(The final prompt passed to the model appends `<|start_header_id|>assistant<|end_header_id|>\n\n` to await generation).*

---

## [5] mistral-7b-instruct-v0.2.Q4_K_M.gguf

- **Template ID**: `mistral`
- **Family**: `mistral`
- **Size**: 7B (from filename)
- **Stop Sequences**: `["</s>", "[INST]"]`

### Structure
The system prompt is merged into the first user turn. `<s>` is not added manually because the tokenizer inserts it.

```text
[INST] {System Prompt (if provided) followed by double newline}

{User Message} [/INST] {Assistant Response} </s>
```

---

## [6] openchat-3.5-0106.Q4_K_M.gguf

- **Template ID**: `openchat`
- **Family**: `openchat`
- **Size**: 7B (from filename)
- **Stop Sequences**: `["<|end_of_turn|>", "GPT4 Correct User:"]`

### Structure
The system prompt acts as a clean preamble.

```text
<s>{System Prompt (if provided) followed by double newline}

GPT4 Correct User: {User Message}<|end_of_turn|>GPT4 Correct Assistant: {Assistant Response}<|end_of_turn|>
```
*(The final prompt passed to the model appends `GPT4 Correct Assistant:` to await the model's response).*
