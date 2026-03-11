import unittest

from core.context_builder import ContextBuilder
from core.orchestrator import Orchestrator
from core.prompt_templates import build_chat_prompt, build_chat_prompt_from_messages, get_model_template_info


SYSTEM_WITH_MEMORY = """You are a persistent AI assistant named Jarvis.

<knowledge>
[id=abc] The user likes Python | confidence=0.92 | score=0.88 | source=personal_memory
</knowledge>

<answer_policy>
- Answer only from memory evidence when applicable.
</answer_policy>

<conversation_rules>
- Keep responses concise and natural.
</conversation_rules>

System Time: 2026-03-11 12:01:48
Local Knowledge Confidence: HIGH

IMPORTANT: Answer only from memory evidence when applicable.
"""


class DummyAuthorityMemory:
    def get_injected_prompt(self, prompt: str, compact: bool = False) -> str:
        return prompt

    def load(self):
        return {}


class DummyItem:
    def __init__(self, memory_id, text, confidence=0.92, score=0.88, collection="personal_memory"):
        self.memory_id = memory_id
        self.text = text
        self.confidence = confidence
        self.score = score
        self.collection = collection


class DummyPacket:
    identity_items = []
    fact_items = [DummyItem("abc", "The user likes Python"), DummyItem("fav", "favorite number 2545")]
    semantic_items = []
    episodic_items = []
    project_items = []
    document_items = []
    research_items = []
    answer_policy = ["Answer only from memory evidence when applicable."]


class DummyPolicy:
    def __init__(self, name="personal_memory", strict_grounding=True):
        self.name = name
        self.strict_grounding = strict_grounding


class DummyMemoryManager:
    def __init__(self):
        self.authority_memory = DummyAuthorityMemory()

    def retrieve_context(self, user_input, n_results=5, model_profile=None):
        return {
            "type": "knowledge_collection",
            "evidence_packet": DummyPacket(),
            "policy": DummyPolicy(),
            "scored_items": [{"score": 0.88, "type": "fact", "text": "The user likes Python"}],
        }


class PromptPipelineModelsTest(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder(DummyMemoryManager())

    def test_native_chat_models_keep_memory_xml(self):
        prompt = build_chat_prompt("Meta-Llama-3-8B-Instruct.gguf", SYSTEM_WITH_MEMORY, "what do you know about me?")
        self.assertIn("<knowledge>", prompt)

        prompt = build_chat_prompt("Qwen2.5-7B-Instruct.gguf", SYSTEM_WITH_MEMORY, "what do you know about me?")
        self.assertIn("<knowledge>", prompt)

        prompt = build_chat_prompt("Phi-3-mini.gguf", SYSTEM_WITH_MEMORY, "what do you know about me?")
        self.assertIn("<knowledge>", prompt)

    def test_clean_prompt_models_strip_xml(self):
        for model_name in [
            "WizardCoder-Python-7B.gguf",
            "Mistral-7B-Instruct.gguf",
            "OpenChat-3.5-7B.gguf",
            "gemma-2-9b-it.gguf",
            "deepseek-7b.gguf",
        ]:
            with self.subTest(model_name=model_name):
                prompt = build_chat_prompt(model_name, SYSTEM_WITH_MEMORY, "what do you know about me?")
                self.assertNotIn("<knowledge>", prompt)
                self.assertIn("The user likes Python", prompt)

    def test_code_model_family_detection(self):
        self.assertEqual(get_model_template_info("WizardCoder-Python-7B.gguf").family, "wizardcoder")
        self.assertEqual(get_model_template_info("deepseek-coder-6.7b.gguf").family, "deepseek-coder")
        self.assertEqual(get_model_template_info("qwen2.5-coder-7b-instruct-q4_k_m.gguf").family, "qwen-coder")

    def test_code_models_differentiate_code_and_chat_queries(self):
        for model_name in [
            "WizardCoder-Python-7B.gguf",
            "deepseek-coder-6.7b.gguf",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        ]:
            with self.subTest(model_name=model_name):
                prompt, temperature = self.builder.build_context(
                    user_input="write a fibonacci function",
                    history=[],
                    model_name=model_name,
                )
                self.assertNotIn("<knowledge>", prompt)
                self.assertEqual(temperature, 0.15)
                if "wizardcoder" in model_name.lower():
                    self.assertIn("expert programmer", prompt.lower())
                else:
                    self.assertIn("expert programmer", prompt.lower())

                prompt, temperature = self.builder.build_context(
                    user_input="hi how are you",
                    history=[],
                    model_name=model_name,
                )
                self.assertNotIn("<knowledge>", prompt)
                self.assertNotIn("expert programmer", prompt.lower())
                self.assertIn("helpful personal ai assistant", prompt.lower())
                self.assertIn("favorite number 2545", prompt.lower())
                self.assertEqual(temperature, 0.35)

    def test_llama3_context_builder_keeps_memory_injection(self):
        prompt, temperature = self.builder.build_context(
            user_input="what do you know about me?",
            history=[],
            model_name="Meta-Llama-3-8B-Instruct.gguf",
        )
        self.assertIn("<knowledge>", prompt)
        self.assertIn("The user likes Python", prompt)
        self.assertIn("favorite number 2545", prompt)
        self.assertIn("repeat it exactly", prompt)
        self.assertLessEqual(temperature, 0.6)
        self.assertIn("personal memory recall assistant", prompt.lower())

    def test_mistral_prompt_does_not_start_with_manual_bos(self):
        prompt = build_chat_prompt("Mistral-7B-Instruct.gguf", SYSTEM_WITH_MEMORY, "explain recursion")
        self.assertTrue(prompt.startswith("[INST]"))
        self.assertFalse(prompt.startswith("<s>[INST]"))
        self.assertIn("[/INST]", prompt)

    def test_mistral_prompt_keeps_memory_context_in_first_inst_block(self):
        prompt, _ = self.builder.build_context(
            user_input="what is my fav number ?",
            history=[],
            model_name="Mistral-7B-Instruct.gguf",
        )
        self.assertIn("The user likes Python", prompt)
        self.assertIn("favorite number 2545", prompt)
        self.assertIn("[INST]", prompt)

    def test_wizardcoder_preserves_internal_helper_instructions(self):
        prompt = build_chat_prompt_from_messages(
            "WizardCoder-Python-7B.gguf",
            [
                {
                    "role": "system",
                    "content": (
                        "You are an internal memory distillation module for an AI. "
                        "Your job is to read the raw retrieved memories/documents and the user's current query, "
                        "and synthesize ONLY the facts from the memories that are directly relevant to answering the query. "
                        "Condense them into a dense, concise set of bullet points. "
                        "DO NOT answer the user's query. DO NOT hallucinate. DO NOT explain what you are doing. "
                        "ONLY output the distilled facts. If none of the memories are relevant to the query at all, output 'NO_RELEVANT_CONTEXT'."
                    ),
                },
                {
                    "role": "user",
                    "content": "User Query: what do you know about me?\n\nRaw Memories:\nThe user likes Python.",
                },
            ],
        )
        self.assertIn("internal memory distillation module", prompt)
        self.assertIn("NO_RELEVANT_CONTEXT", prompt)
        self.assertNotIn("Previous conversation:", prompt)

    def test_orchestrator_respects_code_model_base_temperatures(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        self.assertEqual(
            orchestrator._tuned_temperature("hi how are you", "deepseek-coder-6.7b.gguf", 0.35),
            0.35,
        )
        self.assertEqual(
            orchestrator._tuned_temperature("write a fibonacci function", "WizardCoder-Python-7B.gguf", 0.15),
            0.15,
        )

    def test_wizardcoder_chat_prompt_does_not_embed_chat_role_transcript(self):
        prompt = build_chat_prompt(
            "WizardCoder-Python-7B.gguf",
            "You are a helpful personal AI assistant.\nRelevant context: favorite color blue. pet dog Bruno.",
            "what is my fav color?",
            [
                {"role": "user", "content": "hi jarvis how are you"},
                {"role": "assistant", "content": "Hi Flex, I am doing well."},
            ],
        )
        self.assertNotIn("Previous conversation:", prompt)
        self.assertNotIn("User:", prompt)
        self.assertNotIn("Assistant:", prompt)
        self.assertIn("favorite color blue", prompt)

    def test_personal_and_identity_queries_are_detected(self):
        self.assertTrue(self.builder._is_personal_query("what do you know about me?"))
        self.assertTrue(self.builder._is_personal_query("hi jarvis what do you know about flex ?"))
        self.assertTrue(self.builder._is_identity_query("who am i ?"))
        self.assertTrue(self.builder._is_identity_query("who are you ?"))

    def test_personal_memory_queries_drop_chat_history_from_prompt(self):
        prompt, _ = self.builder.build_context(
            user_input="what is my fav number ?",
            history=[
                {"role": "user", "content": "what is my favorite number?"},
                {"role": "assistant", "content": "Your favorite number is 42."},
            ],
            model_name="Meta-Llama-3-8B-Instruct.gguf",
        )
        self.assertNotIn("Your favorite number is 42.", prompt)
        self.assertIn("favorite number 2545", prompt)


if __name__ == "__main__":
    unittest.main()
