from .base_model_inference import *
import requests
import math
import re
from io import BytesIO
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, LlamaTokenizer


class CogAgentProcessor(BaseModelInference):
    def __init__(self, model_name, local_save_path=""):
        super().__init__(model_name, local_save_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

    def inference(self, *args, **kwargs):
        self._extract_arguments(**kwargs)
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer, query=self.user_prompt, history=[], images=[self.raw_image]
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": [[input_by_model["images"][0].to(self.device).to(torch.float16)]],
            "cross_images": [
                [input_by_model["cross_images"][0].to(self.device).to(torch.float16)]
            ],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}  # "temperature": 0.9
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
        self.result_rext = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

    def extract_answers(self):
        return self.result_rext.split("ASSISTANT:")[-1].strip()

    def _extract_arguments(self, **kwargs):
        self.user_prompt = kwargs["user_prompt"]
        self.raw_image = kwargs["raw_image"]
        self.max_new_tokens = kwargs.get("max_new_tokens", 300)
        self.do_sample = kwargs.get("do_sample", False)
        self.temperature = kwargs.get("temperature", 1)
