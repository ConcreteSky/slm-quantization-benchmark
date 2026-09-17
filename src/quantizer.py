import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class QuantizationLoader:
    def __init__(self, model_id, precision, device=0, revision="main"):
        if precision not in {"fp16", "int8", "nf4", "4bit"}:
            raise ValueError(f"Unsupported precision: {precision}")
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for this benchmark protocol")
        if type(device) is not int or not 0 <= device < torch.cuda.device_count():
            raise ValueError("device must identify an available CUDA GPU")
        self.model_id = model_id
        self.precision = "nf4" if precision == "4bit" else precision
        self.device = device
        self.revision = revision

    def load(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer requires a padding or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        options = {
            "revision": self.revision,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": torch.float16,
            "device_map": {"": self.device},
            "attn_implementation": "eager",
        }
        if self.precision == "int8":
            options["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.precision == "nf4":
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **options)
        model.eval()
        return model, tokenizer
