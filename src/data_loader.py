from datasets import load_dataset
from transformers import AutoTokenizer

CLASS_NAMES = ("World", "Sports", "Business", "Sci/Tech")
CLASS_CODES = (" A", " B", " C", " D")


def get_dataset(
    dataset_name,
    split,
    sample_limit,
    *,
    tokenizer=None,
    max_length=512,
    seed=42,
):
    if dataset_name not in {"ag_news", "fancyzhx/ag_news"}:
        raise ValueError("Only the AG News four-class protocol is supported")
    if sample_limit is not None and (type(sample_limit) is not int or sample_limit < 1):
        raise ValueError("sample_limit must be a positive integer or None")
    if type(max_length) is not int or max_length < 64:
        raise ValueError("max_length must be an integer of at least 64")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-4k-instruct", trust_remote_code=False
        )
    dataset = load_dataset("fancyzhx/ag_news", split=split)
    dataset = dataset.add_column("sample_id", list(range(len(dataset))))
    dataset = dataset.shuffle(seed=seed)
    if sample_limit is not None:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))
    if not len(dataset):
        raise ValueError("The selected dataset is empty")
    prefix = tokenizer.encode(
        "Classify the news article. A: World; B: Sports; C: Business; "
        "D: Sci/Tech. Return the category letter.\nArticle: ",
        add_special_tokens=True,
    )
    suffix = tokenizer.encode("\nCategory:", add_special_tokens=False)
    reserve = max(
        len(tokenizer.encode(code, add_special_tokens=False)) for code in CLASS_CODES
    )
    budget = max_length - len(prefix) - len(suffix) - reserve
    if budget < 1:
        raise ValueError("max_length is too small for the classification prompt")

    def tokenize(batch):
        result = tokenizer(
            batch["text"], truncation=True, max_length=max_length, padding=False
        )
        articles = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
            padding=False,
        )["input_ids"]
        result["prompt_ids"] = [prefix + article + suffix for article in articles]
        return result

    return dataset.map(tokenize, batched=True, desc="Tokenizing AG News")
