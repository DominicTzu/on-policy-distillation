"""Small helpers for model generation."""


def progress(items, desc=None):
    try:
        from tqdm import tqdm

        return tqdm(items, desc=desc)
    except ImportError:
        return items


def get_model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def decode_new_tokens(tokenizer, output_ids, prompt_length):
    responses = []
    for ids in output_ids:
        generated_ids = ids[prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        responses.append(text)
    return responses
