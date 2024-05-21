from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import numpy as np
import random


def setup_seed(seed):
    # TODO: Improve randomization to make it global and permanent
    # When using CUDA, the env var in the .env file comes into play!
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For Multi-GPU, exception safe (https://github.com/pytorch/pytorch/issues/108341)
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def generate_text(
    prompt, 
    model_name="gpt2", 
    max_length=50, 
    num_return_sequences=1, 
    do_sample=True, 
    top_k=50,
    temperature=0.7
):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    generated_text = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(generated_text)
    return


if __name__ == '__main__':
    setup_seed(1)
    prompt = "Thanks for all the"
    generate_text(prompt)