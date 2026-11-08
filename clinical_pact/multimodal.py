"""Single-example Qwen2.5-VL prompt and immutable decision memory construction."""
import re
from pathlib import Path
import torch
import yaml
from jinja2 import Environment, StrictUndefined
from PIL import Image
from .core import token_margins
from .decision_loss import aligned_response_logits
from .scoring import score_response


def make_inputs(processor, item, strategy='', device='cpu'):
    config = yaml.safe_load((Path(__file__).parent / 'prompts.yaml').read_text())
    question = re.sub(r'<image(?:[_ ]?\d+)?>', '', item['question']).strip()
    text = Environment(undefined=StrictUndefined, autoescape=False).from_string(config['canonical_user']).render(question=question, strategy=strategy)
    with Image.open(item['image']) as source:
        image = source.convert('RGB')
    messages = [{'role': 'system', 'content': config['system']}, {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': text}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=False, return_tensors='pt')
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}


@torch.no_grad()
def greedy_reference(model, processor, item, device, max_new_tokens=128):
    """Generate and validate a scored reference; return private data or a rejection reason."""
    inputs = make_inputs(processor, item, device=device)
    config = model.generation_config
    if config.forced_bos_token_id is not None or config.forced_eos_token_id is not None or config.suppress_tokens or config.begin_suppress_tokens:
        raise ValueError('implicit generation constraints make reference margins invalid')
    generated = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0, use_cache=True)
    start = inputs['input_ids'].shape[1]
    response = generated[:, start:]
    eos_ids = processor.tokenizer.eos_token_id
    if response.shape[1] == 0 or int(response[0, -1]) != eos_ids:
        return None, 'missing_eos_or_truncated'
    text = processor.tokenizer.batch_decode(response, skip_special_tokens=True)[0]
    if not score_response(text, item['answer']):
        return None, 'incorrect'
    teacher_inputs = dict(inputs)
    teacher_inputs['input_ids'] = generated
    teacher_inputs['attention_mask'] = torch.ones_like(generated)
    logits, targets = aligned_response_logits(model, teacher_inputs, start)
    if not torch.equal(logits.argmax(-1), targets):
        raise RuntimeError('generation and teacher-forcing differ: stop and diagnose')
    margins = token_margins(logits, targets)[0]
    if not torch.isfinite(margins).all() or (margins < 0.05).any():
        return None, 'near_tie'
    return {'response_ids': tuple(int(x) for x in response[0]), 'margins': tuple(float(x) for x in margins), 'eos_id': eos_ids}, None
