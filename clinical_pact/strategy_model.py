"""Local bounded-grammar proposer and single-example current-policy evaluator.

These interfaces are used by the bounded distributed development driver.
They never use an external reflection service or pass medical samples to the
proposer. Callers must account for model loading, all calls and audit costs.
"""
import json
import time
from .strategy_cycle import PROCEDURES, proposal_texts
from .scoring import score_response


class LocalProcedureProposer:
    def __init__(self, model, tokenizer, device):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.cost = []

    def __call__(self, feedback, procedures):
        import torch
        if tuple(procedures) != PROCEDURES:
            raise ValueError('proposal grammar differs from approved procedures')
        if set(feedback) != {'n', 'canonical_correct', 'current_correct'}:
            raise ValueError('only aggregate audit feedback is allowed')
        n = feedback['n']
        counts = [feedback['canonical_correct']]
        if type(n) is not int or n <= 0 or not isinstance(feedback['current_correct'], list) or len(feedback['current_correct']) != 2:
            raise ValueError('invalid aggregate feedback')
        counts += feedback['current_correct']
        if any(type(count) is not int or count < 0 or count > n for count in counts):
            raise ValueError('invalid verifier counts')
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('proposer must be a frozen local model in eval mode')
        message = [{'role':'system', 'content':'Select two reusable answering procedures. Return only a JSON array of two arrays of integer procedure IDs. Each array has one to four distinct IDs. Do not add text, facts or answers.'},
                   {'role':'user', 'content':json.dumps({'aggregate_training_feedback':feedback,'allowed_procedures':dict(enumerate(PROCEDURES))},ensure_ascii=True)}]
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        start = time.monotonic()
        entry = {'prompt_tokens':inputs['input_ids'].numel(), 'generated_tokens':None, 'accepted':False}
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                             max_new_tokens=64, repetition_penalty=1.0,
                                             use_cache=True, synced_gpus=False)
            new_tokens = output[:, inputs['input_ids'].shape[1]:]
            entry['generated_tokens'] = new_tokens.numel()
        finally:
            entry['elapsed_seconds'] = time.monotonic()-start
            self.cost.append(entry)
        # Reject malformed proposals without repairs or uncharged retries.
        result = json.loads(self.tokenizer.decode(new_tokens[0], skip_special_tokens=True))
        proposal_texts(result)
        entry['accepted'] = True
        return result


class PolicyStrategyEvaluator:
    def __init__(self, model, processor, training_items, device, *, split):
        if split != 'train':
            raise ValueError('strategy evaluator only accepts training data')
        items = list(training_items)
        self.items = {item['id']:item for item in items}
        if not items or len(items) != len(self.items):
            raise ValueError('training item IDs must be nonempty and unique')
        self.model, self.processor, self.device = model, processor, device
        self.cost = []

    def __call__(self, text, audit_ids):
        import torch
        from .multimodal import make_inputs
        if self.model.training:
            raise ValueError('freeze current policy updates and use eval mode during audit')
        if any(key not in self.items for key in audit_ids):
            raise ValueError('audit ID outside declared training pool')
        config = self.model.generation_config
        if config.forced_bos_token_id is not None or config.forced_eos_token_id is not None or config.suppress_tokens or config.begin_suppress_tokens:
            raise ValueError('implicit generation constraints are not part of the audit protocol')
        scores = []
        for key in audit_ids:
            item = self.items[key]
            inputs = make_inputs(self.processor, item, text, self.device, append_strategy=True)
            started = time.monotonic()
            with torch.inference_mode():
                result = self.model.generate(**inputs,do_sample=False,num_beams=1,max_new_tokens=128,
                                             repetition_penalty=1.0,use_cache=True,synced_gpus=False)
            response = result[:,inputs['input_ids'].shape[1]:]
            answer = self.processor.tokenizer.decode(response[0],skip_special_tokens=True)
            scores.append(int(score_response(answer,item['answer'])))
            self.cost.append({'prompt_tokens':inputs['input_ids'].numel(),'generated_tokens':response.numel(),
                              'terminated':bool(response.numel() and int(response[0,-1]) == self.processor.tokenizer.eos_token_id),
                              'elapsed_seconds':time.monotonic()-started})
        return tuple(scores)
