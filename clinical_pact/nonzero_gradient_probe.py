"""Controlled actual-model auxiliary-gradient check, without optimizer updates.

Uses the first immutable private training reference and the exact stage-one
model. A detached logit intervention makes one EOS competitor win by tau;
this is an engineering intervention, not a training-method or efficacy result.
"""
import json
import os
import time
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .multimodal import make_inputs
from .decision_loss import aligned_response_logits
from .core import decision_deficits, token_margins


def main():
    root = Path(os.environ['OUT'])
    start_time = time.time()
    torch.cuda.set_device(0)
    model_path = root / 'stage1/model'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=128*28*28, max_pixels=512*28*28)
    records = torch.load(root / 'private_memory.pt', map_location='cpu', weights_only=False)
    if not records:
        raise RuntimeError('no original private training reference')
    item, reference = records[0]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter = model.lm_head.weight
    parameter.requires_grad_(True)
    inputs = make_inputs(processor, item, device='cuda')
    response_start = inputs['input_ids'].shape[1]
    response = torch.tensor([reference['response_ids']], device='cuda', dtype=torch.long)
    if int(response[0, -1]) != reference['eos_id']:
        raise RuntimeError('cached response lacks EOS')
    inputs['input_ids'] = torch.cat((inputs['input_ids'], response), dim=1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    logits, targets = aligned_response_logits(model, inputs, response_start)
    mask = torch.ones_like(targets, dtype=torch.bool)
    tau = torch.tensor([[.5*min(m,2.) for m in reference['margins']]], device='cuda', dtype=torch.float32)
    baseline = decision_deficits(logits, targets, mask, tau).item()
    if baseline > 1e-4:
        raise RuntimeError('immediate reference replay has nonzero deficit')
    z = logits.float()
    index = targets.shape[1]-1
    target = int(targets[0,index])
    values, indices = z[0,index].topk(2)
    competitor = int(indices[1] if int(indices[0]) == target else indices[0])
    gap = token_margins(z, targets)[0,index]
    offset = (gap.detach() + tau[0,index]).detach()
    modified = z.clone()
    modified[0,index,competitor] = z[0,index,competitor] + offset
    actual = decision_deficits(modified, targets, mask, tau).sum()
    explicit = 1-(z[0,index,target]-z[0,index,competitor]-offset)/tau[0,index]
    if not (actual.item() >= 1 and torch.allclose(actual, explicit, atol=1e-5, rtol=1e-5)):
        raise RuntimeError('controlled gap intervention failed')
    gradient = torch.autograd.grad(actual, parameter, retain_graph=True)[0]
    expected = torch.autograd.grad(explicit, parameter)[0]
    if not torch.isfinite(gradient).all() or not torch.isfinite(expected).all():
        raise RuntimeError('nonfinite auxiliary gradient')
    magnitude = float(gradient.float().abs().max())
    error = float((gradient.float()-expected.float()).abs().max())
    relative = error / max(magnitude,1e-12)
    result = {'reference_rule':'first stored eligible training memory', 'baseline_deficit':baseline,
              'controlled_deficit':actual.item(), 'explicit_deficit':explicit.item(),
              'gradient_max_abs':magnitude, 'gradient_difference_max_abs':error, 'gradient_relative_max_error':relative,
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(), 'elapsed_seconds':time.time()-start_time,
              'scope':'single real-model selected physical output-head parameter; only its requires_grad enabled; controlled EOS-logit perturbation; no optimizer updates; not full-parameter ZeRO, accumulation, retention or efficacy validation'}
    with (root/'nonzero_gradient_result.json').open('x') as handle:
        json.dump(result,handle,indent=2)
    if magnitude == 0 or relative > 1e-3:
        raise RuntimeError('zero or mismatching physical-parameter gradient; aggregate retained')


if __name__ == '__main__':
    main()
