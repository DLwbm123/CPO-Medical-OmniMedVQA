"""Repeated *same-objective* CUDA backward control; no optimizer or tuning.

This does not repeat the failed cross-objective gate or change its result.
"""
import json
import os
from pathlib import Path
import time
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .core import decision_deficits, token_margins
from .decision_loss import aligned_response_logits
from .multimodal import make_inputs


def main():
    start = time.time()
    source, out = Path(os.environ['SOURCE']), Path(os.environ['OUT'])
    processor = AutoProcessor.from_pretrained(source/'stage1/model',local_files_only=True,min_pixels=128*28*28,max_pixels=512*28*28)
    # Trusted private artifact produced by this project, never a remote upload.
    item, reference = torch.load(source/'private_memory.pt',map_location='cpu',weights_only=False)[0]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(source/'stage1/model',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    for p in model.parameters():p.requires_grad_(False)
    parameter = model.lm_head.weight
    parameter.requires_grad_(True)
    inputs = make_inputs(processor,item,device='cuda')
    response_start = inputs['input_ids'].shape[1]
    response = torch.tensor([reference['response_ids']],device='cuda')
    if int(response[0,-1]) != reference['eos_id']:raise RuntimeError('EOS mismatch')
    inputs['input_ids'] = torch.cat((inputs['input_ids'],response),dim=1)
    inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    tau = torch.tensor([[.5*min(v,2.) for v in reference['margins']]],device='cuda')
    results = []
    for deterministic in (False,True):
        torch.use_deterministic_algorithms(deterministic)
        record = {'deterministic_algorithms':deterministic,'repetitions':3}
        try:
            logits, targets = aligned_response_logits(model,inputs,response_start)
            z = logits.float()
            mask = torch.ones_like(targets,dtype=torch.bool)
            baseline = decision_deficits(z,targets,mask,tau)
            if float(baseline.max()) > 1e-4:raise RuntimeError('baseline mismatch')
            idx, target = targets.shape[1]-1,int(targets[0,-1])
            _, indices = z[0,idx].topk(2)
            competitor = int(indices[1] if int(indices[0]) == target else indices[0])
            offset = (token_margins(z,targets)[0,idx].detach()+tau[0,idx]).detach()
            modified = z.clone()
            modified[0,idx,competitor] = z[0,idx,competitor]+offset
            loss = decision_deficits(modified,targets,mask,tau).sum()
            first_input = first_parameter = None
            input_errors, parameter_errors = [],[]
            magnitude = 0.
            for _ in range(3):
                upstream = torch.autograd.grad(loss,z,retain_graph=True)[0]
                gradient = torch.autograd.grad(loss,parameter,retain_graph=True)[0]
                if not torch.isfinite(gradient).all():raise RuntimeError('nonfinite gradient')
                if first_input is None:
                    first_input, first_parameter = upstream.detach(), gradient.detach()
                    magnitude = float(gradient.float().abs().max())
                else:
                    input_errors.append(float((upstream-first_input).abs().max()))
                    parameter_errors.append(float((gradient.float()-first_parameter.float()).abs().max()))
            record.update(status='measured',loss=float(loss),parameter_gradient_max_abs=magnitude,
                          input_vjp_repeat_max_abs_errors=input_errors,parameter_repeat_max_abs_errors=parameter_errors,
                          parameter_repeat_relative_max_errors=[v/max(magnitude,1e-12) for v in parameter_errors])
            del logits,z,baseline,modified,loss,upstream,gradient,first_input,first_parameter
        except RuntimeError as error:
            record.update(status='runtime_error',error_type=type(error).__name__)
            with (out/('deterministic_'+str(deterministic)+'_error.txt')).open('x') as handle:handle.write(str(error))
        results.append(record)
    torch.use_deterministic_algorithms(False)
    with (out/'result.json').open('x') as handle:
        json.dump({'conditions':results,'elapsed_seconds':time.time()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                   'scope':'same-objective repetition only; selected real tied parameter; no cross-objective equivalence retest, optimizer, model updates, tuning or efficacy claim; original failed gate remains failed'},handle,indent=2)


if __name__ == '__main__':main()
