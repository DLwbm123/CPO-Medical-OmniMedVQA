# Controlled actual-model gradient check: numeric gate not passed

Used the unchanged stage-one Qwen2.5-VL model and first stored eligible MedBook training reference. The unperturbed decision deficit was zero. A detached EOS-competitor logit intervention produced deficit 2.0, matching the explicit single-token objective. The selected physical output-head parameter had a finite nonzero maximum gradient of 120.

However, the two algebraically equivalent objectives' actual bf16 model gradients differed: maximum absolute difference 0.5 and relative maximum error 0.0041667. This exceeds the **predeclared 0.001** criterion, so the check exited 1 and is not marked passed. Reduced-precision arithmetic could contribute, but that explanation has not been established; no threshold was relaxed after seeing the outcome. This does not establish a semantic method failure or scientific effectiveness.

The first attempt failed at the same assertion without retaining the numeric aggregate. An instrumentation repair saved aggregate values before asserting; rerun used the identical sample, weights, intervention and tolerance, the original absolute extension deadline, original two-hour parent clock and 46GB output guard. Both failure records are retained privately. The targeted extension performed no optimizer updates or weight writes, used one GPU, and is not a rerun of the completed two-stage training.

Scope is deliberately limited: one physical parameter's gradient through an actual model with only that parameter trainable. There is no full-parameter ZeRO, accumulation, optimizer or continual-performance validation here. Private samples, memory content and model weights are excluded.
