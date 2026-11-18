"""Synthetic wiring demo, not a real-building experiment or a learned correction."""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image

from facade_mcp.mask_evaluation import MaskEvaluation, evaluate_masks
from facade_mcp.model_gateway import ModelConfig
from facade_mcp.optimization import run_optimization
from facade_mcp.review_cycle import ReviewRequest


async def main(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    reference = np.zeros((32,32),dtype=np.uint8)
    reference[8:24,14:18] = 255
    Image.fromarray(reference).save(output / "reference.png")
    Image.fromarray(np.zeros_like(reference)).save(output / "prediction.png")
    spec = MaskEvaluation.model_validate({"dataset_version":"synthetic-demo-only-v1",
        "reference_approved_by":"synthetic-fixture-generator-not-human-annotation",
        "pairs":[{"instance_id":"synthetic-C01","prediction":"prediction.png","reference":"reference.png"}]})
    (output / "mask-request.json").write_text(spec.model_dump_json(indent=2),encoding="utf-8")

    def measure(round_id):
        result = evaluate_masks(spec,output)
        name = f"measurement-{round_id}.json"
        (output / name).write_text(json.dumps(result,indent=2),encoding="utf-8")
        return ReviewRequest.model_validate({"task":"Synthetic wiring verification only",
            "evidence":{name:json.dumps(result)},"metrics":[{"id":"macro_dice",
            "value":result["macro_dice"],"threshold":.9,"reference_verified":True,
            "evidence_ids":[name]}],"acceptance_version":"demo-only-not-production-targets",
            "playbook_version":"empty"})

    async def fixture_executor(proposal,current):
        # Intentional oracle copy to exercise controller wiring; never presented
        # as a segmentation algorithm, model improvement, or independent benchmark.
        Image.fromarray(reference).save(output / "prediction.png")
        return measure(1)

    config = ModelConfig.model_validate_json((Path(__file__).parent / "model-gateway.mock.json").read_text())
    state = await run_optimization(config,measure(0),output,fixture_executor,max_experiments=2)
    print(json.dumps({"synthetic_only":True,"oracle_fixture_executor":True,**state},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    asyncio.run(main(parser.parse_args().output.resolve()))
