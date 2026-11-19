"""Optional high-level image workflow MCP; model inference remains in model MCP."""
from __future__ import annotations

from fastmcp import FastMCP

from .image_inspection import ImageInspectionRequest, run_image_inspection
from .model_gateway import ModelConfig
from .playbook import Playbook
from .storage import ArtifactStore


server = FastMCP('FacadeFixer Image Active Inspection')


@server.tool
async def facade_image_inspect(request: dict, model_config_path: str,
                               playbook_path: str | None = None, allow_api: bool = False) -> dict:
    """Inspect ONE still PNG/JPEG; max 2 original-pixel crops, frozen P0, draft report.

    model_config_path/playbook_path must be readable under allowed roots. Remote
    VLM calls require both api_enabled in the config and allow_api=True. No video
    sampling or automatic rule promotion occurs in this tool.
    """
    store = ArtifactStore.from_env()
    from .image_failures import record_preflight_failure
    try:
        settings = ModelConfig.model_validate_json(store.resolve_input(model_config_path).read_text(encoding='utf-8-sig'))
        playbook = Playbook.model_validate_json(store.resolve_input(playbook_path).read_text(encoding='utf-8-sig')) if playbook_path else None
        validated=ImageInspectionRequest.model_validate(request)
    except Exception as error:
        record_preflight_failure(store,error)
        raise
    return await run_image_inspection(validated,store,settings,playbook,allow_api=allow_api)


@server.tool
def facade_image_failure_summary(job_id: str | None = None, output_html: bool = False) -> dict:
    """Read persistent failure reasons/counts; observations never count as failures."""
    from .failure_memory import FailureMemory
    store = ArtifactStore.from_env()
    memory = FailureMemory(store.root)
    result = memory.summary(job_id=job_id)
    if output_html:
        if job_id:
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',job_id):
                raise ValueError('Invalid job ID for HTML export')
            output=store.root/f'failure-dashboard-{job_id}.html'
        else:
            output=store.root/'failure-dashboard.html'
        result['dashboard']=memory.export_html(output,job_id=job_id)
    return result


@server.tool
def facade_image_record_feedback(feedback: dict) -> dict:
    """Record a reviewer-declared quality failure with existing job-local evidence.

    Reusing a job/occurrence key is idempotent; this never changes P0 or reruns models.
    """
    from .image_failures import FailureFeedback, record_feedback
    return record_feedback(ArtifactStore.from_env(),FailureFeedback.model_validate(feedback))


@server.tool
def facade_image_rules_latest() -> dict:
    """Read the immutable validated P0 selected automatically for the next run."""
    from .rule_lifecycle import RuleLifecycle
    book=RuleLifecycle(ArtifactStore.from_env()).latest()
    return {'playbook':book.model_dump(mode='json') if book else None}


@server.tool
async def facade_image_rule_propose(model_config_path: str, playbook_path: str | None = None,
                                   job_ids: list[str] | None = None, allow_api: bool = False) -> dict:
    """Summarize historical non-mock failures into a cited candidate P0; does not adopt it."""
    from .rule_lifecycle import RuleLifecycle
    store=ArtifactStore.from_env()
    settings=ModelConfig.model_validate_json(store.resolve_input(model_config_path).read_text(encoding='utf-8-sig'))
    parent=Playbook.model_validate_json(store.resolve_input(playbook_path).read_text(encoding='utf-8-sig')) if playbook_path else None
    return await RuleLifecycle(store).propose(settings,parent=parent,job_ids=job_ids,allow_api=allow_api)


@server.tool
async def facade_image_rule_validate(candidate_path: str, dataset_path: str,
                                    model_config_path: str, allow_api: bool = False) -> dict:
    """Replay baseline/candidate on independent labeled images and promote only after gates pass."""
    from .rule_lifecycle import RuleLifecycle
    store=ArtifactStore.from_env()
    settings=ModelConfig.model_validate_json(store.resolve_input(model_config_path).read_text(encoding='utf-8-sig'))
    return await RuleLifecycle(store).validate_and_promote(candidate_path,dataset_path,settings,allow_api=allow_api)


@server.tool
async def facade_image_rule_cycle(dataset_path: str, model_config_path: str,
                                 job_ids: list[str] | None = None, allow_api: bool = False) -> dict:
    """Run one bounded OFFLINE failure→proposal→paired evaluation→promotion cycle."""
    from .rule_lifecycle import RuleLifecycle
    store=ArtifactStore.from_env()
    settings=ModelConfig.model_validate_json(store.resolve_input(model_config_path).read_text(encoding='utf-8-sig'))
    lifecycle=RuleLifecycle(store)
    proposal=await lifecycle.propose(settings,job_ids=job_ids,allow_api=allow_api)
    return await lifecycle.validate_and_promote(proposal['candidate_playbook_path'],dataset_path,settings,allow_api=allow_api)


@server.tool
async def facade_image_validate_vision(model_config_path: str, allow_api: bool = False) -> dict:
    """Test remote vision/action JSON with synthetic color-ID evidence; mock never proves capability."""
    from .model_gateway import validate_vision_connection
    store=ArtifactStore.from_env()
    settings=ModelConfig.model_validate_json(store.resolve_input(model_config_path).read_text(encoding='utf-8-sig'))
    return await validate_vision_connection(settings,store.root/'vision-validation.json',allow_api=allow_api)


def main():
    server.run()


if __name__ == '__main__':
    main()
