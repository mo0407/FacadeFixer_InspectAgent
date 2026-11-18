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


def main():
    server.run()


if __name__ == '__main__':
    main()
