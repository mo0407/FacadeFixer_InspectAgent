"""Explicit, resumable CLI for the Figure-1 expert and VLM stages.

Example (expert-only; no VLM):
    python -m facade_mcp.figure1_runner --images-dir datasets/CUBIT-InSeg/test/images \
        --expected-count 203 --workspace workspace --output workspace/figure1-run --experts-only \
        --source-repo . --source-python /path/to/model/python --experts-config model_assets/cubit-experts.local.json

The model MCP receives explicit source paths, avoiding stale machine-specific
values from a copied .env. A complete real VLM run requires both a model config
and an explicit --allow-api flag. This entry point never starts training and
does not load annotation files.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager, contextmanager
import json
import os
from pathlib import Path
from typing import Iterator

from .baseline import BaselineConfig, run_baseline_batch, run_baseline_experts_batch
from .model_gateway import ModelConfig, ModelGateway
from .orchestration import ToolGateway, server_config


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}


def _positive_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected count must be a positive integer') from None
    if count < 1:
        raise argparse.ArgumentTypeError('expected count must be a positive integer')
    return count


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Run the explicitly configured Figure-1 experiment')
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--images-dir', type=Path, help='Directory of JPEG/PNG images (not recursive)')
    inputs.add_argument('--image', type=Path, action='append', help='Image path; repeat to select several')
    p.add_argument('--expected-count', type=_positive_count, required=True,
                   help='Exact expected image count; use 203 for the requested batch')
    p.add_argument('--workspace', type=Path, default=Path('workspace'))
    p.add_argument('--output', type=Path, required=True,
                   help='Experiment output directory under workspace; stable across resume')
    p.add_argument('--source-repo', type=Path, required=True, help='Server-owned builtin model source/weights root')
    p.add_argument('--source-python', type=Path, required=True, help='Python for the builtin detector')
    p.add_argument('--experts-config', type=Path, required=True, help='Server-owned expert registry JSON')
    p.add_argument('--experts-allowed-root', type=Path, action='append', default=[],
                   help='Extra allowed roots for expert registry paths; repeat as needed')
    p.add_argument('--device', default='cuda:0', help='Model device, e.g. cpu or cuda:0')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--experts-only', action='store_true', help='Run or resume local experts only; never invoke a VLM')
    mode.add_argument('--complete', action='store_true', help='Complete baseline using a real explicitly configured VLM')
    p.add_argument('--model-config', type=Path, help='Model gateway JSON; required only with --complete')
    p.add_argument('--allow-api', action='store_true', help='Explicitly enable the configured VLM call')
    p.add_argument('--confidence-floor', type=float, default=.001)
    p.add_argument('--decision-batch-size', type=_positive_count, default=8)
    p.add_argument('--retry-failed', action='store_true', help='Explicitly retry previously failed jobs')
    p.add_argument('--acknowledge-unknown', action='store_true',
                   help='Acknowledge an unresolved prior VLM request before retry')
    return p


def _images(args) -> list[Path]:
    if args.images_dir:
        directory = args.images_dir.expanduser().resolve()
        if not directory.is_dir():
            raise ValueError('Image directory not found')
        paths = [path.resolve() for path in sorted(directory.iterdir(), key=lambda p: p.name)
                 if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    else:
        paths = [path.expanduser().resolve() for path in args.image]
        if any(not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES for path in paths):
            raise ValueError('Every selected image must be an existing JPEG or PNG')
    if len(paths) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} images; found {len(paths)}')
    if len(set(paths)) != len(paths):
        raise ValueError('Selected image paths must be unique')
    return paths


def _prepare(args):
    if args.experts_only and (args.model_config or args.allow_api or args.acknowledge_unknown):
        raise ValueError('Expert-only mode does not accept a VLM configuration or API flags')
    if args.complete and (args.model_config is None or not args.allow_api):
        raise ValueError('Completing a baseline requires --model-config and --allow-api')
    if args.acknowledge_unknown and not args.retry_failed:
        raise ValueError('Acknowledging an unknown request also requires --retry-failed')
    images = _images(args)
    workspace = args.workspace.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(workspace) or output == workspace:
        raise ValueError('Output must be a subdirectory of workspace')
    if any(image.is_relative_to(output) for image in images):
        raise ValueError('Output directory cannot contain input images')
    source_repo = args.source_repo.expanduser().resolve()
    source_python = args.source_python.expanduser().resolve()
    experts_config = args.experts_config.expanduser().resolve()
    if not source_repo.is_dir() or not source_python.is_file() or not os.access(source_python, os.X_OK):
        raise ValueError('Builtin model source root or executable Python not found')
    if not experts_config.is_file():
        raise ValueError('Expert registry JSON not found')
    config = BaselineConfig(expected_count=args.expected_count, confidence_floor=args.confidence_floor,
                            decision_batch_size=args.decision_batch_size)
    model_config = None
    if args.complete:
        model_config = ModelConfig.model_validate_json(args.model_config.expanduser().resolve().read_text(encoding='utf-8-sig'))
        connection = model_config.connections[model_config.roles['generator']]
        if not model_config.api_enabled or connection.adapter == 'mock' or not connection.expected_model:
            raise ValueError('A real enabled model with an expected model identity is required')
    model_env = {'FACADE_SOURCE_REPO':str(source_repo), 'FACADE_SOURCE_PYTHON':str(source_python),
                 'FACADE_EXPERTS_CONFIG':str(experts_config), 'FACADE_SOURCE_DEVICE':args.device,
                 'FACADE_EXPERTS_ALLOWED_ROOTS':';'.join(str(p.expanduser().resolve()) for p in args.experts_allowed_root)}
    return images, workspace, output, config, model_config, model_env


@contextmanager
def _output_lock(output: Path) -> Iterator[None]:
    """Keep concurrent processes from corrupting resumable experiment journals."""
    if os.name != 'posix':
        raise RuntimeError('Figure-1 batch execution currently requires POSIX file locking')
    import fcntl

    if output.exists() and not output.is_dir():
        raise ValueError('Experiment output exists and is not a directory')
    def reject_unrelated_contents():
        allowed_uninitialized = {'.figure1.lock', 'tool_calls.jsonl'}
        files = {p.name for p in output.iterdir()}
        if files - allowed_uninitialized and not ((output/'expert-manifest.json').is_file() or
                                                   (output/'baseline-manifest.json').is_file()):
            raise ValueError('Refusing to write in a nonempty directory without an experiment manifest')

    if output.is_dir():
        reject_unrelated_contents()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.figure1.lock').open('a+', encoding='utf-8')
    acquired = False
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            raise RuntimeError('Another Figure-1 runner is using this output directory') from None
        reject_unrelated_contents()
        yield
    finally:
        if acquired:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _model_server_config(workspace: Path, roots: list[Path], model_env: dict[str, str]) -> dict:
    config = server_config(workspace, roots)
    config['mcpServers'] = {'model': config['mcpServers']['model']}
    env = config['mcpServers']['model']['env']
    # The detector subprocess needs local model configuration, not the VLM
    # credential inherited by the parent process for a later review stage.
    for name in list(env):
        if any(part in name.upper() for part in ('API_KEY', 'ACCESS_TOKEN', 'SECRET', 'PASSWORD', 'AUTHORIZATION')):
            env.pop(name, None)
    env.update(model_env)
    return config


@asynccontextmanager
async def _gateway(workspace: Path, roots: list[Path], trace_path: Path, model_env: dict[str, str]):
    from fastmcp.client.transports import MCPConfigTransport
    from langchain.mcp import MCPAdapter

    config = _model_server_config(workspace, roots, model_env)
    async with MCPAdapter(MCPConfigTransport(config, name_as_prefix=False)) as adapter:
        yield ToolGateway(await adapter.list_tools(), trace_path)


async def run(args) -> dict:
    images, workspace, output, config, model_config, model_env = _prepare(args)
    with _output_lock(output):
        roots = sorted({workspace, *(image.parent for image in images)}, key=str)
        async with _gateway(workspace, roots, output/'tool_calls.jsonl', model_env) as gateway:
            capabilities = await gateway.call('facade_models_capabilities', {})
            if args.experts_only:
                return await run_baseline_experts_batch(images, output, gateway, capabilities, config,
                                                        retry_failed=args.retry_failed)
            vlm = ModelGateway(model_config, allow_api=True)
            return await run_baseline_batch(images, output, gateway, vlm, capabilities, config,
                                            simulation=False, retry_failed=args.retry_failed,
                                            acknowledge_unknown=args.acknowledge_unknown)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    # This matches the standard CLI: explicit environment variables take priority.
    from dotenv import load_dotenv
    env_file = Path(os.environ.get('FACADE_CONFIG_FILE', str(Path.cwd()/'.env')))
    if env_file.is_file():
        load_dotenv(env_file, override=False)
    try:
        summary = asyncio.run(run(args))
    except Exception as error:
        # Error details may include caller inputs; do not print secret-bearing
        # configuration, provider bodies or a traceback in this entry point.
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}, ensure_ascii=False))
        return 1
    completed, failed = summary.get('completed', []), summary.get('failed', [])
    result = {'status': summary.get('status', 'completed' if not failed else 'failed'),
              'completed': len(completed), 'failed': len(failed),
              'expected_count': args.expected_count, 'simulation': summary.get('simulation'),
              'output': str(args.output.resolve()), 'halted_reason': summary.get('halted_reason')}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if len(completed) == args.expected_count and not failed and not summary.get('halted_reason') else 1


if __name__ == '__main__':
    raise SystemExit(main())
