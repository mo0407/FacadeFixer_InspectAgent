"""Prepare and run the independent, non-publishing P0-v2 revision."""
from __future__ import annotations
import argparse
import asyncio
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import sys

from .model_gateway import ModelConfig
from .p0_v2 import prepare_revision, run_revision
from .storage import ArtifactStore


def _parser():
    parser=argparse.ArgumentParser(description='Prepare or run the auditable P0 v2 candidate')
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--allow-root',type=Path,action='append',default=[])
    commands=parser.add_subparsers(dest='command',required=True)
    for action in ('prepare','run'):
        sub=commands.add_parser(action)
        sub.add_argument('--parent-manifest',type=Path,required=True)
        sub.add_argument('--parent-sha256',required=True)
        sub.add_argument('--model-config',type=Path,required=True)
        sub.add_argument('--expected-count',type=int,default=203)
        sub.add_argument('--allow-incomplete',action='store_true')
        if action=='run':
            sub.add_argument('--expected-revision-id',required=True)
            sub.add_argument('--allow-api',action='store_true')
            sub.add_argument('--max-new-calls',type=int,default=1024)
    return parser


def _lock(stack,root):
    for name in ('.figure1-p0-bootstrap.lock','figure1-p0-retry-supervisor.lock',
                 'figure1-p0-budget-supervisor.lock','figure1-p0-generic-launch.lock',
                 'figure1-p0-semantic-launch.lock','figure1-p0-v2.lock'):
        fd=os.open(root/name,os.O_RDWR|os.O_CREAT|getattr(os,'O_NOFOLLOW',0),0o600)
        stack.callback(os.close,fd)
        try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another P0 process holds an execution lock') from None


def main(argv=None):
    args=_parser().parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(os.environ.get('FACADE_CONFIG_FILE',str(Path.cwd()/'.env'))),override=False)
        roots=[Path.cwd(),*args.allow_root]
        roots += [Path(x) for x in os.environ.get('FACADE_ALLOWED_INPUT_ROOTS','').split(';') if x.strip()]
        store=ArtifactStore(args.workspace,roots)
        settings=ModelConfig.model_validate_json(store.resolve_input(str(args.model_config)).read_text(encoding='utf-8-sig'))
        if args.command=='run' and not args.allow_api:
            raise ValueError('Explicit --allow-api is required')
        with ExitStack() as stack:
            _lock(stack,store.root)
            manifest,folder,verified=prepare_revision(store,settings,args.parent_manifest,args.parent_sha256,
                expected_count=args.expected_count,allow_incomplete=args.allow_incomplete,
                expected_revision_id=args.expected_revision_id if args.command=='run' else None)
            if args.command=='prepare':
                state={'status':'prepared','revision_id':manifest['revision_sha256'],
                    'revision_dir':str(folder),'progress_path':str(folder/'progress.json'),
                    'completed_batches':len(verified['prefix']),'total_batches':len(manifest['batches']),
                    'accounted_event_count':sum(len(g['event_ids']) for g in manifest['batches'][:len(verified['prefix'])]),
                    'total_event_count':sum(len(g['event_ids']) for g in manifest['batches']),
                    'reused_v1_phase_count':manifest['reuse_v1_phase_count'],'new_calls_this_run':0,
                    'usage':{'reported_input_tokens':0,'reported_output_tokens':0,'reported_total_tokens':0,'calls_without_complete_usage':0}}
            else:
                if args.expected_revision_id!=manifest['revision_sha256']:
                    raise ValueError('Expected revision does not match the frozen input')
                state=asyncio.run(run_revision(store,settings,manifest,folder,verified,max_new_calls=args.max_new_calls))
        print(json.dumps(state,ensure_ascii=False,allow_nan=False))
        return 0 if state['status'] in ('prepared','candidate_only') else 2
    except Exception as exc:
        # Never print exception bodies: network/provider details may be sensitive.
        print(json.dumps({'status':'error','error_type':type(exc).__name__,
                          'revision_id':None,'revision_dir':None,'progress_path':None}))
        return 1


if __name__=='__main__':
    sys.exit(main())
