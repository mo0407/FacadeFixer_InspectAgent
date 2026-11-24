"""Remote-safe CLI for Figure-1 matching and resumable candidate P0 creation.

Run with ``python -m facade_mcp.baseline_p0_cli``.  This entry point never
launches detector inference or training and never promotes a candidate rule.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from .baseline_p0 import match_baseline_batch
from .baseline_p0_coverage import propose_full_coverage_p0
from .model_gateway import ModelConfig
from .storage import ArtifactStore


def _parser():
    parser=argparse.ArgumentParser(description='Match the complete Figure-1 batch and propose a nonpublished P0')
    parser.add_argument('--workspace',type=Path,default=Path(os.environ.get('FACADE_WORKSPACE','workspace')))
    parser.add_argument('--allow-root',type=Path,action='append',default=[])
    commands=parser.add_subparsers(dest='command',required=True)
    match=commands.add_parser('match',help='Match a completed Figure-1 batch to explicit polygon references')
    match.add_argument('--summary',type=Path,required=True)
    match.add_argument('--labels',type=Path,required=True)
    match.add_argument('--class-map',required=True,help='Explicit reference ID to canonical class JSON, e.g. {"0":"C","1":"S"}')
    match.add_argument('--expected-count',type=int,default=203)
    match.add_argument('--iou',type=float,default=.5)
    declaration=match.add_mutually_exclusive_group()
    declaration.add_argument('--reference-complete',action='store_true',
        help='Only after reference quality review: declare annotated classes exhaustive for FP scoring')
    declaration.add_argument('--assume-reference-complete',action='store_true',
        help='Conditionally treat valid annotated classes as exhaustive, without asserting independent review')
    curate=commands.add_parser('curate',help='Reflect and curate every eligible mismatch into an unregistered P0 candidate')
    curate.add_argument('--match-batch',type=Path,required=True)
    curate.add_argument('--model-config',type=Path,required=True)
    curate.add_argument('--expected-count',type=int,default=203)
    curate.add_argument('--chunk-size',type=int,default=6)
    curate.add_argument('--max-calls',type=int,default=1024)
    curate.add_argument('--allow-api',action='store_true',help='Allow configured VLM requests with image crops')
    curate.add_argument('--allow-incomplete',action='store_true',
        help='Explicitly produce partial_reference_review if reference issues remain; never publish')
    curate.add_argument('--acknowledge-unknown',action='store_true',
        help='Acknowledge a prior provider request of unknown outcome before retrying it')
    return parser


def main(argv=None):
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    from dotenv import load_dotenv
    load_dotenv(Path(os.environ.get('FACADE_CONFIG_FILE',str(Path.cwd()/'.env'))),override=False)
    args=_parser().parse_args(argv)
    env_roots=[Path(x) for x in os.environ.get('FACADE_ALLOWED_INPUT_ROOTS','').split(';') if x.strip()]
    store=ArtifactStore(args.workspace,[Path.cwd(),*env_roots,*args.allow_root])
    if args.command=='match':
        try:
            class_map=json.loads(args.class_map)
        except json.JSONDecodeError:
            raise ValueError('class-map must be valid JSON') from None
        assumption=('For the specified classes, valid supplied annotations are provisionally treated as exhaustive; '
                    'this is an unreviewed assumption, not evidence of annotation completeness.'
                    if args.assume_reference_complete else None)
        result=match_baseline_batch(store,str(args.summary),str(args.labels),class_map,
            expected_count=args.expected_count,iou_threshold=args.iou,
            reference_complete=args.reference_complete or args.assume_reference_complete,
            reference_assumption=assumption)
        status=result['status']
        if status=='completed' and not result['reference_complete']:
            status='needs_completeness_declaration'
        result={'status':status,'match_batch_path':result['match_batch_path'],
            'matched_count':result['matched_count'],'expected_count':result['expected_count'],
            'counts':result['counts'],'simulation':result['simulation'],
            'reference_complete':result['reference_complete'],
            'reference_assumption':result.get('reference_assumption'),
            'reference_review_cases':[{'image_path':x['image_path'],'match_path':x['match_path']}
                                      for x in result['cases'] if x['status']!='completed'],
            'note':'No VLM, model training or rule publication was performed.'}
    else:
        config_path=store.resolve_input(str(args.model_config))
        settings=ModelConfig.model_validate_json(config_path.read_text(encoding='utf-8-sig'))
        result=asyncio.run(propose_full_coverage_p0(store,settings,str(args.match_batch),
            allow_api=args.allow_api,chunk_size=args.chunk_size,expected_count=args.expected_count,
            max_calls=args.max_calls,allow_incomplete=args.allow_incomplete,
            acknowledge_unknown=args.acknowledge_unknown))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
    return 0 if result['status'] in ('completed','candidate_only') else 2


if __name__=='__main__':
    sys.exit(main())
