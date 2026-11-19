from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from .storage import ArtifactStore
from .demo import create_demo
from .orchestration import run_agent, run_workflow, server_config


def load_request(path:Path)->dict:
    path = path.resolve()
    request = json.loads(path.read_text(encoding="utf-8-sig"))
    def resolve(value):
        value = str(value)
        return value if value.startswith("artifact:") else str((path.parent/Path(value)).resolve())
    request["image_paths"] = [resolve(value) for value in request.get("image_paths",[])]
    annotation = request.get("detection",{}).get("annotation_path")
    if annotation:
        request["detection"]["annotation_path"] = resolve(annotation)
    return request


def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(Path(os.environ.get("FACADE_CONFIG_FILE",str(Path.cwd()/".env"))),override=False)
    parser = argparse.ArgumentParser(description="建筑巡检 MCP + LangChain 第一版")
    parser.add_argument("--workspace",type=Path,default=Path(os.environ.get("FACADE_WORKSPACE","workspace")))
    parser.add_argument("--allow-root",action="append",type=Path,default=[])
    parser.add_argument("--mask-repo",type=Path,default=Path(os.environ["FACADE_MASK_REPO"]) if os.getenv("FACADE_MASK_REPO") else None)
    sub = parser.add_subparsers(dest="command",required=True)
    run = sub.add_parser("run",help="通过 LangGraph 和 MCP 执行配置中的完整流程")
    run.add_argument("--config",type=Path,required=True)
    demo = sub.add_parser("demo",help="无付费调用：合成照片＋已给检测框＋真实OpenCV＋报告")
    demo.add_argument("--fixture-dir",type=Path,default=Path("examples/generated_demo"))
    agent = sub.add_parser("agent",help="LangChain模型自主调度MCP，需要自己的模型配置")
    agent.add_argument("--prompt",required=True)
    agent.add_argument("--model")
    config = sub.add_parser("mcp-config",help="输出五个stdio服务的客户端配置（survey、mask、report、model、video）")
    config.add_argument("--output",type=Path)
    config.add_argument("--include-image",action="store_true",help="同时导出图像主动检查MCP入口")
    sub.add_parser("models",help="检查本地模型配置、权重和依赖（不执行模型推理）")
    model_check = sub.add_parser("model-config-check",help="校验三角色模型配置，不连接API")
    model_check.add_argument("--config",type=Path,required=True)
    image_inspect = sub.add_parser("image-inspect",help="单图锚定→最多两次原像素局部检查→VLM最终核验")
    image_inspect.add_argument("--config",type=Path,required=True)
    image_inspect.add_argument("--model-config",type=Path,required=True)
    image_inspect.add_argument("--playbook",type=Path)
    image_inspect.add_argument("--allow-api",action="store_true",help="允许将本轮概览与局部证据图发往配置的VLM")
    image_demo = sub.add_parser("image-demo",help="单图流程合成演示；无模型推理、无API调用")
    image_demo.add_argument("--output-dir",type=Path,default=Path("examples/generated_image_demo"))
    failure_summary = sub.add_parser("failure-summary",help="查询失败原因、重复次数及证据；可导出HTML")
    failure_summary.add_argument("--job-id")
    failure_summary.add_argument("--output",type=Path)
    failure_feedback = sub.add_parser("failure-feedback",help="提交带同任务证据的人工质量问题反馈")
    failure_feedback.add_argument("--config",type=Path,required=True)
    rule_latest=sub.add_parser('rule-latest',help='查看已验证发布、供下一轮加载的P0')
    for name in ('rule-propose','rule-validate','rule-cycle'):
        lifecycle=sub.add_parser(name,help='离线生成/验证/发布规则；检查中不修改P0')
        lifecycle.add_argument('--model-config',type=Path,required=True)
        lifecycle.add_argument('--allow-api',action='store_true')
        if name in ('rule-propose','rule-cycle'):
            lifecycle.add_argument('--playbook',type=Path)
            lifecycle.add_argument('--job-id',action='append')
        if name in ('rule-validate','rule-cycle'):
            lifecycle.add_argument('--dataset',type=Path,required=True)
        if name=='rule-validate':
            lifecycle.add_argument('--candidate',required=True)
    vision_check=sub.add_parser('vision-check',help='用合成编号色块验证真实视觉与结构化动作能力')
    vision_check.add_argument('--model-config',type=Path,required=True)
    vision_check.add_argument('--output',type=Path)
    vision_check.add_argument('--allow-api',action='store_true')
    review = sub.add_parser("review-cycle",help="运行一次三角色评审；不执行视频检测或自动调优")
    review.add_argument("--config",type=Path,required=True)
    review.add_argument("--request",type=Path,required=True)
    review.add_argument("--playbook",type=Path,help="带版本和稳定条目ID的已采用规则快照")
    review.add_argument("--allow-api",action="store_true",help="明确允许向配置的API发送本次评审文本")
    mask_eval = sub.add_parser("evaluate-masks",help="计算人工配对的预测mask与参考mask的像素指标")
    mask_eval.add_argument("--request",type=Path,required=True)
    mask_eval.add_argument("--output",type=Path,required=True)
    video_prepare = sub.add_parser("video-prepare",help="默认2fps分批抽帧，可加密近墙关键时段")
    video_prepare.add_argument("--video",type=Path,required=True)
    video_prepare.add_argument("--building-id",required=True)
    video_prepare.add_argument("--sample-fps",type=float,default=2.0)
    video_prepare.add_argument("--keyframe-range",type=float,nargs=2,action="append",metavar=("START","END"))
    video_prepare.add_argument("--keyframe-fps",type=float,default=4.0)
    video_prepare.add_argument("--max-frames",type=int,default=2000)
    video_detect = sub.add_parser("video-detect",help="复核墙面输入→本地检测/分割→逐帧候选报告草稿")
    video_detect.add_argument("--config",type=Path,required=True)
    dashboard = sub.add_parser("optimization-dashboard",help="导出每轮指标、改善和规则的HTML看板")
    dashboard.add_argument("--state",type=Path,required=True)
    dashboard.add_argument("--output",type=Path,required=True)
    dashboard.add_argument("--playbook",type=Path)
    args = parser.parse_args()
    if args.command == "optimization-dashboard":
        from .optimization_dashboard import export_dashboard
        playbook = json.loads(args.playbook.read_text(encoding="utf-8-sig")) if args.playbook else None
        export_dashboard(args.state, args.output, playbook)
        print(json.dumps({"dashboard": str(args.output.resolve())},ensure_ascii=False))
        return
    if args.command == "evaluate-masks":
        from .mask_evaluation import MaskEvaluation, evaluate_masks
        spec = MaskEvaluation.model_validate(json.loads(args.request.read_text(encoding="utf-8-sig")))
        result = evaluate_masks(spec,args.request.resolve().parent)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    if args.command in ("model-config-check","review-cycle"):
        from .model_gateway import ModelConfig
        from .review_cycle import ReviewRequest, run_review
        settings = ModelConfig.model_validate(json.loads(args.config.read_text(encoding="utf-8-sig")))
        if args.command == "model-config-check":
            result = {"valid":True,"roles":settings.roles,"api_enabled":settings.api_enabled,
                      "network_called":False,"capabilities_verified":False}
        else:
            request = ReviewRequest.model_validate(json.loads(args.request.read_text(encoding="utf-8-sig")))
            from .playbook import Playbook
            playbook = Playbook.model_validate_json(args.playbook.read_text(encoding="utf-8-sig")) if args.playbook else None
            result = asyncio.run(run_review(settings,request,args.workspace,allow_api=args.allow_api,playbook=playbook))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    roots = [Path.cwd(),*[Path(value) for value in os.environ.get("FACADE_ALLOWED_INPUT_ROOTS","").split(";") if value.strip()],*args.allow_root]
    if args.command in ("run", "video-detect", "image-inspect"):
        roots.append(args.config.resolve().parent)
    if args.command == "image-demo":
        roots.append(args.output_dir.resolve())
    if args.command == "demo":
        roots.append(args.fixture_dir.resolve())
    if args.command in ('rule-validate','rule-cycle'):
        roots.append(args.dataset.resolve().parent)
    store = ArtifactStore(args.workspace,roots)
    if args.command=='rule-latest':
        from .rule_lifecycle import RuleLifecycle
        book=RuleLifecycle(store).latest()
        result={'playbook':book.model_dump(mode='json') if book else None}
    elif args.command in ('rule-propose','rule-validate','rule-cycle'):
        from .rule_lifecycle import RuleLifecycle
        from .model_gateway import ModelConfig
        from .playbook import Playbook
        lifecycle=RuleLifecycle(store)
        settings=ModelConfig.model_validate_json(args.model_config.read_text(encoding='utf-8-sig'))
        async def run_rules():
            if args.command in ('rule-propose','rule-cycle'):
                parent=Playbook.model_validate_json(args.playbook.read_text(encoding='utf-8-sig')) if args.playbook else None
                proposal=await lifecycle.propose(settings,parent=parent,job_ids=args.job_id,allow_api=args.allow_api)
                if args.command=='rule-propose':
                    return proposal
                candidate=proposal['candidate_playbook_path']
            else:
                candidate=args.candidate
            return await lifecycle.validate_and_promote(candidate,str(args.dataset.resolve()),settings,allow_api=args.allow_api)
        result=asyncio.run(run_rules())
    elif args.command=='vision-check':
        from .model_gateway import ModelConfig, validate_vision_connection
        settings=ModelConfig.model_validate_json(args.model_config.read_text(encoding='utf-8-sig'))
        result=asyncio.run(validate_vision_connection(settings,args.output or store.root/'vision-validation.json',allow_api=args.allow_api))
    elif args.command == "failure-summary":
        from .failure_memory import FailureMemory
        memory=FailureMemory(store.root)
        result=memory.summary(job_id=args.job_id)
        if args.output:
            result['dashboard']=memory.export_html(args.output,job_id=args.job_id)
    elif args.command == "failure-feedback":
        from .image_failures import FailureFeedback, record_feedback
        feedback=FailureFeedback.model_validate_json(args.config.read_text(encoding="utf-8-sig"))
        result=record_feedback(store,feedback)
    elif args.command == "image-inspect":
        from .image_inspection import ImageInspectionRequest, run_image_inspection
        from .model_gateway import ModelConfig
        from .playbook import Playbook
        from .image_failures import record_preflight_failure
        try:
            data=json.loads(args.config.read_text(encoding="utf-8-sig"))
            for key in ("image_path","scope_mask_path"):
                if isinstance(data.get(key),str) and not data[key].startswith("artifact:"):
                    data[key]=str((args.config.resolve().parent/data[key]).resolve())
            settings=ModelConfig.model_validate_json(args.model_config.read_text(encoding="utf-8-sig"))
            playbook=Playbook.model_validate_json(args.playbook.read_text(encoding="utf-8-sig")) if args.playbook else None
            request=ImageInspectionRequest.model_validate(data)
        except Exception as error:
            record_preflight_failure(store,error)
            raise
        result=asyncio.run(run_image_inspection(request,store,settings,playbook,allow_api=args.allow_api))
    elif args.command == "image-demo":
        from .image_demo import run_image_demo
        result=asyncio.run(run_image_demo(args.output_dir,store))
    elif args.command == "video-detect":
        from .video_detection_workflow import VideoDetectionRequest, run_video_detection
        data=json.loads(args.config.read_text(encoding="utf-8-sig"))
        if not data["wall_inputs"].startswith("artifact:"):
            data["wall_inputs"]=str((args.config.resolve().parent/ data["wall_inputs"]).resolve())
        result=asyncio.run(run_video_detection(VideoDetectionRequest.model_validate(data),store))
    elif args.command == "video-prepare":
        from .orchestration import run_video_prepare
        result = asyncio.run(run_video_prepare(str(args.video.resolve()),args.building_id,store,args.sample_fps,args.max_frames,
                                               args.keyframe_range,args.keyframe_fps))
    elif args.command == "mcp-config":
        result = server_config(store.root,roots,args.mask_repo)
        if args.include_image:
            result["mcpServers"]["image"]={**result["mcpServers"]["model"],"args":["-m","facade_mcp.image_server"]}
        # Config contains only explicit relevant env; never export process secrets.
        keep = {"FACADE_WORKSPACE","FACADE_ALLOWED_INPUT_ROOTS","FACADE_MASK_REPO","FACADE_CONFIG_FILE",
                "FACADE_REPORT_REPO","FACADE_MASK_MODELS_CONFIG","FACADE_DETECTOR_URL","FACADE_DETECTOR_MODEL",
                "FACADE_DETECTOR_PYTHON","FACADE_DETECTOR_WEIGHTS","FACADE_DETECTOR_DEVICE",
                "FACADE_SOURCE_REPO","FACADE_SOURCE_PYTHON","FACADE_SOURCE_DEVICE","FACADE_SOURCE_TIMEOUT_SECONDS",
                "FACADE_EXPERTS_CONFIG","FACADE_EXPERTS_ALLOWED_ROOTS",
                "FACADE_SELECTOR_URL","FACADE_SELECTOR_MODEL",
                "PYTHONPATH","PYTHONIOENCODING","FASTMCP_SHOW_SERVER_BANNER","FASTMCP_CHECK_FOR_UPDATES"}
        for server in result["mcpServers"].values():
            server["env"] = {key:value for key,value in server["env"].items() if key in keep}
        if args.output:
            store.write_json(args.output,result)
    elif args.command == "demo":
        result = asyncio.run(run_workflow(create_demo(args.fixture_dir),store,args.mask_repo))
    elif args.command == "run":
        result = asyncio.run(run_workflow(load_request(args.config),store,args.mask_repo))
    elif args.command == "models":
        from .model_service import ModelService
        result = ModelService(store).list_tools()
    else:
        result = asyncio.run(run_agent(args.prompt,store,args.mask_repo,args.model))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
