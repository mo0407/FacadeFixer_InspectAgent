import asyncio
import json

import numpy as np
import pytest
from PIL import Image

from facade_mcp.mask_evaluation import MaskEvaluation, evaluate_masks
from facade_mcp.optimization import run_optimization
from test_model_gateway import config, request


def test_mask_metrics_and_provenance(tmp_path):
    Image.fromarray(np.array([[255, 255], [0, 0]], dtype=np.uint8)).save(tmp_path / "ref.png")
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(tmp_path / "pred.png")
    spec = MaskEvaluation.model_validate(dict(dataset_version="fixture", reference_approved_by="test",
        pairs=[dict(instance_id="C01",prediction="pred.png",reference="ref.png")]))
    result = evaluate_masks(spec,tmp_path)
    assert result["macro_dice"] == pytest.approx(2/3)
    assert result["macro_iou"] == .5
    assert len(result["instances"][0]["reference_sha256"]) == 64
    Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(tmp_path / "pred.png")
    assert evaluate_masks(spec,tmp_path)["macro_dice"] == 0
    Image.fromarray(np.zeros((2,2),dtype=np.uint8)).save(tmp_path / "ref.png")
    with pytest.raises(ValueError,match="Empty positive"):
        evaluate_masks(spec,tmp_path)


@pytest.mark.parametrize("array",[np.zeros((3,3),dtype=np.uint8),np.full((2,2),128,dtype=np.uint8),np.zeros((2,2,3),dtype=np.uint8)])
def test_invalid_masks(tmp_path,array):
    Image.fromarray(np.ones((2,2),dtype=np.uint8)).save(tmp_path / "ref.png")
    Image.fromarray(array).save(tmp_path / "pred.png")
    spec=MaskEvaluation.model_validate(dict(dataset_version="fixture",reference_approved_by="test",
        pairs=[dict(instance_id="x",prediction="pred.png",reference="ref.png")]))
    with pytest.raises(ValueError):evaluate_masks(spec,tmp_path)


def test_loop_rechecks_after_each_experiment(tmp_path):
    from pathlib import Path
    from facade_mcp.optimization import RoundResult
    from facade_mcp.optimization_dashboard import ModelSnapshot, RoundDetails

    calls=[]
    async def execute(proposal,current):
        calls.append(proposal)
        current.metrics[0].value += .1
        name = f'round-{len(calls)}.html'
        (Path(proposal['run_folder']) / name).write_text('<h1>Report</h1>', encoding='utf-8')
        return RoundResult(current, RoundDetails(title='近墙精检', report=name,
            models=[ModelSnapshot(stage='检测', model='YOLO12m', parameters={'推理尺寸': 1024})]))
    state=asyncio.run(run_optimization(config(),request(.5,.69,True),tmp_path,execute,max_experiments=3))
    assert state["status"]=="targets_met"
    assert state["experiments_executed"]==len(calls)==2
    assert state["experience_promoted"] is False
    dashboard = next(tmp_path.glob('optimization_*/dashboard.html')).read_text(encoding='utf-8')
    assert '候选规则' in dashboard and '第 2 轮测试' in dashboard
    assert '>+20 个百分点</span>' in dashboard and '>+10 个百分点</span>' in dashboard
    assert 'href="round-1.html"' in dashboard and 'href="round-2.html"' in dashboard
    assert 'YOLO12m' in dashboard and '推理尺寸 1024' in dashboard
    assert 'id="round-0" class="round" hidden' in dashboard
    assert 'id="round-2" class="round">' in dashboard
    assert '暂无本轮报告' in dashboard


def test_budget_is_not_success(tmp_path):
    async def execute(proposal,current):return current
    state=asyncio.run(run_optimization(config(),request(.5,.9,True),tmp_path,execute,max_experiments=2))
    assert state["status"]=="budget_exhausted" and state["experiments_executed"]==2


@pytest.mark.parametrize("value,threshold,status",[(1,.9,"targets_met"),(None,.9,"pending_review"),(.5,None,"pending_review")])
def test_terminal_conditions_do_not_execute(tmp_path,value,threshold,status):
    async def execute(*args):raise AssertionError("must not execute")
    state=asyncio.run(run_optimization(config(),request(value,threshold,True),tmp_path,execute,max_experiments=2))
    assert state["status"]==status and state["experiments_executed"]==0


def test_cannot_lower_threshold(tmp_path):
    async def execute(proposal,current):
        current.metrics[0].threshold=.1
        return current
    with pytest.raises(ValueError,match="frozen"):
        asyncio.run(run_optimization(config(),request(.5,.9,True),tmp_path,execute,max_experiments=2))
    state=json.loads(next(tmp_path.glob("optimization_*/state.json")).read_text())
    assert state["status"]=="failed"


def test_execution_failure_is_not_retried(tmp_path):
    calls=[]
    async def execute(*args):
        calls.append(1)
        raise RuntimeError("private error")
    with pytest.raises(RuntimeError):
        asyncio.run(run_optimization(config(),request(.5,.9,True),tmp_path,execute,max_experiments=3))
    assert len(calls)==1
    data=next(tmp_path.glob("optimization_*/state.json")).read_text()
    assert "private error" not in data
