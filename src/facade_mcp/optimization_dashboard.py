"""Local, self-contained view over optimization state and rule snapshots."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from os.path import relpath
from urllib.parse import quote

from pydantic import Field
from .model_gateway import Strict


class ModelSnapshot(Strict):
    stage: str
    model: str
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RoundDetails(Strict):
    title: str
    report: str | None = None
    report_status: str = '待复核'
    report_summary: str | None = None
    models: list[ModelSnapshot] = Field(default_factory=list)


def _report_link(value: str | None, folder: Path, output_folder: Path) -> str | None:
    if not value:
        return None
    path = (folder / value).resolve()
    if path.is_relative_to(folder.resolve()) and path.suffix.lower() == '.html' and path.is_file():
        try:
            return quote(Path(relpath(path, output_folder.resolve())).as_posix())
        except ValueError:  # Windows drives can differ.
            return path.as_uri()
    return None


_LABELS = {'macro_dice':'Mask Dice', 'macro_iou':'Mask IoU', 'recall':'召回率',
           'precision':'精确率', 'pass':'达标', 'fail':'未达标', 'pending':'待验收',
           'targets_met':'已达标', 'budget_exhausted':'未达标 · 已到轮次上限',
           'pending_review':'待验收', 'running':'进行中', 'executing':'正在执行', 'failed':'执行失败'}
_RATE_METRICS = {'recall', 'precision'}


def _label(value):
    return _text(_LABELS.get(value, value))


def _metric_value(key, value):
    return f'{value:.2%}'.replace('.00%', '%') if key in _RATE_METRICS and value is not None else _number(value)


def _change(key, value):
    return f'{value * 100:+.6g} 个百分点' if key in _RATE_METRICS else f'{value:+.6g}'


def _text(value):
    return escape(str(value)) if value is not None else '—'


def _number(value):
    return '—' if value is None else f'{value:.6g}'


def _workflow_diagram():
    roles = (
        ('Generator', '方案生成', '当前结果、验收目标和规则库版本', '本轮计划',
         '<rect x="5" y="3" width="14" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/>'),
        ('Reflector', '失败复盘', '计划、指标和证据', '失败原因假设、建议调整',
         '<circle cx="10" cy="10" r="5.5"/><path d="m14 14 6 6"/>'),
        ('Curator', '规则整理', '复盘结论和证据', '候选规则及适用条件',
         '<path d="m4 8 8-4 8 4-8 4-8-4Zm0 4 8 4 8-4M4 16l8 4 8-4"/>'),
    )
    cards = []
    for name, title, inputs, outputs, icon in roles:
        cards.append(f"""
        <li><div class="role-title">
          <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor"
               stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{icon}</svg>
          <div><strong>{title}</strong><small>{name}</small></div>
        </div><dl><dt>输入</dt><dd>{inputs}</dd><dt>输出</dt><dd>{outputs}</dd></dl></li>""")
    return f"""
    <details class="workflow">
      <summary><strong>查看调优流程与三个模型角色</strong></summary>
      <div class="workflow-body">
        <h2>调优流程图</h2>
        <ol class="flow">
          <li><strong>视频分析与报告</strong><span>抽帧 → 检测 → 分割 → 报告</span></li>
          <li><strong>程序评估</strong><span>将本轮指标与目标比较</span></li>
          <li><strong>未达标且可继续</strong><span>三角色评审 → 调整方案 → 复测</span></li>
        </ol>
        <p class="flow-loop">↶ 复测结果返回「程序评估」，继续比较同一套验收目标。</p>
        <ul class="flow-outcomes">
          <li><strong>达标</strong> · 结束本次调优</li>
          <li><strong>缺少依据</strong> · 等待人工验收</li>
          <li><strong>轮次用尽</strong> · 记录未达标项</li>
        </ul>
        <h2>三个模型角色</h2>
        <ol class="flow">{''.join(cards)}</ol>
        <p class="note">三个角色可分别配置模型 API。候选规则通过独立验证和审核后，才进入已采用规则库。</p>
      </div>
    </details>"""


def render_dashboard(state: dict, folder: Path, playbook: dict | None = None,
                     output_folder: Path | None = None) -> str:
    history = state['history']
    baseline = {}
    previous = {}
    sections = []
    navigation = []
    candidate_rules = []
    selected = len(history) - 1
    headline = ''
    for index, entry in enumerate(history):
        metrics = entry['evaluation']['metrics']
        current = {m['metric_id']: m['value'] for m in metrics if m['status'] != 'pending'}
        if index == 0:
            baseline = current
        directions = {m['id']: m['direction'] for m in entry['request']['metrics']}
        rows = []
        for metric in metrics:
            key = metric['metric_id']
            sign = 1 if directions[key] == 'min' else -1

            def delta(reference):
                if index == 0 or key not in reference or key not in current:
                    return '—'
                change = sign * (current[key] - reference[key])
                kind = 'gain' if change > 0 else 'loss' if change < 0 else 'same'
                return f'<span class="{kind}">{_change(key, change)}</span>'

            rows.append(f'<tr><th scope="row">{_label(key)}</th><td>{_metric_value(key, metric["value"])}</td>'
                        f'<td>{"≥" if sign == 1 else "≤"} {_metric_value(key, metric["threshold"])}</td>'
                        f'<td>{_label(metric["status"])}</td><td>{delta(previous)}</td><td>{delta(baseline)}</td></tr>')
        previous = current
        primary = next((m for m in metrics if m.get('required')), None)
        metric_summary = (f'{_label(primary["metric_id"])} {_metric_value(primary["metric_id"], primary["value"])}'
                          if primary else '')
        if index == selected and primary is not None:
            key = primary['metric_id']
            target = f' / 目标 {_metric_value(key, primary["threshold"])}' if primary['threshold'] is not None else ''
            headline = f'<span>{metric_summary}{target}</span>'
            if index and key in current and key in baseline:
                sign = 1 if directions[key] == 'min' else -1
                change = sign * (current[key] - baseline[key])
                headline += f'<span class="{"gain" if change > 0 else "loss" if change < 0 else "same"}">{_change(key, change)} · 较基线</span>'
        review = ''
        if entry.get('review_output'):
            # Only read sibling review artifacts produced inside this run.
            name = Path(entry['review_output']).name
            review_folder = (folder / name).resolve()
            if name.startswith('review_') and review_folder.parent == folder.resolve():
                for filename, label in [('reflection.json', '下一轮调整建议'), ('candidate_experience.json', '候选规则 · 待验证')]:
                    path = review_folder / filename
                    if path.is_file() and path.resolve().parent == review_folder:
                        data = json.loads(path.read_text(encoding='utf-8'))
                        if filename == 'candidate_experience.json':
                            candidate_rules.extend(data.get('candidates', []))
                        else:
                            insights = ''.join(f'<li>{_text(item.get("controlled_change"))}</li>' for item in data.get('insights', []))
                            if insights:
                                review += f'<details><summary>{label}</summary><ul>{insights}</ul></details>'
        title = '基线' if index == 0 else f'第 {_text(entry["experiment"])} 轮测试'
        detail = RoundDetails.model_validate(entry['details']) if entry.get('details') else None
        subtitle = f' · {_text(detail.title)}' if detail else ''
        href = _report_link(detail.report, folder, output_folder or folder) if detail else None
        report_action = f'<a class="report-link" target="_blank" rel="noopener" href="{_text(href)}">打开完整报告 ↗</a>' if href else ''
        report_status = _text(detail.report_status) if href and detail else '暂无本轮报告'
        summary = f'<p>{_text(detail.report_summary)}</p>' if detail and detail.report_summary else ''
        model_rows = ''
        if detail and detail.models:
            for model in detail.models:
                params = ' · '.join(f'{_text(k)} {_text(v)}' for k, v in model.parameters.items())
                model_rows += f'<li><strong>{_text(model.stage)}</strong><span>{_text(model.model)}</span><span>{params}</span></li>'
        models = f'<ul class="models">{model_rows}</ul>' if model_rows else '<p class="muted">本轮未记录模型与参数。</p>'
        active = index == selected
        navigation.append(f'<button type="button" data-round="{index}" aria-pressed="{str(active).lower()}" aria-controls="round-{index}">'
                          f'<strong>{title}</strong><span>{_text(detail.title) if detail else ""}</span>'
                          f'<span>{metric_summary}</span></button>')
        sections.append(f'<article id="round-{index}" class="round"{"" if active else " hidden"}>'
                        f'<h2>{title}{subtitle}</h2><section class="report"><div><h3>本轮报告</h3><span class="muted">{report_status}</span>{summary}</div>{report_action}</section>'
                        '<section><h3>本轮指标</h3><div class="scroll"><table><thead><tr>'
                        '<th>指标</th><th>本轮值</th><th>目标</th><th>状态</th><th>较上一轮改善</th><th>较基线改善</th>'
                        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
                        f'<section><h3>模型与参数</h3>{models}</section>{review}</article>')
    rules = '<p class="muted">暂无已采用规则快照。</p>'
    if playbook is not None:
        if not isinstance(playbook.get('rules'), list) or not isinstance(playbook.get('version'), str):
            raise ValueError('Playbook snapshot requires version and rules')
        expected = history[0]['request']['playbook_version'] if history else None
        if expected is not None and playbook['version'] != expected:
            raise ValueError('Playbook snapshot version does not match the run')
        items = ''.join(_rule(rule, '已采用') for rule in playbook['rules'])
        rules = f'<p>版本 {_text(playbook["version"])}</p><ul>{items}</ul>' if items else f'<p>版本 {_text(playbook["version"])}，规则列表为空。</p>'
    if candidate_rules:
        rules += '<ul>' + ''.join(_rule(rule, '候选规则 · 待验证') for rule in candidate_rules) + '</ul>'
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FacadeFixer 调优看板</title>
<style>
*{{box-sizing:border-box}}body{{margin:auto;max-width:1440px;padding:24px;font:16px system-ui,sans-serif;color:#14213a;background:#f3f6fb}}
h1,h2,h3{{margin:0 0 14px}}h1{{font-size:24px}}h2{{font-size:22px}}h3{{font-size:18px}}header{{display:flex;gap:20px;align-items:baseline}}
.summary{{display:flex;flex-wrap:wrap;align-items:center;gap:12px 24px;margin:12px 0 18px;padding:18px 22px;background:#e0ebff;border-radius:12px}}
.summary strong{{font-size:24px}}.layout{{display:grid;grid-template-columns:minmax(245px,1fr) minmax(0,3fr);gap:16px;align-items:start}}
section,.round,aside{{background:white;padding:16px;border:1px solid #d8e1ef;border-radius:12px}}.round section+section{{margin-top:12px}}
nav button{{font:inherit;display:block;text-align:left;width:100%;padding:12px;margin-bottom:8px;color:inherit;background:white;border:1px solid #d8e1ef;border-radius:8px;cursor:pointer}}
nav button span{{display:block;color:#495972}}nav button[aria-pressed="true"]{{border-color:#244bd9;background:#eef3ff}}
table{{border-collapse:collapse;width:100%;text-align:left}}th,td{{padding:9px;border-bottom:1px solid #d8e1ef;white-space:nowrap}}thead{{background:#eef3ff}}
.muted,.note{{color:#495972}}details{{margin-top:10px}}.scroll{{overflow:auto}}.gain{{color:#214fbd;font-weight:700}}.loss{{color:#a32437;font-weight:700}}
.report{{display:flex;justify-content:space-between;gap:10px;align-items:center}}.report-link{{background:#2349dc;color:white;padding:9px 12px;border-radius:8px;text-decoration:none}}
.rules{{margin-top:20px;border-top:1px solid #d8e1ef;padding-top:16px}}ul{{margin:0;padding-left:20px}}li{{margin:8px 0}}
.pill{{display:inline-block;background:#e0ebff;border-radius:6px;padding:2px 6px;font-size:13px}}.pending{{background:#fff0d0;color:#754a10}}
.models{{list-style:none;padding:0}}.models li{{display:grid;grid-template-columns:110px minmax(110px,1fr) 2fr;gap:10px;padding:6px;border-bottom:1px solid #e7ecf4}}
.workflow{{margin:0 0 16px;background:#fff;border:1px solid #d8e1ef;border-radius:12px}}
.workflow>summary{{cursor:pointer;padding:12px 16px;color:#214fbd}}
.workflow-body{{padding:6px 16px 14px}}.workflow-body h2{{font-size:18px}}
.flow{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:26px;list-style:none;padding:0}}
.flow li{{position:relative;margin:0;padding:12px;background:#f3f6ff;border:1px solid #d8e1ef;border-radius:8px}}
.flow li:not(:last-child)::after{{content:'→';position:absolute;left:calc(100% + 5px);top:16px;color:#214fbd}}
.flow li>span,.role-title small{{display:block;color:#495972;font-size:14px;margin-top:4px}}
.flow-loop{{color:#214fbd;margin:10px 0}}.flow-outcomes{{display:flex;flex-wrap:wrap;gap:6px 18px;list-style:none;padding:0;margin:0 0 14px}}
.flow-outcomes li{{margin:0;padding:4px 8px;background:#f4f6f9;border-radius:6px;font-size:14px}}
.role-title{{display:flex;align-items:center;gap:9px}}.role-title svg{{color:#214fbd;flex:none}}
.flow dl{{display:grid;grid-template-columns:auto 1fr;gap:5px 10px;margin:10px 0 0;font-size:14px}}.flow dt{{font-weight:600}}.flow dd{{margin:0}}
@media(max-width:760px){{body{{padding:12px}}.layout{{grid-template-columns:1fr}}.models li{{display:block}}.models span{{display:block}}}}
@media(max-width:620px){{.flow{{grid-template-columns:1fr;gap:20px}}.flow li:not(:last-child)::after{{content:'↓';left:16px;top:100%}}}}
</style><header><strong>FacadeFixer</strong><h1>调优看板</h1>{'<span class="pill">演示数据</span>' if state.get('synthetic_only') else ''}</header><main>
<div class="summary"><strong>{_label(state['status'])}</strong><span>已执行 {_text(state['experiments_executed'])} 轮</span>{headline}</div>
{_workflow_diagram()}
<div class="layout"><aside><h2>测试轮次</h2><nav aria-label="测试轮次">{''.join(navigation)}</nav>
<div class="rules"><h2>规则库</h2>{rules}</div></aside><div>{''.join(sections) or '<p>暂无测试结果。</p>'}</div></div>
<p class="note">正数表示改善。缺少确认依据时显示 —。</p></main>
<script>
for (const button of document.querySelectorAll('[data-round]')) {{
  button.addEventListener('click', () => {{
    for (const other of document.querySelectorAll('[data-round]')) {{
      const active = other === button;
      other.setAttribute('aria-pressed', String(active));
      document.getElementById(other.getAttribute('aria-controls')).hidden = !active;
    }}
  }});
}}
</script></html>'''


def _rule(rule, status):
    if not isinstance(rule, dict):
        rule = {'recommendation': rule}
    pending = ' pending' if status != '已采用' else ''
    description = ''.join(f'<p>{_text(rule[key])}</p>' for key in ('condition', 'limitations') if rule.get(key))
    return (f'<li><details><summary>{_text(rule.get("recommendation"))} '
            f'<span class="pill{pending}">{status}</span></summary>{description}</details></li>')


def export_dashboard(state_path: Path, output: Path, playbook: dict | None = None):
    state = json.loads(state_path.read_text(encoding='utf-8-sig'))
    page = render_dashboard(state, state_path.parent, playbook, output.parent)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding='utf-8')
