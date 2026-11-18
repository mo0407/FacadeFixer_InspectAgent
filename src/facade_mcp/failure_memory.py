"""Local, append-only failure observations; no inference, file discovery or network.

Callers must pass sanitized context and catalog summaries, never raw exceptions,
request/response bodies, credentials, or input image content. Quality events are
reviewer/evaluator declarations backed by references, not automatic ground truth.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import sqlite3
import time
import uuid
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Literal


_CATALOG = {
    "runtime": {
        "input_missing": ("输入缺失", "所需输入或已登记的输入引用不可用。", "核对输入引用、允许访问的目录及文件是否存在。"),
        "input_invalid": ("输入无效", "输入格式、尺寸或参数不满足当前流程要求。", "根据输入规范修正格式、尺寸及参数后重试。"),
        "source_changed": ("图像证据不一致", "原图、快照或专家结果的图像哈希校验不一致。", "重新建立只读输入快照并核对哈希。"),
        "playbook_changed": ("规则快照变化", "本轮冻结的 P0 规则在执行期间发生变化。", "恢复本轮规则快照并排查共享对象修改。"),
        "model_weights_missing": ("模型权重缺失", "专家模型所需的本地权重未找到。", "配置并核对对应模型权重，不要用空结果代替检测。"),
        "model_runtime_error": ("专家模型执行失败", "专家模型加载或推理未完成。", "检查模型依赖、设备和推理参数，保留本轮证据。"),
        "mcp_connection_error": ("MCP 连接失败", "无法完成专家 MCP 服务连接或工具调用。", "检查 MCP 启动命令、环境及服务状态。"),
        "api_timeout": ("模型请求超时", "远端请求超时，处理结果和计费状态尚不确定。", "先核对供应商请求记录，确认结果后再决定是否重试。"),
        "api_unauthorized": ("模型授权失败", "远端拒绝当前模型调用授权。", "核对本地密钥引用、模型权限与供应商配置。"),
        "api_rate_limited": ("模型请求限流", "远端服务因请求额度或速率限制拒绝请求。", "核对配额与限流窗口，安排受控重试。"),
        "api_http_error": ("模型 HTTP 错误", "远端返回未成功的 HTTP 状态。", "根据已脱敏状态码核查供应商记录及请求配置。"),
        "api_transport_error": ("模型传输失败", "模型请求未能完成网络传输。", "检查本地网络和端点配置，核对是否存在未确定的远端请求。"),
        "model_output_invalid": ("模型输出无效", "模型输出不满足约定的结构或字段限制。", "核对输出 schema 与提示词，拒绝不完整结果。"),
        "invalid_inspection_action": ("检查动作无效", "模型给出的锚点、专家或动作不满足本轮约束。", "使用已登记锚点和允许的模型，避免重复动作。"),
        "candidate_budget_exceeded": ("候选预算超限", "候选数量超过本轮允许的上限。", "核对检测阈值和候选预算，避免静默丢弃候选。"),
        "final_review_invalid": ("最终核验无效", "最终核验存在遗漏、重复或未知候选编号。", "逐候选重新核验，确保每个候选恰好一个结论。"),
        "unclassified_runtime_error": ("待分类运行异常", "执行未完成，但当前证据不足以确定原因。", "补充脱敏的阶段和证据引用，人工确认原因分类。"),
    },
    "quality": {
        "missed_defect": ("漏检", "评审或评价指出应识别的缺陷未进入最终结果。", "结合标注证据复查覆盖范围、阈值及局部检查策略。"),
        "false_positive": ("误检", "评审或评价指出结果包含非缺陷目标。", "对照证据复查材质、光照和上下文干扰。"),
        "wrong_class": ("缺陷分类错误", "评审或评价指出缺陷类别不符合标注。", "核对类别映射及该类别的区分规则。"),
        "bbox_inaccurate": ("定位框不准确", "评审或评价指出定位框范围或位置不准确。", "使用原图坐标证据复核边界及裁剪偏移。"),
        "mask_inaccurate": ("Mask 不准确", "评审或评价指出像素级区域未贴合缺陷。", "核对原像素证据与分割参数，修正掩膜并保留修改依据。"),
        "scope_error": ("检测范围错误", "评审或评价指出结果超出主建筑墙面范围。", "复核范围掩膜，排除门、窗、外楼梯及背景建筑。"),
        "report_error": ("报告错误", "评审或评价指出报告内容、证据对应或格式有误。", "对照候选编号、证据和报告模板修正内容。"),
        "duplicate_defect": ("缺陷重复", "评审或评价指出同一缺陷被重复计入结果。", "复核同类去重及成员追溯，保持一处缺陷的证据关联。"),
        "metric_below_target": ("指标未达标", "评价结果显示某项明确指标低于目标。", "核对指标定义、真值和阈值，再进行有证据的调优。"),
    },
    "observation": {
        "unknown_class": ("类别待映射", "发现未进入当前类别表的专家输出。", "保留原类别信息，人工确认映射后再判断。"),
        "scope_uncertain": ("范围待确认", "当前证据不足以确认候选位于目标墙面。", "补充原图或范围标注，保持待复核状态。"),
        "evidence_downsampled": ("证据经过降采样", "本轮核验展示的局部证据未保留全部原始像素。", "补充原像素局部证据后再确认细小缺陷。"),
        "harness_override": ("流程护栏调整结论", "流程约束将模型建议调整为待复核状态。", "查看护栏依据及对应证据，人工确认模型建议。"),
        "candidate_rejected": ("模型排除候选", "VLM 核验将候选排除；该结论不代表真值误报。", "保留核验证据和理由，必要时由人工评审确认。"),
    },
}

REASONS = MappingProxyType({code: MappingProxyType(dict(
    kind=kind, title=values[0], summary=values[1], recommended_action=values[2]))
    for kind, entries in _CATALOG.items() for code, values in entries.items()})

_SECRET = re.compile(
    r"(?:\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{8,}|\bgithub_pat_[A-Za-z0-9_]{12,}"
    r"|\bgh[pousr]_[A-Za-z0-9]{12,}|\bBearer\s+\S+"
    r"|\b(?:api[_-]?key|access[_-]?token|password|authorization)\s*[:=]\s*\S+)", re.I)
_PRIVATE_KEYS = {"apikey", "authorization", "token", "accesstoken", "refreshtoken",
                 "password", "secret", "exception", "rawexception", "traceback",
                 "headers", "requestbody", "responsebody", "rawerror"}


def _text(value, name, maximum, *, optional=False):
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    if any(ord(char) < 32 and char not in "\n\t" for char in value) or _SECRET.search(value):
        raise ValueError(f"{name} contains disallowed control characters or credential-like content")
    return value


def _safe_context(value):
    """Validate safe JSON only; never serialize arbitrary objects or exceptions."""
    if not isinstance(value, dict):
        raise ValueError("context must be a JSON object of sanitized fields")
    nodes = 0

    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        if nodes > 512 or depth > 6:
            raise ValueError("context exceeds structural limits")
        if isinstance(item, dict):
            for key, child in item.items():
                _text(key, "context key", 128)
                if re.sub(r"[^a-z0-9]", "", key.lower()) in _PRIVATE_KEYS:
                    raise ValueError("context must not contain credential or raw exception fields")
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, str):
            if item:
                _text(item, "context string", 2048)
        elif item is None or isinstance(item, bool):
            pass
        elif isinstance(item, (int, float)):
            try:
                finite = math.isfinite(item)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("context numbers must be finite")
        else:
            raise ValueError("context accepts JSON values only, not raw exceptions or objects")

    visit(value, 0)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 16384:
        raise ValueError("context exceeds 16384 UTF-8 bytes")
    return encoded


class FailureMemory:
    """Workspace-local SQLite event ledger with idempotent occurrence keys.

    Retries of the same (job_id, occurrence_key) return the first record unchanged.
    A different reason for that identity is an error, never an implicit update.
    """

    def __init__(self, workspace: Path):
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = workspace / "failure_memory.sqlite3"
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL,
                job_id TEXT NOT NULL, stage TEXT NOT NULL, reason_code TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('runtime','quality','observation')),
                occurrence_key TEXT NOT NULL, summary TEXT NOT NULL,
                evidence_refs TEXT NOT NULL, context TEXT NOT NULL, source TEXT NOT NULL,
                cause_status TEXT NOT NULL CHECK(cause_status IN ('observed','hypothesis','confirmed')),
                hypothesis TEXT, UNIQUE(job_id, occurrence_key))""")
            db.execute("CREATE INDEX IF NOT EXISTS events_reason ON events(reason_code, job_id)")
            db.execute("CREATE INDEX IF NOT EXISTS events_time ON events(recorded_at, event_id)")

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=30000")
        return db

    @staticmethod
    def _event(row):
        event = dict(row)
        event["evidence_refs"] = json.loads(event["evidence_refs"])
        event["context"] = json.loads(event["context"])
        reason = REASONS[event["reason_code"]]
        # Presentation fields always come from the catalog, never caller context.
        event.update(kind=reason["kind"], title=reason["title"],
                     recommended_action=reason["recommended_action"])
        return event

    def record(self, *, job_id: str, stage: str, reason_code: str, occurrence_key: str,
               summary: str | None = None, evidence_refs: list[str] | None = None,
               context: dict | None = None, source: str = "harness",
               cause_status: Literal["observed", "hypothesis", "confirmed"] = "observed",
               hypothesis: str | None = None) -> dict:
        job_id = _text(job_id, "job_id", 128)
        stage = _text(stage, "stage", 128)
        occurrence_key = _text(occurrence_key, "occurrence_key", 256)
        if not isinstance(reason_code, str) or reason_code not in REASONS:
            raise ValueError("Unknown reason_code; use the fixed REASONS catalog")
        reason = REASONS[reason_code]
        summary = _text(summary if summary is not None else reason["summary"], "summary", 2000)
        source = _text(source, "source", 64)
        hypothesis = _text(hypothesis, "hypothesis", 2000, optional=True)
        if cause_status not in ("observed", "hypothesis", "confirmed"):
            raise ValueError("Invalid cause_status")
        refs = [] if evidence_refs is None else evidence_refs
        if not isinstance(refs, list) or len(refs) > 64:
            raise ValueError("evidence_refs must be a list of at most 64 references")
        refs = [_text(ref, "evidence reference", 2048) for ref in refs]
        encoded_context = _safe_context({} if context is None else context)
        with closing(self._connect()) as db, db:
            # One writer owns identity lookup and insert; concurrent retries cannot race.
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM events WHERE job_id=? AND occurrence_key=?",
                                  (job_id, occurrence_key)).fetchone()
            if existing is not None:
                if existing["reason_code"] != reason_code:
                    raise ValueError("Occurrence already recorded with a different reason_code")
                return {**self._event(existing), "inserted": False}
            if reason["kind"] == "quality" and not refs:
                raise ValueError("Quality declarations require at least one evidence reference")
            event_id = uuid.uuid4().hex
            recorded_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            db.execute("""INSERT INTO events (event_id,recorded_at,job_id,stage,reason_code,kind,
                occurrence_key,summary,evidence_refs,context,source,cause_status,hypothesis)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                event_id, recorded_at, job_id, stage, reason_code, reason["kind"], occurrence_key,
                summary, json.dumps(refs, ensure_ascii=False, allow_nan=False), encoded_context,
                source, cause_status, hypothesis))
            return {**self._event(db.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()),
                    "inserted": True}

    def summary(self, job_id: str | None = None) -> dict:
        if job_id is not None:
            _text(job_id, "job_id", 128)
        with closing(self._connect()) as db, db:
            rows = db.execute("SELECT * FROM events" + (" WHERE job_id=?" if job_id is not None else "")
                              + " ORDER BY recorded_at DESC, event_id DESC",
                              (job_id,) if job_id is not None else ()).fetchall()
        events = [self._event(row) for row in rows]
        grouped = defaultdict(list)
        for event in events:
            grouped[event["reason_code"]].append(event)
        reasons = []
        for code, members in grouped.items():
            definition = REASONS[code]
            reasons.append(dict(reason_code=code, kind=definition["kind"], title=definition["title"],
                summary=definition["summary"], recommended_action=definition["recommended_action"],
                count=len(members), affected_jobs=len({m["job_id"] for m in members}),
                repeat_count=max(len(members)-1, 0),
                first_seen=min(m["recorded_at"] for m in members),
                last_seen=max(m["recorded_at"] for m in members)))
        reasons.sort(key=lambda reason: (-reason["count"], reason["reason_code"]))
        return dict(job_id=job_id, events=events, latest_event=events[0] if events else None,
            reasons=reasons, totals=dict(
                runtime_failures=sum(e["kind"] == "runtime" for e in events),
                quality_failures=sum(e["kind"] == "quality" for e in events),
                observations=sum(e["kind"] == "observation" for e in events),
                events=len(events), affected_jobs=len({e["job_id"] for e in events})),
            interpretation=dict(
                observation="观察项不计入失败；记录待确认现象或流程护栏行为。",
                quality="质量失败是 reviewer/evaluator 的有证据声明，不代表自动验证的真值。",
                repeat_count="同一原因的事件数减一；同一事件重试不增加计数，影响作业数单独统计。",
                cause_status="observed 为已观察现象，hypothesis 为待验证解释，confirmed 为调用方确认。"))

    def publish_summary(self, output: Path, job_id: str | None = None,
                        extra: dict | None = None) -> dict:
        """Publish a fresh JSON snapshot while excluding other ledger writers.

        WAL readers remain available. The write reservation covers the separate
        summary read and atomic replacement, so an older publisher cannot replace
        a newer writer's snapshot. Callers must use this method for a shared path.
        """
        if extra is None:
            extra = {}
        if not isinstance(extra, dict) or any(not isinstance(key, str) for key in extra):
            raise ValueError("extra must be a JSON object with string keys")
        protected = {"job_id", "events", "reasons", "totals", "latest_event", "interpretation"}
        if protected.intersection(extra):
            raise ValueError("extra must not override canonical summary fields")
        try:
            # Snapshot caller metadata before taking the database write lock.
            extras = json.loads(json.dumps(extra, ensure_ascii=False, allow_nan=False))
        except (TypeError, ValueError, OverflowError):
            raise ValueError("extra must contain finite JSON values only") from None
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with closing(self._connect()) as db, db:
                db.execute("BEGIN IMMEDIATE")
                data = {**self.summary(job_id), **extras}
                temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                                     encoding="utf-8")
                # Windows readers may briefly deny replacing an open file. Keep
                # the database reservation throughout this bounded local retry.
                deadline = time.monotonic() + 2
                while True:
                    try:
                        os.replace(temporary, output)
                        break
                    except PermissionError as error:
                        if (os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33)
                                or time.monotonic() >= deadline):
                            raise
                        time.sleep(.01)
                return data
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def export_html(self, output: Path, job_id: str | None = None) -> dict:
        """Export only this ledger's data; never open referenced evidence files."""
        data = self.summary(job_id)
        esc = lambda value: html.escape(str(value), quote=True)
        labels = {"runtime": "运行失败", "quality": "质量声明", "observation": "观察项"}
        frequency_rows = "".join(
            "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
                reason["title"], reason["reason_code"], labels[reason["kind"]], reason["count"],
                reason["affected_jobs"], reason["repeat_count"], reason["first_seen"],
                reason["last_seen"], reason["recommended_action"])) + "</tr>"
            for reason in data["reasons"])
        event_rows = "".join(
            "<tr>" + "".join(f"<td>{esc(value)}</td>" for value in (
                event["recorded_at"], event["job_id"], event["stage"], event["title"],
                labels[event["kind"]], event["cause_status"], event["source"], event["summary"],
                "；".join(event["evidence_refs"]), event["hypothesis"] or "",
                json.dumps(event["context"], ensure_ascii=False))) + "</tr>"
            for event in data["events"])
        totals = data["totals"]
        text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>失败经验统计</title>
<style>body{{font:15px/1.6 "Microsoft YaHei",sans-serif;color:#18243b;margin:32px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #d7dfe9;padding:9px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}th{{background:#eef3fb}}.scroll{{overflow-x:auto}}small{{color:#56647a}}</style>
<h1>失败经验统计</h1><p>范围：{esc(job_id if job_id is not None else '全部作业')}</p>
<p>运行失败 {totals['runtime_failures']} · 质量声明 {totals['quality_failures']} · 观察项 {totals['observations']} · 事件 {totals['events']} · 影响作业 {totals['affected_jobs']}</p>
<p>{esc(data['interpretation']['observation'])}<br>{esc(data['interpretation']['quality'])}<br>{esc(data['interpretation']['repeat_count'])}</p>
<h2>原因频次</h2><div class="scroll"><table><thead><tr><th>原因</th><th>代码</th><th>性质</th><th>事件数</th><th>影响作业数</th><th>重复次数</th><th>首次出现</th><th>最近出现</th><th>建议</th></tr></thead><tbody>{frequency_rows}</tbody></table></div>
<h2>事件记录（最新在前）</h2><div class="scroll"><table><thead><tr><th>时间</th><th>作业</th><th>阶段</th><th>原因</th><th>性质</th><th>原因状态</th><th>来源</th><th>描述</th><th>证据引用</th><th>待验证解释</th><th>安全上下文</th></tr></thead><tbody>{event_rows}</tbody></table></div>
<small>此页面仅展示本地已记录声明，不自动读取证据文件，不发起模型或网络请求。</small></html>'''
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return dict(path=str(output), totals=data["totals"], job_id=job_id)
