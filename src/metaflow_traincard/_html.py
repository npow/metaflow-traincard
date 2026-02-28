"""
Card HTML renderer.  render_card_html(state) -> HTML string.

The generated page is fully self-contained (Chart.js from CDN + inline CSS/JS)
so it renders correctly in the Metaflow UI iframe without any extra dependencies.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict

# fmt: off
_CHART_COLORS = [
    "#4f8ef7", "#f7874f", "#4fc6f7", "#a94ff7",
    "#f74f8e", "#4ff78e", "#f7d44f", "#f74f4f",
]
# fmt: on


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, r = divmod(seconds, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _phase_color(phase: str) -> str:
    return {
        "train": "#4f8ef7",
        "eval": "#a94ff7",
        "save": "#f7d44f",
        "done": "#4ff78e",
        "init": "#888888",
        "failed": "#f74f4f",
    }.get(phase, "#888888")


def _phase_label(phase: str) -> str:
    return {
        "train": "⚡ TRAINING",
        "eval": "📊 EVALUATING",
        "save": "💾 SAVING",
        "done": "✅ DONE",
        "init": "⏳ INIT",
        "failed": "💥 FAILED",
    }.get(phase, phase.upper())


def render_card_html(state: Dict[str, Any]) -> str:
    """
    Render a complete self-contained HTML card from a Reporter state dict.
    """
    elapsed = state.get("elapsed_seconds", time.time() - state.get("start_time", time.time()))
    phase = state.get("phase", "init")
    step = state.get("step", 0)
    epoch = state.get("epoch", 0)
    world_size = state.get("world_size", 1)
    rank = state.get("rank", 0)
    stalled = state.get("stalled", False)
    failure = state.get("failure")
    checkpoints = state.get("checkpoints", [])
    logs = state.get("logs", [])
    metrics = state.get("metrics", {})
    system = state.get("system", {})
    restart_count = state.get("restart_count", 0)

    if failure:
        phase = "failed"

    phase_color = _phase_color(phase)
    phase_label = _phase_label(phase)
    stall_banner = (
        '<div class="stall-banner">⚠️ Training appears stalled — no metric updates in 5+ minutes</div>'
        if stalled else ""
    )

    # ------------------------------------------------------------------
    # Chart data — one dataset per top-level metric name
    # ------------------------------------------------------------------
    chart_blocks = _build_chart_blocks(metrics)

    # ------------------------------------------------------------------
    # System section
    # ------------------------------------------------------------------
    system_html = _build_system_html(system)

    # ------------------------------------------------------------------
    # Checkpoints table
    # ------------------------------------------------------------------
    checkpoint_html = _build_checkpoints_html(checkpoints)

    # ------------------------------------------------------------------
    # Logs section
    # ------------------------------------------------------------------
    log_html = _build_log_html(logs)

    # ------------------------------------------------------------------
    # Failure section
    # ------------------------------------------------------------------
    failure_html = _build_failure_html(failure) if failure else ""

    # Distributed badge
    dist_badge = ""
    if world_size > 1:
        dist_badge = f'<span class="badge badge-dist">🔗 {world_size} GPUs</span>'

    resume_badge = ""
    if restart_count > 0:
        resume_badge = f'<span class="badge badge-resume">🔄 Restart #{restart_count}</span>'

    # HTML-escape < > & so the JSON is safe to embed in a <script> block
    state_json = (
        json.dumps(state, indent=None)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>TrainCard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #ffffff;
    --surface: #f8f9fa;
    --border: #e2e8f0;
    --text: #1a202c;
    --text-muted: #718096;
    --accent: {phase_color};
    --success: #4ff78e;
    --warning: #f7d44f;
    --error: #f74f4f;
    --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', Menlo, monospace;
    --radius: 8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 16px;
    font-size: 13px;
    line-height: 1.5;
  }}

  /* Status Header */
  .header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 18px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: var(--radius);
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .phase-badge {{
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--accent);
    white-space: nowrap;
  }}
  .header-stats {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    flex: 1;
  }}
  .stat {{
    display: flex;
    flex-direction: column;
  }}
  .stat-label {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-value {{ font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .badges {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }}
  .badge {{
    font-size: 10px; font-weight: 600;
    padding: 2px 8px; border-radius: 100px;
    white-space: nowrap;
  }}
  .badge-dist {{ background: #e8f0fe; color: #4f8ef7; }}
  .badge-resume {{ background: #fff8e0; color: #b7850f; }}

  /* Stall banner */
  .stall-banner {{
    background: #fff8e0;
    border: 1px solid #f7d44f;
    border-radius: var(--radius);
    padding: 10px 14px;
    margin-bottom: 16px;
    font-weight: 600;
    color: #7d5a00;
    font-size: 12px;
  }}

  /* Section */
  .section {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    margin-bottom: 16px;
    overflow: hidden;
  }}
  .section-header {{
    padding: 10px 16px;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border);
    background: var(--bg);
  }}
  .section-body {{ padding: 16px; }}

  /* Charts grid */
  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
  }}
  .chart-card {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }}
  .chart-title {{
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }}
  .chart-canvas-wrap {{ position: relative; height: 140px; }}

  /* System stats */
  .system-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 12px;
  }}
  .sys-stat {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
    text-align: center;
  }}
  .sys-stat-label {{ font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
  .sys-stat-value {{ font-size: 18px; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .sys-bar-wrap {{ margin-top: 6px; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }}
  .sys-bar {{ height: 100%; border-radius: 2px; transition: width 0.3s; }}
  .bar-ok    {{ background: #4ff78e; }}
  .bar-warn  {{ background: #f7d44f; }}
  .bar-crit  {{ background: #f74f4f; }}

  /* GPU card */
  .gpu-grid {{ display: flex; flex-direction: column; gap: 8px; }}
  .gpu-row {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    display: grid;
    grid-template-columns: 60px 1fr 60px;
    align-items: center;
    gap: 8px;
  }}
  .gpu-label {{ font-size: 11px; font-weight: 600; color: var(--text-muted); }}
  .gpu-bar-wrap {{ height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
  .gpu-bar {{ height: 100%; border-radius: 3px; }}
  .gpu-val {{ font-size: 12px; font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }}
  .no-gpu {{ color: var(--text-muted); font-size: 12px; text-align: center; padding: 12px; }}

  /* Checkpoints */
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
  }}
  td {{
    padding: 8px 8px;
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: var(--surface); }}
  .best-badge {{
    display: inline-block;
    background: #e8f9f0;
    color: #16a34a;
    font-size: 9px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 100px;
    margin-left: 6px;
    vertical-align: middle;
  }}
  .empty-state {{ color: var(--text-muted); font-size: 12px; text-align: center; padding: 20px; }}

  /* Logs */
  .log-container {{
    background: #1a1a2e;
    border-radius: 6px;
    padding: 12px;
    max-height: 250px;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.6;
  }}
  .log-line {{
    display: flex;
    gap: 12px;
    padding: 1px 0;
    word-break: break-all;
  }}
  .log-ts {{ color: #4a5568; white-space: nowrap; flex-shrink: 0; }}
  .log-text {{ color: #cbd5e0; flex: 1; }}
  .log-text.error {{ color: #fc8181; }}
  .log-text.warn  {{ color: #f6ad55; }}
  .log-search {{
    width: 100%;
    padding: 7px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 11px;
    margin-bottom: 8px;
    background: var(--bg);
    color: var(--text);
  }}

  /* Failure */
  .failure-section {{
    background: #fff5f5;
    border: 1px solid #feb2b2;
    border-radius: var(--radius);
    margin-bottom: 16px;
    overflow: hidden;
  }}
  .failure-header {{
    background: #f74f4f;
    color: white;
    padding: 10px 16px;
    font-weight: 700;
    font-size: 13px;
  }}
  .failure-body {{ padding: 16px; }}
  .failure-type {{ font-family: var(--font-mono); font-size: 13px; font-weight: 700; color: #c53030; margin-bottom: 6px; }}
  .failure-msg {{ font-family: var(--font-mono); font-size: 11px; color: #2d3748; margin-bottom: 12px; }}
  .traceback-toggle {{ font-size: 11px; color: #4f8ef7; cursor: pointer; text-decoration: underline; margin-bottom: 8px; }}
  .traceback {{
    display: none;
    background: #1a1a2e;
    color: #fc8181;
    border-radius: 6px;
    padding: 12px;
    font-family: var(--font-mono);
    font-size: 10px;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 300px;
    overflow-y: auto;
  }}
  .oom-warning {{
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
    color: #92400e;
    font-weight: 600;
    margin-top: 10px;
  }}
</style>
</head>
<body>

<!-- ─── Status Header ──────────────────────────────────────────────── -->
<div class="header">
  <div class="phase-badge">{phase_label}</div>
  <div class="header-stats">
    <div class="stat">
      <span class="stat-label">Step</span>
      <span class="stat-value">{step:,}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Epoch</span>
      <span class="stat-value">{epoch}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Elapsed</span>
      <span class="stat-value">{_fmt_duration(elapsed)}</span>
    </div>
  </div>
  <div class="badges">
    {dist_badge}
    {resume_badge}
  </div>
</div>

{stall_banner}
{failure_html}

<!-- ─── Training Metrics ───────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">Training Metrics</div>
  <div class="section-body">
    {chart_blocks if chart_blocks else '<div class="empty-state">No metrics recorded yet.</div>'}
  </div>
</div>

<!-- ─── System Telemetry ──────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">System Telemetry</div>
  <div class="section-body">{system_html}</div>
</div>

<!-- ─── Checkpoints ───────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">Checkpoints ({len(checkpoints)})</div>
  <div class="section-body">{checkpoint_html}</div>
</div>

<!-- ─── Logs ──────────────────────────────────────────────────────── -->
<div class="section">
  <div class="section-header">Logs (last {min(len(logs), 100)} lines)</div>
  <div class="section-body">
    <input class="log-search" type="text" id="log-search" placeholder="Filter logs…" oninput="filterLogs(this.value)"/>
    <div class="log-container" id="log-container">
      {_build_log_html(logs)}
    </div>
  </div>
</div>

<script>
// ─── Chart rendering ─────────────────────────────────────────────────
const CHART_COLORS = {json.dumps(_CHART_COLORS)};
const state = {state_json};

function buildCharts() {{
  if (!state.metrics) return;
  const metricNames = Object.keys(state.metrics);
  metricNames.forEach((name, idx) => {{
    const canvasId = 'chart-' + name.replace(/[^a-zA-Z0-9]/g, '_');
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const points = state.metrics[name] || [];
    const steps = points.map(p => p.step);
    const values = points.map(p => p.value);
    const color = CHART_COLORS[idx % CHART_COLORS.length];
    new Chart(canvas, {{
      type: 'line',
      data: {{
        labels: steps,
        datasets: [{{
          label: name,
          data: values,
          borderColor: color,
          backgroundColor: color + '18',
          borderWidth: 2,
          pointRadius: values.length > 200 ? 0 : 2,
          fill: true,
          spanGaps: false,
          tension: 0.3,
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            mode: 'index',
            intersect: false,
            callbacks: {{
              label: ctx => ctx.parsed.y !== null ? ctx.parsed.y.toFixed(6).replace(/\\.?0+$/, '') : 'N/A'
            }}
          }},
        }},
        scales: {{
          x: {{
            ticks: {{ maxTicksLimit: 6, font: {{ size: 10 }} }},
            grid: {{ color: 'rgba(0,0,0,0.05)' }},
            title: {{ display: true, text: 'step', font: {{ size: 9 }}, color: '#718096' }},
          }},
          y: {{
            ticks: {{ maxTicksLimit: 5, font: {{ size: 10 }} }},
            grid: {{ color: 'rgba(0,0,0,0.05)' }},
          }}
        }},
        animation: false,
      }}
    }});
  }});
}}
buildCharts();

// ─── Log filter ──────────────────────────────────────────────────────
function filterLogs(query) {{
  const lines = document.querySelectorAll('#log-container .log-line');
  const q = query.toLowerCase();
  lines.forEach(line => {{
    const text = line.querySelector('.log-text');
    line.style.display = (!q || (text && text.textContent.toLowerCase().includes(q))) ? '' : 'none';
  }});
}}

// ─── Traceback toggle ─────────────────────────────────────────────────
function toggleTraceback(id) {{
  const el = document.getElementById(id);
  el.style.display = el.style.display === 'none' || !el.style.display ? 'block' : 'none';
}}
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# Section builders
# ------------------------------------------------------------------

def _build_chart_blocks(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return ""
    charts = []
    for i, name in enumerate(metrics):
        safe_id = "chart-" + name.replace(" ", "_").replace("/", "_").replace(".", "_").replace("-", "_")
        charts.append(f"""<div class="chart-card">
  <div class="chart-title">{name}</div>
  <div class="chart-canvas-wrap"><canvas id="{safe_id}"></canvas></div>
</div>""")
    return f'<div class="charts-grid">{"".join(charts)}</div>'


def _build_system_html(system: Dict[str, Any]) -> str:
    if not system:
        return '<div class="empty-state">No system telemetry yet. Call reporter.system(stats) to add GPU/CPU stats.</div>'

    blocks = []

    # GPU section
    gpu_utils = system.get("gpu_utilization", [])
    gpu_mem = system.get("gpu_memory_used_gb", [])
    gpu_total = system.get("gpu_memory_total_gb", [])
    gpu_temp = system.get("gpu_temperature", [])

    if gpu_utils or gpu_mem:
        # Normalize to lists
        if isinstance(gpu_utils, (int, float)):
            gpu_utils = [gpu_utils]
        if isinstance(gpu_mem, (int, float)):
            gpu_mem = [gpu_mem]
        if isinstance(gpu_total, (int, float)):
            gpu_total = [gpu_total]
        if isinstance(gpu_temp, (int, float)):
            gpu_temp = [gpu_temp]

        gpu_rows = []
        for i in range(max(len(gpu_utils), len(gpu_mem))):
            util = gpu_utils[i] if i < len(gpu_utils) else None
            mem_used = gpu_mem[i] if i < len(gpu_mem) else None
            mem_tot = gpu_total[i] if i < len(gpu_total) else None
            temp = gpu_temp[i] if i < len(gpu_temp) else None

            bar_cls = "bar-ok"
            if util is not None and util > 90:
                bar_cls = "bar-warn"
            if util is not None and util < 30:
                bar_cls = "bar-crit"

            util_str = f"{util:.0f}%" if util is not None else "—"
            mem_str = f"{mem_used:.1f}G" if mem_used is not None else "—"
            if mem_tot:
                mem_str += f"/{mem_tot:.0f}G"
            temp_str = f"{temp:.0f}°C" if temp is not None else ""
            label = f"GPU {i}" + (f" {temp_str}" if temp_str else "")

            util_pct = util if util is not None else 0
            gpu_rows.append(f"""<div class="gpu-row">
  <div class="gpu-label">{label}</div>
  <div class="gpu-bar-wrap"><div class="gpu-bar {bar_cls}" style="width:{min(util_pct,100):.0f}%; background:{_bar_color(util_pct if util_pct else 0)}"></div></div>
  <div class="gpu-val">{util_str}</div>
</div>
<div class="gpu-row" style="margin-top:-10px; border-top: none;">
  <div class="gpu-label" style="font-size:9px;">VRAM</div>
  <div class="gpu-bar-wrap"><div class="gpu-bar" style="width:{_mem_pct(mem_used,mem_tot):.0f}%; background:#a94ff7"></div></div>
  <div class="gpu-val">{mem_str}</div>
</div>""")

        blocks.append(f"""<div style="margin-bottom:12px;">
  <div style="font-size:10px;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">GPU</div>
  <div class="gpu-grid">{"".join(gpu_rows)}</div>
</div>""")

    # CPU / RAM / Disk
    scalar_stats = []
    cpu = system.get("cpu_percent")
    ram_used = system.get("ram_used_gb")
    ram_total = system.get("ram_total_gb")
    disk_read = system.get("disk_read_mbps")
    disk_write = system.get("disk_write_mbps")

    if cpu is not None:
        bar_cls = "bar-ok" if cpu < 80 else "bar-warn"
        scalar_stats.append(_sys_stat_card("CPU", f"{cpu:.0f}%", cpu, bar_cls))

    if ram_used is not None:
        pct = _mem_pct(ram_used, ram_total)
        bar_cls = "bar-ok" if pct < 80 else "bar-warn"
        label = f"{ram_used:.1f}G" + (f"/{ram_total:.0f}G" if ram_total else "")
        scalar_stats.append(_sys_stat_card("RAM", label, pct, bar_cls))

    if disk_read is not None:
        scalar_stats.append(_sys_stat_card("Disk Read", f"{disk_read:.0f} MB/s", None, "bar-ok"))
    if disk_write is not None:
        scalar_stats.append(_sys_stat_card("Disk Write", f"{disk_write:.0f} MB/s", None, "bar-ok"))

    if scalar_stats:
        blocks.append(f'<div class="system-grid">{"".join(scalar_stats)}</div>')

    return "".join(blocks) if blocks else '<div class="empty-state">No telemetry data.</div>'


def _bar_color(pct: float) -> str:
    if pct >= 80:
        return "#4ff78e"
    if pct >= 40:
        return "#f7d44f"
    return "#f74f4f"


def _mem_pct(used, total) -> float:
    if used is None or not total:
        return 0.0
    return min((used / total) * 100.0, 100.0)


def _sys_stat_card(label: str, value: str, pct, bar_cls: str) -> str:
    bar_html = ""
    if pct is not None:
        bar_html = f'<div class="sys-bar-wrap"><div class="sys-bar {bar_cls}" style="width:{min(pct,100):.0f}%"></div></div>'
    return f"""<div class="sys-stat">
  <div class="sys-stat-label">{label}</div>
  <div class="sys-stat-value">{value}</div>
  {bar_html}
</div>"""


def _build_checkpoints_html(checkpoints) -> str:
    if not checkpoints:
        return '<div class="empty-state">No checkpoints saved yet.</div>'

    # Mark best = lowest loss from metadata if available
    rows = []
    best_idx = None
    for i, ckpt in enumerate(checkpoints):
        meta = ckpt.get("metadata", {})
        if "eval_loss" in meta or "val_loss" in meta:
            loss = meta.get("eval_loss", meta.get("val_loss", float("inf")))
            if best_idx is None or loss < checkpoints[best_idx].get("metadata", {}).get(
                "eval_loss", checkpoints[best_idx].get("metadata", {}).get("val_loss", float("inf"))
            ):
                best_idx = i

    for i, ckpt in enumerate(checkpoints):
        ts = ckpt.get("time", 0)
        t_str = _fmt_duration(time.time() - ts) + " ago" if ts else "—"
        path = ckpt.get("path", "—")
        step = ckpt.get("step", "—")
        step_str = f"{step:,}" if isinstance(step, int) else str(step)
        best_mark = '<span class="best-badge">BEST</span>' if i == best_idx else ""
        meta = ckpt.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k not in ("size_bytes",))
        size = meta.get("size_bytes")
        size_str = _fmt_size(size) if size else "—"
        rows.append(f"""<tr>
  <td>{path}{best_mark}</td>
  <td style="text-align:right">{step_str}</td>
  <td style="text-align:right">{size_str}</td>
  <td>{t_str}</td>
  <td style="color:#718096">{meta_str}</td>
</tr>""")

    return f"""<table>
<thead><tr>
  <th>Path</th><th style="text-align:right">Step</th>
  <th style="text-align:right">Size</th><th>Saved</th><th>Metadata</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>"""


def _fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _build_log_html(logs) -> str:
    if not logs:
        return ""
    recent = logs[-100:]
    lines = []
    for entry in recent:
        ts = entry.get("time", 0)
        t_str = time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""
        line = str(entry.get("line", "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        level = entry.get("level", "info").lower()
        level_cls = ""
        if "error" in level or "error" in line.lower() or "exception" in line.lower() or "traceback" in line.lower():
            level_cls = " error"
        elif "warn" in level or "warning" in line.lower():
            level_cls = " warn"
        lines.append(
            f'<div class="log-line"><span class="log-ts">{t_str}</span>'
            f'<span class="log-text{level_cls}">{line}</span></div>'
        )
    return "\n".join(lines)


def _build_failure_html(failure: Dict[str, Any]) -> str:
    if not failure:
        return ""
    exc_type = failure.get("type", "Unknown Error")
    message = failure.get("message", "")
    tb = failure.get("traceback", "")
    step = failure.get("step", "—")
    oom = failure.get("oom_suspected", False)

    tb_html = ""
    if tb:
        safe_tb = str(tb).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        tb_html = f"""<div class="traceback-toggle" onclick="toggleTraceback('tb')">▶ Show traceback</div>
<pre class="traceback" id="tb" style="display:none">{safe_tb}</pre>"""

    oom_html = ""
    if oom:
        oom_html = '<div class="oom-warning">⚠️ Out-of-memory suspected. Check GPU VRAM usage above or reduce batch size.</div>'

    return f"""<div class="failure-section">
  <div class="failure-header">💥 Training Failed at Step {step}</div>
  <div class="failure-body">
    <div class="failure-type">{exc_type}</div>
    <div class="failure-msg">{message}</div>
    {tb_html}
    {oom_html}
  </div>
</div>"""
