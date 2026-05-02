"""
dashboard.py
============

Generate a clean, focused HTML dashboard from experiment results.

The dashboard answers two questions:
  1. At the same matrix size, which algorithm is fastest?
  2. How does adding cores affect execution time?

All comparisons are fair: same n, same input. A dropdown lets
you switch between matrix sizes.

Usage:
    python dashboard.py [results_dir]
    open results/dashboard.html
"""

from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from benchmark import collect, add_speedup


def _to_float(s, default=None):
    try:    return float(s)
    except: return default

def _to_int(s, default=None):
    try:    return int(s)
    except: return default


def normalize(rows):
    out = []
    for r in rows:
        out.append({
            "method":       r.get("Method", ""),
            "n":            _to_int(r.get("Matrix dimension"), 0),
            "time_s":       _to_float(r.get("Time (seconds)"), 0.0),
            "cores":        _to_int(r.get("Cores used"), 1),
            "base_case":    _to_int(r.get("Strassen base case")),
            "depth":        _to_int(r.get("Parallel depth")),
            "verification": r.get("Verification", ""),
        })
    return out


def build_dashboard(results_dir, out_path):
    raw = add_speedup(collect(results_dir))
    rows = normalize(raw)
    if not rows:
        print(f"No info files found under {results_dir}", file=sys.stderr)
        return 1

    sizes = sorted({r["n"] for r in rows})
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = TEMPLATE.replace("__DATA__", json.dumps(rows)).replace("__TS__", ts)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Wrote {out_path}  ({len(rows)} runs, n ∈ {{{', '.join(map(str, sizes))}}})")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Strassen — Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg:#0f1117; --surface:#181c25; --surface2:#1e2330; --border:#2a2f3d;
  --text:#d4d8e3; --muted:#6b7280; --accent:#4f8ff7; --green:#34d399;
  --orange:#f59e42; --purple:#a78bfa; --red:#f87171;
  --seq:#4f8ff7; --par:#34d399; --str:#f59e42; --pstr:#a78bfa;
}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--text);font-family:Inter,-apple-system,system-ui,sans-serif;line-height:1.5}

/* top bar */
.topbar{display:flex;align-items:center;gap:20px;padding:20px 32px;border-bottom:1px solid var(--border);flex-wrap:wrap}
.topbar h1{font-size:18px;font-weight:600;white-space:nowrap}
.topbar select{background:var(--surface2);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:14px;cursor:pointer}
.topbar .stats{margin-left:auto;color:var(--muted);font-size:13px}

/* headline */
.headline{padding:20px 32px 0;font-size:15px;color:var(--muted)}
.headline strong{color:var(--text)}

/* layout */
main{padding:20px 32px 48px;display:flex;flex-direction:column;gap:20px;max-width:1200px;margin:0 auto}
.row{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.row{grid-template-columns:1fr}}

.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:20px}
.card h2{font-size:15px;font-weight:600;margin-bottom:2px}
.card .sub{font-size:12px;color:var(--muted);margin-bottom:14px}
.chart-box{position:relative;height:300px}

/* table */
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl th{text-align:left;padding:8px;border-bottom:1px solid var(--border);color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.04em;cursor:pointer;user-select:none;white-space:nowrap}
.tbl th:hover{color:var(--text)}
.tbl th.asc::after{content:" ▲";color:var(--accent)}
.tbl th.desc::after{content:" ▼";color:var(--accent)}
.tbl td{padding:8px;border-bottom:1px solid var(--border);font-family:"SF Mono",Menlo,Consolas,monospace;font-size:12px}
.tbl tr:hover td{background:var(--surface2)}
.tag{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:600}
.tag-Strassen{background:rgba(245,158,66,.15);color:var(--str)}
.tag-ParStrassen{background:rgba(167,139,250,.15);color:var(--pstr)}
.pass{color:var(--green)} .fail{color:var(--red)}

footer{text-align:center;color:var(--muted);font-size:11px;padding:12px;border-top:1px solid var(--border)}
</style></head><body>

<div class="topbar">
  <h1>Strassen — Benchmark Results</h1>
  <label>Matrix size: <select id="nSel"></select></label>
  <div class="stats" id="stats"></div>
</div>
<div class="headline" id="headline"></div>

<main>
  <div class="row">
    <div class="card">
      <h2>Execution time by method</h2>
      <div class="sub">Best configuration per method at this n. Lower is better.</div>
      <div class="chart-box"><canvas id="barChart"></canvas></div>
    </div>
    <div class="card">
      <h2>Time vs number of cores</h2>
      <div class="sub">Parallel methods only. Shows how adding cores reduces time.</div>
      <div class="chart-box"><canvas id="coresChart"></canvas></div>
    </div>
  </div>
  <div class="card">
    <h2>All runs at this size</h2>
    <div class="sub">Click a column header to sort.</div>
    <div style="overflow-x:auto"><table class="tbl"><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
  </div>
</main>

<footer>Generated __TS__</footer>

<script>
const DATA = __DATA__;
const METHODS = ['Strassen','ParStrassen'];
const COLORS = {Strassen:'#f59e42',ParStrassen:'#a78bfa'};
const CHART_OPTS = {
  responsive:true, maintainAspectRatio:false,
  plugins:{legend:{labels:{color:'#d4d8e3'}},
    tooltip:{backgroundColor:'#181c25',borderColor:'#2a2f3d',borderWidth:1,titleColor:'#d4d8e3',bodyColor:'#d4d8e3'}},
  scales:{
    x:{ticks:{color:'#6b7280'},grid:{color:'#1e2330'}},
    y:{ticks:{color:'#6b7280'},grid:{color:'#1e2330'}}
  }
};

let barChart, coresChart;
let sortKey='time_s', sortDir=1;

// ---- populate size dropdown ---- //
const ns = [...new Set(DATA.map(d=>d.n))].sort((a,b)=>a-b);
const sel = document.getElementById('nSel');
ns.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;sel.appendChild(o)});
// default to largest n
sel.value = ns[ns.length-1];
sel.addEventListener('change', render);

function atN(){ return DATA.filter(d=>d.n===+sel.value && d.time_s>0); }

// ---- best per method ---- //
function bestPerMethod(rows){
  const m={};
  rows.forEach(r=>{
    if(!m[r.method]||r.time_s<m[r.method].time_s) m[r.method]=r;
  });
  return METHODS.filter(k=>m[k]).map(k=>m[k]);
}

// ---- headline ---- //
function renderHeadline(rows){
  const el=document.getElementById('headline');
  const best=bestPerMethod(rows);
  if(best.length<2){el.textContent='';return}
  const strassen=best.find(r=>r.method==='Strassen');
  const parstrassen=best.find(r=>r.method==='ParStrassen');
  best.sort((a,b)=>a.time_s-b.time_s);
  const winner=best[0];
  if(strassen && parstrassen && parstrassen.time_s < strassen.time_s){
    const ratio=(strassen.time_s/parstrassen.time_s).toFixed(1);
    el.innerHTML=`At <strong>n = ${sel.value}</strong>, <strong>ParStrassen</strong> is <strong>${ratio}× faster</strong> than Strassen (${parstrassen.time_s.toFixed(3)}s vs ${strassen.time_s.toFixed(3)}s)`;
  } else if(strassen) {
    el.innerHTML=`At <strong>n = ${sel.value}</strong>, <strong>Strassen</strong> is fastest at <strong>${strassen.time_s.toFixed(3)}s</strong> (parallelism overhead outweighs gains at this size)`;
  } else {
    el.innerHTML=`At <strong>n = ${sel.value}</strong>, best time is <strong>${winner.time_s.toFixed(3)}s</strong> (${winner.method})`;
  }
}

// ---- bar chart ---- //
function renderBar(rows){
  const best=bestPerMethod(rows);
  const ctx=document.getElementById('barChart');
  if(barChart) barChart.destroy();
  barChart=new Chart(ctx,{
    type:'bar',
    data:{
      labels:best.map(r=>r.method),
      datasets:[{
        data:best.map(r=>r.time_s),
        backgroundColor:best.map(r=>COLORS[r.method]),
        borderRadius:4,
        maxBarThickness:60,
      }]
    },
    options:{...CHART_OPTS,
      indexAxis:'y',
      plugins:{...CHART_OPTS.plugins,legend:{display:false},
        tooltip:{...CHART_OPTS.plugins.tooltip,
          callbacks:{label:ctx=>{
            const r=best[ctx.dataIndex];
            const parts=[`${r.time_s.toFixed(4)}s`];
            if(r.cores>1) parts.push(`${r.cores} cores`);
            if(r.base_case!=null) parts.push(`base=${r.base_case}`);
            if(r.depth!=null) parts.push(`depth=${r.depth}`);
            return parts.join('  ·  ');
          }}
        }
      },
      scales:{
        x:{title:{display:true,text:'time (seconds)',color:'#6b7280'},
           ticks:{color:'#6b7280'},grid:{color:'#1e2330'},beginAtZero:true},
        y:{ticks:{color:'#d4d8e3',font:{size:13}},grid:{display:false}}
      }
    }
  });
}

// ---- cores chart ---- //
function renderCores(rows){
  const ctx=document.getElementById('coresChart');
  if(coresChart) coresChart.destroy();
  const datasets=[];

  // ParStrassen: best time at each core count (across base/depth)
  const byCores={};
  rows.filter(r=>r.method==='ParStrassen').forEach(r=>{
    if(!byCores[r.cores]||r.time_s<byCores[r.cores].time_s) byCores[r.cores]=r;
  });
  const pts=Object.values(byCores).sort((a,b)=>a.cores-b.cores);
  if(pts.length){
    datasets.push({
      label:'ParStrassen',
      data:pts.map(p=>({x:p.cores,y:p.time_s})),
      borderColor:COLORS.ParStrassen,
      backgroundColor:COLORS.ParStrassen,
      tension:.25, pointRadius:5, pointHoverRadius:7,
      showLine:pts.length>1,
    });
  }

  // Strassen baseline as horizontal dashed line
  const strassen=rows.filter(r=>r.method==='Strassen');
  if(strassen.length){
    const best=Math.min(...strassen.map(r=>r.time_s));
    const maxC=Math.max(...rows.map(r=>r.cores),2);
    datasets.push({
      label:'Strassen (sequential)',
      data:[{x:1,y:best},{x:maxC,y:best}],
      borderColor:COLORS.Strassen, borderDash:[6,4], pointRadius:0, fill:false,
    });
  }

  coresChart=new Chart(ctx,{
    type:'scatter',
    data:{datasets},
    options:{...CHART_OPTS,
      scales:{
        x:{type:'linear',title:{display:true,text:'cores',color:'#6b7280'},
           ticks:{color:'#6b7280',stepSize:1},grid:{color:'#1e2330'},min:.5},
        y:{title:{display:true,text:'time (seconds)',color:'#6b7280'},
           ticks:{color:'#6b7280'},grid:{color:'#1e2330'},beginAtZero:true}
      }
    }
  });
}

// ---- table ---- //
const COLS=[
  ['method','Method'],['time_s','Time (s)'],['cores','Cores'],
  ['base_case','Base case'],['depth','Depth'],['verification','Verify']
];

function renderTable(rows){
  // header
  document.getElementById('thead').innerHTML='<tr>'+COLS.map(([k,label])=>{
    const cls=k===sortKey?(sortDir>0?'asc':'desc'):'';
    return `<th class="${cls}" data-k="${k}">${label}</th>`;
  }).join('')+'</tr>';
  document.querySelectorAll('.tbl th').forEach(th=>{
    th.onclick=()=>{
      const k=th.dataset.k;
      if(sortKey===k) sortDir=-sortDir; else {sortKey=k;sortDir=1;}
      renderTable(atN());
    };
  });

  // sort
  const sorted=[...rows].sort((a,b)=>{
    let av=a[sortKey],bv=b[sortKey];
    if(av==null&&bv==null)return 0; if(av==null)return 1; if(bv==null)return -1;
    if(typeof av==='number'&&typeof bv==='number') return (av-bv)*sortDir;
    return String(av).localeCompare(String(bv))*sortDir;
  });

  // body
  document.getElementById('tbody').innerHTML=sorted.map(r=>'<tr>'+COLS.map(([k])=>{
    const v=r[k];
    if(k==='method') return `<td><span class="tag tag-${v}">${v}</span></td>`;
    if(k==='verification') return `<td class="${v==='PASS'?'pass':'fail'}">${v||'—'}</td>`;
    if(k==='time_s') return `<td>${v.toFixed(4)}</td>`;
    return `<td>${v!=null?v:'—'}</td>`;
  }).join('')+'</tr>').join('');
}

// ---- stats ---- //
function renderStats(rows){
  const pass=rows.filter(r=>r.verification==='PASS').length;
  const fail=rows.filter(r=>r.verification==='FAIL').length;
  const methods=new Set(rows.map(r=>r.method)).size;
  document.getElementById('stats').textContent=
    `${rows.length} runs · ${methods} methods · ${pass} passed` + (fail?` · ${fail} failed`:'');
}

// ---- master render ---- //
function render(){
  const rows=atN();
  renderStats(rows);
  renderHeadline(rows);
  renderBar(rows);
  renderCores(rows);
  renderTable(rows);
}
render();
</script></body></html>
"""


def main(argv=None):
    args = argv or sys.argv[1:]
    d = args[0] if args else "."
    if not os.path.isdir(d):
        print(f"error: '{d}' is not a directory", file=sys.stderr)
        return 1
    return build_dashboard(d, os.path.join(d, "dashboard.html"))

if __name__ == "__main__":
    sys.exit(main())