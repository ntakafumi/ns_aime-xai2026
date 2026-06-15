"""
ns_aime_xai2026.interactive
====================

Self-contained **interactive** HTML visualization for NS-AIME explanations.

No third-party JS libraries (no pyvis / d3 / vis-network): the output is a
single ``.html`` file with an inline SVG renderer + a tiny force-directed layout
written in vanilla JavaScript.  It mirrors the AIME inverse-operator aesthetic
(cream paper, coral = promote, indigo = inhibit, A-dagger branding) and adds:

* a draggable, self-organising radial logic graph (force layout);
* animated directional "flow" along every edge (feature -> class), so the
  direction of influence is conveyed by motion, not just arrowheads;
* hover tooltips with the rule and the raw/constrained weights;
* class switching buttons;
* a live dual-reporting panel (raw AIME vs NS-AIME bars + RD / GA / RS).

Use :func:`render_logic_graph_html` to write the file.
"""

from __future__ import annotations

from typing import Optional, Sequence, List, Dict, Any
import json
import numpy as np


def _class_payload(B, A, rules, feature_names, class_index, top_k):
    coefs = np.asarray(B)[:, class_index]
    raw = np.asarray(A)[:, class_index]
    class_rules = rules[class_index]
    single = {r["feature_idx"]: r for r in class_rules if r.get("type") == "single"}

    order = np.argsort(np.abs(coefs))[::-1]
    feats = [int(j) for j in order if abs(coefs[j]) > 1e-6][:top_k]
    if not feats:
        feats = [int(order[0])]

    def role(j):
        if j in single:
            return float(single[j]["sign"])
        return 1.0 if coefs[j] >= 0 else -1.0

    def thr_text(j):
        r = single.get(j)
        if r is None:
            return ""
        if r["presence"]:
            return "present"
        op = "≥" if r["sign"] > 0 else "≤"
        td = r.get("threshold_display", r["threshold"])
        return f"{op} {td:.2f}" if abs(td) >= 0.005 else f"{op} 0"

    fpayload = [{
        "name": str(feature_names[j]),
        "wB": float(coefs[j]),
        "wA": float(raw[j]),
        "sign": role(j),
        "thr": thr_text(j),
    } for j in feats]

    ands = [r for r in class_rules if r.get("type") == "and"]
    ands = sorted(ands, key=lambda r: r["importance"], reverse=True)[:3]
    apayload = [{
        "feats": [str(n) for n in r["names"]],
        "imp": float(r["importance"]),
    } for r in ands]

    return {"features": fpayload, "ands": apayload}


def render_logic_graph_html(
    B: np.ndarray,
    A_dagger: np.ndarray,
    rules: List[List[Dict[str, Any]]],
    feature_names: Sequence[str],
    class_names: Optional[Sequence[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    raw_metrics: Optional[Dict[str, float]] = None,
    top_k: int = 10,
    path: Optional[str] = None,
    title: str = "NS-AIME · Interactive Logic Graph",
) -> str:
    """Build the interactive HTML and (optionally) write it to ``path``.

    Parameters
    ----------
    B, A_dagger : (d, m) arrays
        Constrained (Eq. 2) and raw (Eq. 1) explanation matrices.
    rules : list
        Output of :func:`ns_aime_xai2026.utils.induce_rules` / ``NSAIME.rules_``.
    feature_names, class_names : sequences
    metrics, raw_metrics : dict, optional
        ``compute_metrics`` outputs for NS-AIME(B) and raw AIME(A†) - shown in
        the dual-reporting panel.
    top_k : int
        Features per class.
    path : str, optional
        If given, the HTML is written there.

    Returns
    -------
    str
        The full HTML document.
    """
    m = np.asarray(B).shape[1]
    if class_names is None:
        class_names = [f"class {k}" for k in range(m)]

    per_class = [_class_payload(B, A_dagger, rules, feature_names, k, top_k)
                 for k in range(m)]
    data = {
        "title": title,
        "classes": [str(c) for c in class_names],
        "perClass": per_class,
        "metrics": metrics or {},
        "rawMetrics": raw_metrics or {},
    }
    html = _TEMPLATE.replace("/*__DATA__*/", json.dumps(data))
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
    return html


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NS-AIME · Interactive Logic Graph</title>
<style>
  :root{
    --ink:#1B2A4A; --soft:#5C6680; --paper:#FBF9F4; --panel:#FFFFFF;
    --grid:#E6E0D6; --indigo:#21307A; --coral:#D2552E; --gold:#E0A23B;
    --violet:#7A4FA0;
  }
  *{box-sizing:border-box;}
  body{margin:0;background:var(--paper);color:var(--ink);
       font-family:"DejaVu Sans","Segoe UI",Helvetica,Arial,sans-serif;}
  header{padding:20px 26px 4px;}
  h1{font-size:21px;margin:0;font-weight:800;letter-spacing:.2px;}
  .sub{color:var(--soft);font-size:13px;margin-top:4px;}
  .rule{width:54px;height:4px;border-radius:3px;background:var(--coral);margin:10px 0 0;}
  .tabs{display:flex;gap:8px;padding:12px 26px 0;flex-wrap:wrap;}
  .tab{border:1px solid var(--grid);background:#fff;border-radius:20px;
       padding:6px 16px;font-size:13px;font-weight:700;cursor:pointer;color:var(--soft);}
  .tab.active{background:var(--ink);color:#fff;border-color:var(--ink);}
  .wrap{display:flex;gap:18px;padding:14px 26px 28px;flex-wrap:wrap;}
  .card{background:var(--panel);border:1px solid var(--grid);border-radius:14px;
        box-shadow:0 8px 26px rgba(27,42,74,.07);padding:8px;position:relative;}
  .side{min-width:280px;max-width:340px;flex:1;padding:16px 18px;}
  .side h2{font-size:14px;margin:.1em 0 .7em;}
  svg{display:block;border-radius:10px;}
  .legend{display:flex;gap:16px;padding:4px 26px 0;color:var(--soft);font-size:12px;flex-wrap:wrap;}
  .chip{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:middle;}
  table{width:100%;border-collapse:collapse;font-size:12.5px;}
  td{padding:4px 4px;border-bottom:1px solid var(--grid);}
  td.v{text-align:right;font-variant-numeric:tabular-nums;font-weight:700;}
  .bar{height:9px;border-radius:5px;}
  .metric{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px dashed var(--grid);font-size:13px;}
  .metric b{font-variant-numeric:tabular-nums;}
  .flag{color:var(--coral);font-weight:800;}
  .ok{color:var(--indigo);font-weight:800;}
  .hint{color:var(--soft);font-size:11.5px;margin-top:10px;line-height:1.5;}
  .brand{text-align:right;padding:2px 26px 16px;font-size:12px;font-weight:800;color:var(--indigo);opacity:.85;}
  .brand small{font-weight:400;color:var(--soft);font-size:9px;margin-left:8px;}
  #tip{position:fixed;pointer-events:none;background:var(--ink);color:#fff;
       padding:7px 10px;border-radius:8px;font-size:12px;opacity:0;transition:opacity .12s;
       box-shadow:0 6px 18px rgba(0,0,0,.25);z-index:50;max-width:240px;}
  @keyframes flow{to{stroke-dashoffset:-24;}}
  .edge.flow{stroke-dasharray:6 8;animation:flow 1.1s linear infinite;}
</style></head>
<body>
<header>
  <h1 id="title">NS-AIME · Interactive Logic Graph</h1>
  <div class="sub">Drag the nodes. Coral edges <b>promote</b> the class, indigo edges <b>inhibit</b> it; motion flows from each input feature toward the class.</div>
  <div class="rule"></div>
</header>
<div class="tabs" id="tabs"></div>
<div class="legend">
  <span><span class="chip" style="background:var(--coral)"></span>promotes (M=+1)</span>
  <span><span class="chip" style="background:var(--indigo)"></span>inhibits (M=−1)</span>
  <span><span class="chip" style="background:var(--violet)"></span>pairwise AND</span>
  <span><span class="chip" style="background:var(--gold)"></span>class</span>
</div>
<div class="wrap">
  <div class="card" id="graphcard"><svg id="svg" width="720" height="540"></svg></div>
  <div class="side card">
    <h2 id="dr-title">Dual reporting</h2>
    <div id="metrics"></div>
    <h2 style="margin-top:14px;">Attribution · raw A† vs NS-AIME</h2>
    <div id="bars"></div>
    <div class="hint">Bars compare the raw inverse surrogate A† (indigo) with the rule-constrained matrix B (coral). A large RD or low/negative GA is an <b>audit signal</b> — read B together with A†, never instead of it.</div>
  </div>
</div>
<div class="brand">NS-AIME · A† → rules<small>Approximate Inverse Model Explanations</small></div>
<div id="tip"></div>
<script>
const D = /*__DATA__*/;
document.getElementById("title").textContent = D.title;
const C = {ink:"#1B2A4A",soft:"#5C6680",indigo:"#21307A",coral:"#D2552E",gold:"#E0A23B",violet:"#7A4FA0",grid:"#E6E0D6"};
const svg = document.getElementById("svg");
const W = 720, H = 540, cx = W/2, cy = H/2;
let cur = 0, nodes = [], edges = [], dragging = null, raf = null;

function tabs(){
  const t = document.getElementById("tabs"); t.innerHTML="";
  D.classes.forEach((c,k)=>{
    const b=document.createElement("div"); b.className="tab"+(k===cur?" active":"");
    b.textContent=c; b.onclick=()=>{cur=k;build();}; t.appendChild(b);
  });
}

function build(){
  tabs();
  const pc = D.perClass[cur];
  const feats = pc.features, ands = pc.ands;
  const wmax = Math.max(1e-9, ...feats.map(f=>Math.abs(f.wB)));
  nodes = []; edges = [];
  // hub
  const hub = {id:"hub", x:cx, y:cy, vx:0, vy:0, r:34, type:"class",
               color:C.ink, label:D.classes[cur], fixed:true};
  nodes.push(hub);
  // feature nodes on a ring (split promote up / inhibit down)
  const pos = feats.filter(f=>f.sign>0), neg = feats.filter(f=>f.sign<=0);
  function place(arr, up){
    arr.forEach((f,i)=>{
      const a = (up? -1:1) * (Math.PI*0.18 + Math.PI*0.64*(arr.length<=1?0.5:i/(arr.length-1)));
      const x = cx + 230*Math.cos(a), y = cy + 200*Math.sin(a)*(up?1:1);
      const r = 10 + 14*(Math.abs(f.wB)/wmax);
      const nd = {id:"f"+f.name, x:x, y:y, vx:0, vy:0, r:r, type:"feat",
                  color: f.sign>0?C.coral:C.indigo, label:f.name,
                  wB:f.wB, wA:f.wA, thr:f.thr, sign:f.sign, fixed:false};
      nodes.push(nd);
      edges.push({a:nd, b:hub, color:nd.color, w:1.6+5.0*(Math.abs(f.wB)/wmax), flow:true});
    });
  }
  place(pos,true); place(neg,false);
  // AND nodes
  ands.forEach((r,i)=>{
    const a = Math.PI + (i-(ands.length-1)/2)*0.5;
    const nd={id:"and"+i, x:cx+120*Math.cos(a), y:cy+120*Math.sin(a), vx:0,vy:0,
              r:13, type:"and", color:C.violet, label:"&", feats:r.feats, fixed:false};
    nodes.push(nd);
    edges.push({a:nd, b:hub, color:C.violet, w:3.0, flow:true});
    r.feats.forEach(fn=>{
      const fnode = nodes.find(n=>n.id==="f"+fn);
      if(fnode) edges.push({a:fnode, b:nd, color:C.violet, w:1.8, flow:false});
    });
  });
  metrics(); bars();
  for(let i=0;i<140;i++) physics();   // settle
  render();
  if(raf) cancelAnimationFrame(raf);
  loop();
}

function physics(){
  const kRep=5200, kSpring=0.012, L=150, damp=0.86, grav=0.008;
  for(let i=0;i<nodes.length;i++){
    const n=nodes[i]; if(n.fixed) continue;
    for(let j=0;j<nodes.length;j++){
      if(i===j) continue; const o=nodes[j];
      let dx=n.x-o.x, dy=n.y-o.y; let d2=dx*dx+dy*dy+0.01; let d=Math.sqrt(d2);
      const f=kRep/d2; n.vx+=f*dx/d; n.vy+=f*dy/d;
    }
    n.vx += (cx-n.x)*grav; n.vy += (cy-n.y)*grav;  // gentle centering
  }
  edges.forEach(e=>{
    let dx=e.b.x-e.a.x, dy=e.b.y-e.a.y; let d=Math.sqrt(dx*dx+dy*dy)+0.01;
    const f=kSpring*(d-L);
    if(!e.a.fixed){e.a.vx+=f*dx/d; e.a.vy+=f*dy/d;}
    if(!e.b.fixed){e.b.vx-=f*dx/d; e.b.vy-=f*dy/d;}
  });
  nodes.forEach(n=>{
    if(n.fixed||n===dragging) return;
    n.vx*=damp; n.vy*=damp; n.x+=n.vx; n.y+=n.vy;
    n.x=Math.max(40,Math.min(W-40,n.x)); n.y=Math.max(40,Math.min(H-40,n.y));
  });
}

function curve(a,b){
  const mx=(a.x+b.x)/2, my=(a.y+b.y)/2;
  const dx=b.x-a.x, dy=b.y-a.y; const nx=-dy, ny=dx;
  const cxp=mx+nx*0.12, cyp=my+ny*0.12;
  return `M ${a.x} ${a.y} Q ${cxp} ${cyp} ${b.x} ${b.y}`;
}

function render(){
  let s="";
  s+=`<defs><marker id="arrI" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="${C.indigo}"/></marker>`;
  s+=`<marker id="arrC" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="${C.coral}"/></marker>`;
  s+=`<marker id="arrV" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="${C.violet}"/></marker></defs>`;
  edges.forEach(e=>{
    const mk = e.color===C.coral?"arrC":(e.color===C.indigo?"arrI":"arrV");
    s+=`<path class="edge ${e.flow?'flow':''}" d="${curve(e.a,e.b)}" fill="none" stroke="${e.color}" stroke-width="${e.w}" stroke-linecap="round" opacity="0.34"/>`;
    s+=`<path d="${curve(e.a,e.b)}" fill="none" stroke="${e.color}" stroke-width="${e.w}" stroke-linecap="round" opacity="0.9" marker-end="url(#${mk})"/>`;
  });
  nodes.forEach((n,i)=>{
    if(n.type==="class"){
      s+=`<circle cx="${n.x}" cy="${n.y}" r="${n.r+4}" fill="none" stroke="${C.gold}" stroke-width="3"/>`;
      s+=`<circle data-i="${i}" cx="${n.x}" cy="${n.y}" r="${n.r}" fill="${n.color}" stroke="#fff" stroke-width="2"/>`;
      s+=`<text x="${n.x}" y="${n.y+4}" text-anchor="middle" font-size="12" font-weight="800" fill="#fff">${esc(n.label)}</text>`;
    } else if(n.type==="and"){
      s+=`<rect data-i="${i}" x="${n.x-n.r}" y="${n.y-n.r}" width="${2*n.r}" height="${2*n.r}" rx="5" fill="${n.color}" stroke="#fff" stroke-width="2"/>`;
      s+=`<text x="${n.x}" y="${n.y+5}" text-anchor="middle" font-size="15" font-weight="800" fill="#fff">&amp;</text>`;
    } else {
      s+=`<circle data-i="${i}" cx="${n.x}" cy="${n.y}" r="${n.r}" fill="#fff" stroke="${n.color}" stroke-width="3"/>`;
      const lx=n.x + (n.x>=cx? n.r+6 : -(n.r+6));
      const anch = n.x>=cx? "start":"end";
      s+=`<text x="${lx}" y="${n.y+4}" text-anchor="${anch}" font-size="11.5" font-weight="700" fill="${C.ink}">${esc(n.label)}</text>`;
      if(n.thr) s+=`<text x="${(n.x+cx)/2}" y="${(n.y+cy)/2-4}" text-anchor="middle" font-size="10" font-weight="700" fill="${n.color}">${esc(n.thr)}</text>`;
    }
  });
  svg.innerHTML=s;
  bindNodes();
}

function esc(t){return (""+t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}

function bindNodes(){
  svg.querySelectorAll("[data-i]").forEach(el=>{
    const n=nodes[+el.dataset.i];
    el.style.cursor="grab";
    el.addEventListener("mousedown",ev=>{dragging=n;el.style.cursor="grabbing";ev.preventDefault();});
    el.addEventListener("mousemove",ev=>showTip(ev,n));
    el.addEventListener("mouseleave",hideTip);
  });
}
window.addEventListener("mousemove",ev=>{
  if(!dragging) return;
  const r=svg.getBoundingClientRect();
  dragging.x=ev.clientX-r.left; dragging.y=ev.clientY-r.top;
  dragging.vx=0; dragging.vy=0;
});
window.addEventListener("mouseup",()=>{dragging=null;});

const tip=document.getElementById("tip");
function showTip(ev,n){
  let html="";
  if(n.type==="feat"){
    const role=n.sign>0?"<b style='color:"+C.coral+"'>promotes</b>":"<b style='color:"+C.indigo+"'>inhibits</b>";
    html=`<b>${esc(n.label)}</b> ${role}<br>rule: x ${esc(n.thr||"-")}<br>B = ${n.wB.toFixed(3)} · A† = ${n.wA.toFixed(3)}`;
  } else if(n.type==="and"){
    html=`<b>pairwise AND</b><br>${n.feats.map(esc).join(" ∧ ")} ⇒ ${esc(D.classes[cur])}`;
  } else {
    html=`<b>class:</b> ${esc(n.label)}`;
  }
  tip.innerHTML=html; tip.style.opacity=1;
  tip.style.left=(ev.clientX+14)+"px"; tip.style.top=(ev.clientY+14)+"px";
}
function hideTip(){tip.style.opacity=0;}

function loop(){physics();render();raf=requestAnimationFrame(loop);}

function fmt(x,p=2){return (x>=0?"":"")+Number(x).toFixed(p);}
function metrics(){
  const el=document.getElementById("metrics");
  const m=D.metrics, rm=D.rawMetrics;
  if(!m || m.rd===undefined){el.innerHTML="<div class='hint'>metrics not provided</div>";return;}
  const flag=(m.ga!==undefined && m.ga<0.2)||(m.rd!==undefined && m.rd>3.0);
  let h="";
  h+=row("Rule Satisfaction (RS)", (rm&&rm.rs!==undefined?fmt(rm.rs)+" → ":"")+fmt(m.rs), "higher better");
  h+=row("Relative Deviation (RD)", fmt(m.rd), "lower better");
  h+=row("Global Alignment (GA)", fmt(m.ga), "higher better");
  h+=`<div class="metric"><span>audit status</span><b class="${flag?'flag':'ok'}">${flag?'⚠ divergent — inspect':'aligned'}</b></div>`;
  el.innerHTML=h;
}
function row(k,v,note){return `<div class="metric"><span>${k}<br><span style='color:var(--soft);font-size:11px'>${note}</span></span><b>${v}</b></div>`;}

function bars(){
  const el=document.getElementById("bars");
  const feats=D.perClass[cur].features.slice(0,10);
  const mx=Math.max(1e-9,...feats.map(f=>Math.max(Math.abs(f.wA),Math.abs(f.wB))));
  let h="<table>";
  feats.forEach(f=>{
    const a=50*Math.abs(f.wA)/mx, b=50*Math.abs(f.wB)/mx;
    h+=`<tr><td>${esc(f.name)}</td>`+
       `<td style="width:120px"><div class="bar" style="width:${a}%;background:${C.indigo};opacity:.8"></div>`+
       `<div class="bar" style="width:${b}%;background:${C.coral};margin-top:2px"></div></td>`+
       `<td class="v" style="color:${f.wB>=0?C.coral:C.indigo}">${fmt(f.wB,3)}</td></tr>`;
  });
  h+="</table>";
  el.innerHTML=h;
}

build();
</script>
</body></html>
"""
