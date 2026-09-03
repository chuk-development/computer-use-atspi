"""Computer-use WebUI: a controllable agent session with a chat interface.

Left: the VM desktop (view-only VNC). Right: a chat where YOU send the prompts,
watch every action + reasoning + token stat live, and stop / pause / resume the
agent at any time. Prompts come from the UI, not from a script.

Backend: one long-lived AgentSession (a thread) that keeps a single running
conversation. New prompts either start a run (when idle) or are injected into the
running conversation (when busy). Image cap is enforced by prune_images each turn.

Run:  python webui.py     ->  open http://localhost:8090
Needs the desktop container 'cu-live' up with the view-only VNC on :3002.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import client
import computer
from agent_loop import KEEP_IMAGES, SYSTEM, TOOLS, action_label, execute, prune_images

PORT = 8090
VNC_URL = "http://localhost:3002/vnc.html?autoconnect=1&view_only=1&resize=scale&reconnect=1"
MAX_STEPS_PER_RUN = 40
_CLEAR = object()  # sentinel pushed onto the prompt queue to reset the session


def count_images(history: list) -> int:
    return sum(
        1 for m in history if isinstance(m.get("content"), list)
        for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url"
    )


class AgentSession:
    def __init__(self) -> None:
        self.w, self.h = computer.geometry()
        self.history: list = [{"role": "system", "content": SYSTEM.format(w=self.w, h=self.h)}]
        self.events: list = []
        self.ev_lock = threading.Lock()
        self.prompt_q: queue.Queue = queue.Queue()
        self.status = "idle"          # idle | running | paused | stopping
        self.stop_flag = False
        self.pause_flag = False
        self.cumulative = 0
        self.step_n = 0
        self.last = {"tps": 0, "images": 0, "prompt": 0, "completion": 0}
        self.t0 = time.time()
        threading.Thread(target=self._worker, daemon=True).start()

    # ---- event bus ----
    def emit(self, ev: dict) -> None:
        ev["t"] = time.time()
        with self.ev_lock:
            ev["id"] = len(self.events)
            self.events.append(ev)

    def snapshot(self, since: int) -> dict:
        with self.ev_lock:
            new = self.events[since:]
        return {
            "events": new, "next": len(self.events), "status": self.status,
            "stats": {
                "cumulative": self.cumulative, "steps": self.step_n,
                "tps": self.last["tps"], "images": self.last["images"],
                "prompt": self.last["prompt"], "completion": self.last["completion"],
                "elapsed": round(time.time() - self.t0),
            },
        }

    # ---- controls (from HTTP) ----
    def submit_prompt(self, text: str) -> None:
        self.emit({"type": "user", "text": text})
        self.prompt_q.put(text)

    def control(self, cmd: str) -> None:
        if cmd == "pause":
            self.pause_flag = True
        elif cmd == "resume":
            self.pause_flag = False
        elif cmd == "stop":
            self.stop_flag = True
            self.pause_flag = False
        elif cmd == "clear":
            # stop any run and queue a reset (handled in the worker thread)
            self.stop_flag = True
            self.pause_flag = False
            self.prompt_q.put(_CLEAR)
        self.emit({"type": "control", "cmd": cmd})

    def _reset(self) -> None:
        while not self.prompt_q.empty():
            try:
                self.prompt_q.get_nowait()
            except queue.Empty:
                break
        self.history = [{"role": "system", "content": SYSTEM.format(w=self.w, h=self.h)}]
        self.cumulative = 0
        self.step_n = 0
        self.last = {"tps": 0, "images": 0, "prompt": 0, "completion": 0}
        self.t0 = time.time()
        self.status = "idle"
        self.emit({"type": "reset"})

    # ---- worker ----
    def _worker(self) -> None:
        while True:
            item = self.prompt_q.get()
            if item is _CLEAR:
                self._reset()
                continue
            self.stop_flag = False
            try:
                self._run(item)
            except Exception as e:  # keep the session alive on any error
                self.emit({"type": "error", "text": f"{type(e).__name__}: {e}"[:400]})
            self.status = "idle"
            self.emit({"type": "status", "status": "idle"})

    def _run(self, prompt: str) -> None:
        self.status = "running"
        self.emit({"type": "status", "status": "running"})
        shot = computer.screenshot_b64()
        self.history.append({"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{shot}"}}]})

        used = 0
        while used < MAX_STEPS_PER_RUN and not self.stop_flag:
            while self.pause_flag and not self.stop_flag:
                if self.status != "paused":
                    self.status = "paused"
                    self.emit({"type": "status", "status": "paused"})
                time.sleep(0.25)
            if self.stop_flag:
                break
            if self.status != "running":
                self.status = "running"
                self.emit({"type": "status", "status": "running"})

            # inject any prompts the user sent mid-run
            while not self.prompt_q.empty():
                try:
                    self.history.append({"role": "user", "content": self.prompt_q.get_nowait()})
                except queue.Empty:
                    break

            prune_images(self.history, KEEP_IMAGES)
            n_imgs = count_images(self.history)

            # allocate this step's number and stream its reasoning live
            self.step_n += 1
            cur = self.step_n
            used += 1
            self.emit({"type": "step_begin", "n": cur, "images_in_context": n_imgs})

            buf = {"reasoning": "", "content": ""}
            last = {"t": 0.0}

            def on_delta(kind, text):
                buf[kind] += text
                now = time.time()
                if now - last["t"] > 0.12:
                    self.emit({"type": "delta", "n": cur,
                               "reasoning": buf["reasoning"], "content": buf["content"]})
                    last["t"] = now

            t_call = time.time()
            try:
                msg, usage = client.chat_stream(self.history, TOOLS, on_delta=on_delta)
            except Exception as e:
                self.emit({"type": "error", "text": f"{type(e).__name__}: {e}"[:400]})
                break
            latency = time.time() - t_call
            # final flush of streamed text
            self.emit({"type": "delta", "n": cur,
                       "reasoning": buf["reasoning"], "content": buf["content"]})

            ptok = int(usage.get("prompt_tokens", 0) or 0)
            ctok = int(usage.get("completion_tokens", 0) or 0)
            ttok = int(usage.get("total_tokens", ptok + ctok) or (ptok + ctok))
            self.cumulative += ttok
            tps = round(ctok / latency, 1) if latency > 0 else 0.0
            reasoning = msg.get("reasoning_content") or msg.get("content") or ""
            self.last = {"tps": tps, "images": n_imgs, "prompt": ptok, "completion": ctok}

            self.history.append({"role": "assistant", "content": msg.get("content"),
                                 "tool_calls": msg.get("tool_calls")})
            calls = msg.get("tool_calls") or []
            if not calls:
                self.emit({"type": "step", "n": cur, "action": "message",
                           "action_str": "assistant", "reasoning": reasoning,
                           "tool_result": msg.get("content") or reasoning, "tool_call_raw": "",
                           "images_in_context": n_imgs, "final": True, "is_done": False,
                           "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ttok,
                           "cumulative_tokens": self.cumulative, "latency_s": round(latency, 2), "tps": tps})
                break

            # answer every tool_call (only the first is executed; one action per step)
            first_args, first_name, first_result = {}, "wait", ""
            for i, c in enumerate(calls):
                if i == 0:
                    first_name = c.get("function", {}).get("name", "wait")
                    try:
                        first_args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        first_args = {}
                    try:
                        first_result = execute(first_name, first_args)
                    except Exception as e:
                        first_result = f"tool error: {e}"
                    r = first_result
                else:
                    r = "skipped (one action per step)"
                self.history.append({"role": "tool",
                                     "tool_call_id": c.get("id") or f"c{cur}_{i}",
                                     "content": r})

            fn = calls[0].get("function", {})
            raw = json.dumps({"name": fn.get("name"), "arguments": fn.get("arguments"),
                              "extra_calls": max(0, len(calls) - 1)}, ensure_ascii=False)
            ev = {"type": "step", "n": cur, "action": first_name,
                  "action_str": action_label(first_name, first_args), "reasoning": reasoning,
                  "content": msg.get("content") or "", "tool_call_raw": raw,
                  "tool_result": first_result, "images_in_context": n_imgs,
                  "final": first_name in ("done", "message"), "is_done": first_name == "done",
                  "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ttok,
                  "cumulative_tokens": self.cumulative, "latency_s": round(latency, 2), "tps": tps}
            self.emit(ev)

            if first_name == "done":
                break

            time.sleep(1.0)
            shot = computer.screenshot_b64()
            self.history.append({"role": "user", "content": [
                {"type": "text", "text": f"Screenshot after step {self.step_n}:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{shot}"}}]})


SESSION = AgentSession()

PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>computer-use</title>
<style>
:root{--bg:#0b0e14;--panel:#12161f;--panel2:#171c27;--line:#232a38;--fg:#dfe6f2;--dim:#8b96a8;
--accent:#4cc2ff;--good:#3ddc84;--warn:#ffb454;--bad:#ff5c6c;--user:#2a3a52;
--mono:ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
font:13px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
#top{display:flex;align-items:center;gap:14px;padding:9px 15px;border-bottom:1px solid var(--line);background:var(--panel)}
#top h1{font-size:14px;margin:0;font-weight:650}#top .m{color:var(--dim);font:12px var(--mono)}
#pill{margin-left:auto;padding:3px 11px;border-radius:20px;font:11px var(--mono);background:#20303f;color:var(--accent);display:flex;align-items:center;gap:6px}
#pill .dot{width:7px;height:7px;border-radius:50%;background:var(--dim)}
#pill.running .dot{background:var(--good);box-shadow:0 0 8px var(--good);animation:p 1s infinite}
#pill.paused .dot{background:var(--warn)}#pill.idle .dot{background:var(--dim)}#pill.stopping .dot{background:var(--bad)}
@keyframes p{50%{opacity:.4}}
#stats{display:flex;gap:0;border-bottom:1px solid var(--line);background:var(--line)}
.stat{background:var(--panel);padding:7px 14px;flex:1}.stat .k{color:var(--dim);font:10px var(--mono);text-transform:uppercase;letter-spacing:.5px}
.stat .v{font:600 17px var(--mono);margin-top:1px}
#main{display:grid;grid-template-columns:1.5fr 1fr;height:calc(100% - 44px - 48px);min-height:0}
#left{position:relative;border-right:1px solid var(--line);background:#000;min-width:0}
#left iframe{width:100%;height:100%;border:0;display:block}
#fs{position:absolute;top:9px;right:9px;z-index:5;background:#000a;color:#fff;border:1px solid #fff3;border-radius:6px;padding:5px 10px;font:12px var(--mono);cursor:pointer}
#right{display:flex;flex-direction:column;min-height:0;min-width:0}
#log{flex:1;overflow-y:auto;padding:11px}
.msg{margin-bottom:9px}
.user{background:var(--user);border-radius:10px 10px 3px 10px;padding:8px 12px;margin-left:22%;color:#eaf2ff}
.user .lbl{font:10px var(--mono);color:#9db6d8;margin-bottom:2px}
.step{background:var(--panel);border:1px solid var(--line);border-radius:9px;overflow:hidden}
.step .h{display:flex;gap:8px;align-items:center;padding:7px 10px;background:var(--panel2);border-bottom:1px solid var(--line)}
.step .n{font:600 12px var(--mono);color:var(--accent)}.step .act{flex:1;font:12px var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.step .tok{font:10px var(--mono);color:var(--dim);white-space:nowrap}
.step .b{padding:7px 10px}
.step .lbl2{font:10px var(--mono);text-transform:uppercase;letter-spacing:.5px;color:var(--dim);margin:6px 0 2px}
.step .lbl2:first-child{margin-top:0}
.step .rz{color:#c8d3e6;white-space:pre-wrap;font:12px/1.55 var(--mono);max-height:260px;overflow:auto;background:#0e131c;border-radius:6px;padding:6px 8px}
.step .raw{color:var(--warn);white-space:pre-wrap;font:11px/1.5 var(--mono);max-height:160px;overflow:auto;background:#0e131c;border-radius:6px;padding:6px 8px}
.step .res{color:var(--good);font:12px var(--mono);white-space:pre-wrap;max-height:200px;overflow:auto;background:#0e131c;border-radius:6px;padding:6px 8px}
.step.done .n{color:var(--good)}.badge{font:10px var(--mono);background:#20303f;color:var(--accent);padding:1px 5px;border-radius:4px}
.evc{color:var(--warn);font:11px var(--mono);text-align:center;padding:3px}
.err{color:var(--bad);font:12px var(--mono);background:#2a1417;border:1px solid #4a2027;border-radius:8px;padding:8px}
#ctrl{display:flex;gap:7px;padding:8px;border-top:1px solid var(--line);background:var(--panel);align-items:flex-end}
#ctrl textarea{flex:1;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px;font:13px var(--mono);resize:none;height:40px;max-height:120px}
button.b{background:#1c2536;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:8px 12px;font:12px var(--mono);cursor:pointer;white-space:nowrap}
button.b:hover{border-color:var(--accent)}button.send{background:var(--accent);color:#04121c;font-weight:700;border-color:var(--accent)}
button.stop:hover{border-color:var(--bad);color:var(--bad)}button.pause:hover{border-color:var(--warn);color:var(--warn)}
#log::-webkit-scrollbar,.rz::-webkit-scrollbar,.res::-webkit-scrollbar{width:8px}
#log::-webkit-scrollbar-thumb{background:var(--line);border-radius:4px}
@media(max-width:900px){#main{grid-template-columns:1fr;grid-template-rows:1fr 1fr}}
.step.final{border:1px solid var(--good);background:#0f1a14}
.step.final .h{background:#12271b}.finlbl{color:var(--good);font-weight:600}
.step.msg .h{background:#141d2b}.step.msg{border-color:#2a3a52}
.step.inprog{border-color:var(--accent)}.step.inprog .h{background:#13233a}
.spin{display:inline-block;width:9px;height:9px;border:2px solid var(--dim);border-top-color:var(--accent);border-radius:50%;animation:sp .7s linear infinite;vertical-align:middle;margin-right:4px}
@keyframes sp{to{transform:rotate(360deg)}}
.cursor{display:inline-block;width:6px;height:13px;background:var(--accent);animation:bl 1s steps(2) infinite;vertical-align:text-bottom}
@keyframes bl{50%{opacity:0}}
.md p{margin:.35em 0}.md>*:first-child{margin-top:0}.md>*:last-child{margin-bottom:0}
.md code{background:#0a0e15;padding:1px 4px;border-radius:3px;font:11px var(--mono)}
.md pre{background:#0a0e15;padding:7px;border-radius:6px;overflow:auto;font:11px var(--mono)}
.md pre code{background:none;padding:0}.md h1,.md h2,.md h3{font-size:13px;margin:.5em 0 .2em}
.md ul,.md ol{margin:.3em 0 .3em 1.2em;padding:0}.md li{margin:.15em 0}
.md a{color:var(--accent)}.md strong{color:#eaf2ff}.md table{border-collapse:collapse}.md td,.md th{border:1px solid var(--line);padding:2px 6px}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
</head><body>
<div id="top"><h1>computer-use</h1><div class="m" id="model">-</div>
<div id="pill" class="idle"><span class="dot"></span><span id="status">idle</span></div></div>
<div id="stats">
<div class="stat"><div class="k">total tokens</div><div class="v" id="s_total">0</div></div>
<div class="stat"><div class="k">gen tok/s</div><div class="v" id="s_tps">0</div></div>
<div class="stat"><div class="k">images in ctx</div><div class="v" id="s_img">0</div></div>
<div class="stat"><div class="k">steps</div><div class="v" id="s_steps">0</div></div>
<div class="stat"><div class="k">last p/c</div><div class="v" id="s_pc">0/0</div></div>
<div class="stat"><div class="k">elapsed</div><div class="v" id="s_el">0s</div></div>
</div>
<div id="main">
<div id="left"><button id="fs">⛶</button><iframe id="vnc" src="__VNC__" allow="fullscreen"></iframe></div>
<div id="right">
<div id="log"></div>
<div id="ctrl">
<button class="b pause" id="pause">⏸ pause</button>
<button class="b stop" id="stop">⏹ stop</button>
<button class="b" id="clear" title="clear session + context">🗑 new</button>
<textarea id="inp" placeholder="Tell the agent what to do…  (Enter to send)"></textarea>
<button class="b send" id="send">send ▸</button>
</div></div></div>
<script>
const $=s=>document.querySelector(s),log=$("#log");let since=0,paused=false;
$("#fs").onclick=()=>{const f=$("#left");(f.requestFullscreen||f.webkitRequestFullscreen).call(f)};
function esc(s){return(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function atBottom(){return log.scrollHeight-log.scrollTop-log.clientHeight<80}
function add(html){const b=atBottom();const d=document.createElement("div");d.className="msg";d.innerHTML=html;log.appendChild(d);if(b)log.scrollTop=log.scrollHeight}
function md(s){try{return (window.marked?marked.parse(String(s||"")):esc(s))}catch(_){return esc(s)}}
const cards={};
function stepClass(e){return e.action==="done"?"step final":(e.action==="message"?"step msg":"step")}
function stepInner(e){
 const think=e.reasoning?`<div class="lbl2">🧠 thinking</div><div class="rz md">${md(e.reasoning)}</div>`:"";
 if(e.action==="done")return `<div class="h"><span class="n">✓</span><span class="act finlbl">final answer</span><span class="tok">${e.total_tokens}t</span></div><div class="b">${think}<div class="lbl2">▸ result</div><div class="res md">${md(e.tool_result)}</div></div>`;
 if(e.action==="message")return `<div class="h"><span class="n">💬</span><span class="act finlbl">message to you</span><span class="tok">${e.total_tokens}t</span></div><div class="b">${think}<div class="lbl2">▸ message</div><div class="res md">${md(e.tool_result)}</div></div>`;
 return `<div class="h"><span class="n">#${e.n}</span><span class="act">${esc(e.action_str)}</span><span class="tok">${e.total_tokens}t · ${e.tps}/s · <span class="badge">${e.images_in_context}img</span></span></div><div class="b">${think}${e.tool_call_raw?`<div class="lbl2">⚙ tool_call</div><div class="raw">${esc(e.tool_call_raw)}</div>`:""}${e.tool_result?`<div class="lbl2">▸ result</div><div class="res">${esc(String(e.tool_result))}</div>`:""}</div>`;
}
function render(e){
 if(e.type==="user"){add(`<div class="user"><div class="lbl">you</div>${esc(e.text)}</div>`);return}
 if(e.type==="error"){add(`<div class="err">${esc(e.text)}</div>`);return}
 if(e.type==="reset"){log.innerHTML="";Object.keys(cards).forEach(k=>delete cards[k]);add(`<div class="evc">— new session · context cleared —</div>`);return}
 if(e.type==="control"){add(`<div class="evc">— ${esc(e.cmd)} —</div>`);return}
 if(e.type==="step_begin"){
   const b=atBottom();const d=document.createElement("div");d.className="msg";
   d.innerHTML=`<div class="step inprog"><div class="h"><span class="n">#${e.n}</span><span class="act"><span class="spin"></span>working…</span></div><div class="b"><div class="lbl2">🧠 thinking</div><div class="rz md" data-think><span class="cursor"></span></div></div></div>`;
   log.appendChild(d);cards[e.n]=d;if(b)log.scrollTop=log.scrollHeight;return;
 }
 if(e.type==="delta"){
   const d=cards[e.n];if(!d)return;const t=d.querySelector("[data-think]");if(!t)return;
   const txt=(e.reasoning||"")+(e.content?("\n\n"+e.content):"");
   const b=atBottom();t.innerHTML=md(txt)+'<span class="cursor"></span>';if(b)log.scrollTop=log.scrollHeight;return;
 }
 if(e.type==="step"){
   const b=atBottom();let d=cards[e.n];
   if(!d){d=document.createElement("div");d.className="msg";log.appendChild(d)}
   d.innerHTML=`<div class="${stepClass(e)}">${stepInner(e)}</div>`;
   delete cards[e.n];if(b)log.scrollTop=log.scrollHeight;return;
 }
 if(e.type==="assistant"){add(`<div class="step msg"><div class="h"><span class="n">💬</span><span class="act finlbl">assistant</span></div><div class="b"><div class="md">${md(e.text||e.reasoning)}</div></div></div>`);return}
}
async function poll(){
 let d;try{d=await(await fetch("/events?since="+since,{cache:"no-store"})).json()}catch(e){return}
 since=d.next;d.events.forEach(render);
 const st=d.stats;$("#s_total").textContent=st.cumulative.toLocaleString();$("#s_tps").textContent=st.tps;
 $("#s_img").textContent=st.images;$("#s_steps").textContent=st.steps;$("#s_pc").textContent=st.prompt+"/"+st.completion;
 $("#s_el").textContent=st.elapsed+"s";
 const p=$("#pill");p.className=d.status;$("#status").textContent=d.status;
 paused=d.status==="paused";$("#pause").textContent=paused?"▶ resume":"⏸ pause";
}
async function post(u,body){try{await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})}catch(e){}}
$("#send").onclick=()=>{const t=$("#inp").value.trim();if(!t)return;$("#inp").value="";post("/prompt",{text:t})};
$("#inp").onkeydown=e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();$("#send").click()}};
$("#pause").onclick=()=>post("/control",{cmd:paused?"resume":"pause"});
$("#stop").onclick=()=>post("/control",{cmd:"stop"});
$("#clear").onclick=()=>{if(confirm("Clear the session and wipe the context?"))post("/control",{cmd:"clear"})};
setInterval(poll,300);poll();
fetch("/info").then(r=>r.json()).then(i=>$("#model").textContent=i.model+" · "+i.display).catch(()=>{});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self):
        if self.path.startswith("/events"):
            since = 0
            if "since=" in self.path:
                try:
                    since = int(self.path.split("since=")[1].split("&")[0])
                except ValueError:
                    since = 0
            self._json(SESSION.snapshot(since))
        elif self.path.startswith("/info"):
            self._json({"model": client.MODEL, "display": f"{SESSION.w}x{SESSION.h}"})
        else:
            body = PAGE.replace("__VNC__", VNC_URL).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        data = self._read_json()
        if self.path.startswith("/prompt"):
            text = (data.get("text") or "").strip()
            if text:
                SESSION.submit_prompt(text)
            self._json({"ok": True})
        elif self.path.startswith("/control"):
            SESSION.control(data.get("cmd", ""))
            self._json({"ok": True})
        else:
            self._json({"ok": False})


if __name__ == "__main__":
    print(f"WebUI: http://localhost:{PORT}   (VNC view-only -> {VNC_URL})")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
