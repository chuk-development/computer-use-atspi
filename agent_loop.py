"""Linux computer-use agent loop.

screenshot -> model -> ONE action (via the `computer` tool) -> execute with
xdotool -> new screenshot -> repeat.

The point of this file is the CONTEXT DISCIPLINE that fixes the "only N images
per request" cap (the 30-image wall). Two mechanisms:

  1. IMAGE SLIDING WINDOW (`prune_images`): before every request, all but the
     last KEEP_IMAGES screenshots are replaced with a short text stub. So the
     request carries at most KEEP_IMAGES images no matter how many steps ran.
     This is the same idea as Anthropic's `only_n_most_recent_images` and the
     "compaction" the GPT-6 Astra harness used to survive long agent runs.

  2. STEP COMPACTION (`VERBATIM_STEPS`): older steps are folded into a one-line-
     per-step text summary, so the text history does not grow without bound
     either. The model still knows what it already did, just not in full detail.

Merging two screenshots into one image is deliberately NOT done: it halves
effective resolution, so the model reads small UI and hits click targets worse,
and it only postpones the cap. Dropping old images entirely is strictly better.
"""
from __future__ import annotations

import json
import os
import sys
import time

import client
import computer

EVENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_events.jsonl")


def emit(ev: dict) -> None:
    """Append one JSONL event for the live dashboard to tail."""
    ev["t"] = time.time()
    with open(EVENTS_PATH, "a") as f:
        f.write(json.dumps(ev, ensure_ascii=False) + "\n")

KEEP_IMAGES = 3        # max screenshots sent as real images per request
VERBATIM_STEPS = 8     # recent steps kept in full; older ones summarized
MAX_STEPS = 40
PAUSE_S = 1.2          # settle time after an action before the next screenshot

def _tool(name: str, desc: str, props: dict, required: list) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}

_XY = {"x": {"type": "integer", "description": "pixel x, from the left"},
       "y": {"type": "integer", "description": "pixel y, from the top"}}

# One tool per action. Models emit clean {name, arguments} instead of trying to
# flatten a single tool with an action-enum (GLM-5.3-Flash mangles the latter).
TOOLS = [
    _tool("click", "Left-click at pixel (x,y).", dict(_XY), ["x", "y"]),
    _tool("double_click", "Double-click at pixel (x,y).", dict(_XY), ["x", "y"]),
    _tool("right_click", "Right-click at pixel (x,y).", dict(_XY), ["x", "y"]),
    _tool("drag", "Press the mouse at (x1,y1), drag to (x2,y2) and release. Use for "
          "drawing shapes on a canvas, selecting a region, or moving a slider.",
          {"x1": {"type": "integer"}, "y1": {"type": "integer"},
           "x2": {"type": "integer"}, "y2": {"type": "integer"}}, ["x1", "y1", "x2", "y2"]),
    _tool("move_mouse", "Move the mouse to pixel (x,y) without clicking.", dict(_XY), ["x", "y"]),
    _tool("type_text", "Type this text at the current keyboard focus.",
          {"text": {"type": "string"}}, ["text"]),
    _tool("press_key", "Press a key or combo, e.g. Return, Tab, ctrl+s, ctrl+shift+t.",
          {"keys": {"type": "string"}}, ["keys"]),
    _tool("scroll", "Scroll the mouse wheel. Positive = up, negative = down.",
          {"amount": {"type": "integer"}}, ["amount"]),
    _tool("launch", "Open an installed program (your start menu), e.g. 'blender', 'xterm', 'chromium'.",
          {"command": {"type": "string"}}, ["command"]),
    _tool("bash", "Run a shell command inside the VM and receive its stdout/stderr.",
          {"command": {"type": "string"}}, ["command"]),
    _tool("get_ui_tree", "List the interactive on-screen UI elements (buttons, menus, "
          "text fields) with role, name and EXACT center coordinates. Use this to get "
          "precise click positions instead of guessing pixels from the screenshot. "
          "Optionally filter to one app.",
          {"app": {"type": "string", "description": "optional app-name filter, e.g. galculator"}}, []),
    _tool("message", "Send a message to the human watching.",
          {"text": {"type": "string"}}, ["text"]),
    _tool("wait", "Wait about a second for something to finish loading, then re-check the screen.",
          {}, []),
    _tool("done", "The task is finished. Give a short summary of the result.",
          {"summary": {"type": "string"}}, ["summary"]),
]

SYSTEM = (
    "You are a general computer-use agent living inside a Linux XFCE virtual machine "
    "at {w}x{h} pixels. Everything you do stays inside this VM. Make exactly ONE tool "
    "call per step, then look at the fresh screenshot before the next one.\n"
    "Tools: click / double_click / right_click / move_mouse (pixel coords); type_text; "
    "press_key (e.g. Return, Tab, ctrl+s); scroll. Open programs yourself with launch "
    "(your start menu), e.g. launch blender. Run shell commands with bash and read their "
    "output. Tell the human something with message. Use wait only while something is still "
    "loading. Call done with a summary when the task is complete.\n"
    "IMPORTANT for clicking: before you click a button, menu, or text field, call "
    "get_ui_tree to get the element's EXACT center coordinates, then click exactly those. "
    "Do NOT guess pixel positions from the screenshot — that misses small targets. Use the "
    "screenshot to understand the screen, but take click coordinates from the UI tree.\n"
    "Your reasoning is private — to actually DO something you MUST make the matching tool "
    "call, not just describe it. Prefer the simplest reliable path: if a shell command is "
    "faster and more robust than clicking, use bash; otherwise operate the GUI.\n"
    "Finish the WHOLE task before calling done: if the user asked you to draw, create, or "
    "produce something, actually make it and confirm it is visible on screen — never call "
    "done just because an app is open. To draw a shape, first select a tool (in canvas apps "
    "like Excalidraw the keyboard shortcuts are r=rectangle, o=ellipse, a=arrow, l=line, "
    "t=text, then click where you want text), then use drag to draw it on the canvas."
)


def prune_images(messages: list, keep_last: int = KEEP_IMAGES) -> list:
    """Replace all but the last `keep_last` screenshot images with a text stub.

    Guarantees the outgoing request never exceeds `keep_last` images, which is
    what keeps us permanently under any provider's per-request image cap.
    """
    img_msgs = [
        m for m in messages
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m["content"])
    ]
    for m in img_msgs[:-keep_last] if keep_last > 0 else img_msgs:
        step = m.get("_step", "?")
        m["content"] = f"[screenshot from step {step} pruned to stay under the image cap]"
    return messages


def build_messages(task: str, steps: list, w: int, h: int) -> list:
    """Rebuild the full request from the structured step log.

    Recent steps go in verbatim (assistant tool_call + tool result + screenshot);
    older steps collapse to a one-line summary. prune_images() is applied last.
    """
    msgs = [{"role": "system", "content": SYSTEM.format(w=w, h=h)},
            {"role": "user", "content": task}]

    old, recent = steps[:-VERBATIM_STEPS], steps[-VERBATIM_STEPS:]
    if old:
        summary = "\n".join(f"step {s['n']}: {s['action_str']} -> {s['result']}" for s in old)
        msgs.append({"role": "user",
                     "content": "Summary of earlier steps (screenshots dropped):\n" + summary})

    for s in recent:
        msgs.append({
            "role": "assistant",
            "content": s.get("think") or None,
            "tool_calls": [{"id": s["id"], "type": "function",
                            "function": {"name": "computer", "arguments": s["args_json"]}}],
        })
        msgs.append({"role": "tool", "tool_call_id": s["id"], "content": s["result"]})
        if s.get("shot_b64"):
            msgs.append({
                "role": "user", "_step": s["n"],
                "content": [
                    {"type": "text", "text": f"Screenshot after step {s['n']}:"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{s['shot_b64']}"}},
                ],
            })

    return prune_images(msgs)


def action_label(name: str, args: dict) -> str:
    """Short one-line label of a tool call for the log."""
    if name in ("click", "double_click", "right_click", "move_mouse"):
        return f"{name} {args.get('x')},{args.get('y')}"
    if name == "drag":
        return f"drag {args.get('x1')},{args.get('y1')}->{args.get('x2')},{args.get('y2')}"
    if name in ("type_text", "message"):
        t = str(args.get("text", ""))
        return f"{name} {t if len(t) < 60 else t[:57] + '…'}"
    if name == "press_key":
        return f"press_key {args.get('keys', '')}"
    if name == "scroll":
        return f"scroll {args.get('amount', 0)}"
    if name in ("launch", "bash"):
        c = str(args.get("command", ""))
        return f"{name} {c if len(c) < 60 else c[:57] + '…'}"
    if name == "get_ui_tree":
        return f"get_ui_tree {args.get('app', '')}".strip()
    if name == "done":
        return "done"
    return name


def execute(name: str, args: dict) -> str:
    if name == "click":
        computer.click(args["x"], args["y"]); return f"clicked ({args['x']},{args['y']})"
    if name == "double_click":
        computer.double_click(args["x"], args["y"]); return f"double-clicked ({args['x']},{args['y']})"
    if name == "right_click":
        computer.right_click(args["x"], args["y"]); return f"right-clicked ({args['x']},{args['y']})"
    if name == "drag":
        computer.drag(args["x1"], args["y1"], args["x2"], args["y2"])
        return f"dragged ({args['x1']},{args['y1']}) -> ({args['x2']},{args['y2']})"
    if name == "move_mouse":
        computer.move(args["x"], args["y"]); return f"moved to ({args['x']},{args['y']})"
    if name == "type_text":
        computer.type_text(args.get("text", "")); return f"typed: {args.get('text', '')}"
    if name == "press_key":
        computer.key(args.get("keys", "")); return f"pressed {args.get('keys', '')}"
    if name == "scroll":
        computer.scroll(int(args.get("amount", 0) or 0)); return f"scrolled {args.get('amount', 0)}"
    if name == "launch":
        cmd = args.get("command", "")
        parts = cmd.split(None, 1)
        if parts and parts[0] in ("chromium", "chromium-browser", "chrome", "google-chrome"):
            flags = ("--no-sandbox --disable-dev-shm-usage --disable-gpu "
                     "--force-renderer-accessibility --no-first-run --no-default-browser-check "
                     "--user-data-dir=/tmp/cu-profile --start-maximized")
            rest = parts[1] if len(parts) > 1 else ""
            cmd = f"chromium {flags} {rest}".strip()
        computer.launch(cmd)
        return f"launched {cmd!r}"
    if name == "bash":
        cmd = args.get("command", ""); return f"$ {cmd}\n{computer.bash(cmd)}"
    if name == "get_ui_tree":
        return computer.ui_tree(args.get("app", ""))
    if name == "message":
        return args.get("text", "")
    if name == "wait":
        time.sleep(1.0); return "waited"
    if name == "done":
        return args.get("summary", "done")
    return f"unknown tool {name}"


def run(task: str, max_steps: int = MAX_STEPS) -> None:
    w, h = computer.geometry()
    print(f"[cu] desktop {w}x{h}, task: {task}", flush=True)
    print("[cu] watch live: http://localhost:3000", flush=True)

    open(EVENTS_PATH, "w").close()  # fresh session for the dashboard
    t0 = time.time()
    cumulative = 0
    emit({"type": "start", "task": task, "model": client.MODEL,
          "display": f"{w}x{h}", "keep_images": KEEP_IMAGES})

    steps: list = []
    # seed with the initial screen so the first decision is grounded
    steps.append({
        "n": 0, "id": "step0", "action_str": "screenshot(initial)",
        "args_json": json.dumps({"action": "wait"}), "result": "initial screen",
        "think": None, "shot_b64": computer.screenshot_b64(),
    })

    status = "max_steps"
    for n in range(1, max_steps + 1):
        messages = build_messages(task, steps, w, h)
        n_imgs = sum(
            1 for m in messages if isinstance(m.get("content"), list)
            for b in m["content"] if isinstance(b, dict) and b.get("type") == "image_url"
        )
        print(f"[cu] step {n}: {n_imgs} image(s) in context, {len(messages)} messages", flush=True)

        t_call = time.time()
        msg, usage = client.chat(messages, TOOLS)
        latency = time.time() - t_call
        ptok = int(usage.get("prompt_tokens", 0) or 0)
        ctok = int(usage.get("completion_tokens", 0) or 0)
        ttok = int(usage.get("total_tokens", ptok + ctok) or (ptok + ctok))
        cumulative += ttok
        tps = round(ctok / latency, 1) if latency > 0 else 0.0
        reasoning = msg.get("reasoning_content") or msg.get("content") or ""

        calls = msg.get("tool_calls") or []
        if not calls:
            print(f"[cu] no action: {msg.get('content')!r}", flush=True)
            emit({"type": "step", "n": n, "action": "none", "action_str": "(no tool call)",
                  "reasoning": reasoning, "tool_result": "", "images_in_context": n_imgs,
                  "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ttok,
                  "cumulative_tokens": cumulative, "latency_s": round(latency, 2), "tps": tps})
            status = "stopped_no_action"
            break

        call = calls[0]
        name = call["function"]["name"]
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        action_str = action_label(name, args)
        print(f"[cu]   -> {action_str}", flush=True)

        base_ev = {
            "type": "step", "n": n, "action": name, "action_str": action_str,
            "reasoning": reasoning, "images_in_context": n_imgs,
            "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ttok,
            "cumulative_tokens": cumulative, "latency_s": round(latency, 2), "tps": tps,
        }

        if name == "done":
            emit({**base_ev, "tool_result": args.get("summary", "")})
            print(f"[cu] DONE: {args.get('summary')}", flush=True)
            status = "done"
            break

        result = execute(name, args)
        time.sleep(PAUSE_S)
        emit({**base_ev, "tool_result": result})
        steps.append({
            "n": n, "id": call.get("id") or f"step{n}", "action_str": action_str,
            "args_json": call["function"]["arguments"] or "{}", "result": result,
            "think": reasoning, "shot_b64": computer.screenshot_b64(),
        })

    emit({"type": "end", "status": status, "cumulative_tokens": cumulative,
          "elapsed_s": round(time.time() - t0, 1), "steps": len(steps) - 1})
    print(f"[cu] end: {status}, {cumulative} tokens, {round(time.time() - t0, 1)}s", flush=True)


if __name__ == "__main__":
    task = " ".join(sys.argv[1:]) or "Open a terminal and run 'neofetch' or 'uname -a'."
    run(task)
