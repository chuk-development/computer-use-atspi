"""Host-side executor for the Linux computer-use demo.

Drives the running desktop container (linuxserver Webtop / XFCE on DISPLAY :1)
through `docker exec xdotool ...`, and grabs screenshots via `xwd` inside the
container converted to PNG on the host with ImageMagick `convert`.

No agent code runs inside the container: the model loop lives on the host and
only pokes the desktop from outside. Keeps the demo trivial to swap/retarget.
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile

CONTAINER = os.environ.get("CU_CONTAINER", "cu-live")
DISPLAY = os.environ.get("CU_DISPLAY", ":1")


def _x(*args) -> None:
    subprocess.run(
        ["docker", "exec", "-e", f"DISPLAY={DISPLAY}", CONTAINER, "xdotool", *map(str, args)],
        check=False, capture_output=True, text=True,
    )


def click(x: int, y: int, button: int = 1) -> None:
    _x("mousemove", x, y, "click", button)


def double_click(x: int, y: int) -> None:
    _x("mousemove", x, y, "click", "--repeat", "2", "1")


def right_click(x: int, y: int) -> None:
    _x("mousemove", x, y, "click", "3")


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    """Press at (x1,y1), drag to (x2,y2), release. For drawing, selecting, sliders."""
    _x("mousemove", x1, y1, "mousedown", "1", "mousemove", x2, y2, "mouseup", "1")


def move(x: int, y: int) -> None:
    _x("mousemove", x, y)


def type_text(text: str) -> None:
    _x("type", "--delay", "40", text)


def key(combo: str) -> None:
    _x("key", combo)


def scroll(amount: int) -> None:
    button = "4" if amount > 0 else "5"
    for _ in range(abs(amount)):
        _x("click", button)


def launch(cmd: str) -> None:
    """Start a GUI app detached on the desktop, e.g. launch('blender')."""
    subprocess.run(
        ["docker", "exec", "-d", CONTAINER, "sh", "-c", f"DISPLAY={DISPLAY} {cmd}"],
        check=False,
    )


_ATSPI_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atspi_dump.py")
_atspi_ready = False


def ui_tree(app_filter: str = "", timeout: int = 25) -> str:
    """Return the accessibility tree of on-screen apps: interactive elements with
    their EXACT click coordinates. The efficient alternative to guessing from a
    screenshot (Linux AT-SPI, like Playwright's a11y snapshot)."""
    global _atspi_ready
    if not _atspi_ready:
        subprocess.run(["docker", "cp", _ATSPI_SRC, f"{CONTAINER}:/root/atspi_dump.py"],
                       check=False, capture_output=True)
        _atspi_ready = True
    try:
        r = subprocess.run(
            ["docker", "exec", "-e", f"DISPLAY={DISPLAY}", CONTAINER,
             "python3", "/root/atspi_dump.py", app_filter or ""],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        out = ((r.stdout or "") or (r.stderr or "")).strip()
        return out[:4000] if out else "(no interactive elements found)"
    except subprocess.TimeoutExpired:
        return "(ui tree timed out)"


def bash(cmd: str, timeout: int = 90) -> str:
    """Run a shell command INSIDE the VM and return its output (agent-driven)."""
    try:
        r = subprocess.run(
            ["docker", "exec", "-e", f"DISPLAY={DISPLAY}", CONTAINER, "bash", "-lc", cmd],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out[:2000] if out else f"(no output, exit {r.returncode})"
    except subprocess.TimeoutExpired:
        return f"(command still running after {timeout}s; use launch for GUI apps)"


def geometry() -> tuple[int, int]:
    r = subprocess.run(
        ["docker", "exec", "-e", f"DISPLAY={DISPLAY}", CONTAINER, "xdotool", "getdisplaygeometry"],
        capture_output=True, text=True, check=False,
    )
    try:
        w, h = r.stdout.split()
        return int(w), int(h)
    except ValueError:
        return 1024, 768


def screenshot_png() -> bytes:
    subprocess.run(
        ["docker", "exec", CONTAINER, "sh", "-c", f"DISPLAY={DISPLAY} xwd -root -out /tmp/_cu.xwd"],
        check=False, capture_output=True,
    )
    xwd = tempfile.NamedTemporaryFile(suffix=".xwd", delete=False).name
    png = xwd[:-4] + ".png"
    subprocess.run(["docker", "cp", f"{CONTAINER}:/tmp/_cu.xwd", xwd], check=False, capture_output=True)
    subprocess.run(["convert", xwd, png], check=False, capture_output=True)
    try:
        with open(png, "rb") as fh:
            return fh.read()
    finally:
        for p in (xwd, png):
            try:
                os.unlink(p)
            except OSError:
                pass


def screenshot_b64() -> str:
    return base64.b64encode(screenshot_png()).decode()


def screenshot_file(path: str) -> str:
    """Grab a screenshot and save it to disk; return the path."""
    with open(path, "wb") as f:
        f.write(screenshot_png())
    return path


def file_b64(path: str) -> str:
    """Read a saved screenshot back as base64 (loaded into context on demand)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
