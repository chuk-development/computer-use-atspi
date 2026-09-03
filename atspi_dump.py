"""Dump the AT-SPI accessibility tree of the running desktop as a compact list.

Runs INSIDE the container (it needs the desktop's a11y bus). Prints one line per
interactive element: [ref] role "name" @(cx,cy)  — the model reads this instead
of a screenshot and clicks by exact element center. This is the Linux equivalent
of Playwright's accessibility snapshot: text, not pixels.

Usage (inside container):  python3 atspi_dump.py [app-name-filter]
"""
import sys

import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi  # noqa: E402

Atspi.init()

INTERACTIVE = {
    "push button", "toggle button", "radio button", "check box", "menu item",
    "menu", "check menu item", "radio menu item", "text", "entry", "password text",
    "combo box", "list item", "table cell", "page tab", "link", "slider",
    "spin button", "toggle", "icon",
}


TEXT_ROLES = {"text", "entry", "password text", "label", "paragraph",
              "list item", "table cell", "spin button", "combo box"}


def text_value(acc):
    try:
        ti = acc.get_text_iface()
        if ti:
            n = ti.get_character_count()
            if n:
                return Atspi.Text.get_text(ti, 0, n)
    except Exception:
        pass
    return ""


def extents(acc):
    comp = None
    for getter in ("get_component_iface", "get_component"):
        try:
            comp = getattr(acc, getter)()
            if comp:
                break
        except Exception:
            comp = None
    if comp is None:
        try:
            comp = Atspi.Accessible.get_component_iface(acc)
        except Exception:
            return None
    if not comp:
        return None
    try:
        e = comp.get_extents(Atspi.CoordType.SCREEN)
        return (e.x, e.y, e.width, e.height)
    except Exception:
        return None


def walk(acc, out, ref, depth=0):
    try:
        role = acc.get_role_name()
        name = acc.get_name() or ""
    except Exception:
        return
    if role in INTERACTIVE:
        e = extents(acc)
        if e and e[2] > 0 and e[3] > 0 and e[1] >= 0:
            cx, cy = e[0] + e[2] // 2, e[1] + e[3] // 2
            ref[0] += 1
            val = text_value(acc) if role in TEXT_ROLES else ""
            label = val or name or "(unnamed)"
            out.append(f'[{ref[0]}] {role} "{label}" @({cx},{cy})')
    try:
        n = acc.get_child_count()
    except Exception:
        n = 0
    for i in range(min(n, 300)):
        try:
            child = acc.get_child_at_index(i)
        except Exception:
            continue
        if child is not None:
            walk(child, out, ref, depth + 1)


def main():
    target = sys.argv[1].lower() if len(sys.argv) > 1 else None
    desktop = Atspi.get_desktop(0)
    out, ref = [], [0]
    for i in range(desktop.get_child_count()):
        app = desktop.get_child_at_index(i)
        if app is None:
            continue
        an = app.get_name() or ""
        if target and target not in an.lower():
            continue
        if an.lower() in ("gnome-shell", "at-spi2-registryd"):
            continue
        before = ref[0]
        walk(app, out, ref)
        if ref[0] > before:
            out.insert(len(out) - (ref[0] - before), f"== {an} ==")
    print("\n".join(out) if out else "(no interactive elements found)")


if __name__ == "__main__":
    main()
