"""ANSI color helpers for GPU and node rendering."""

from rich.text import Text

from .data import GpuInfo, NodeInfo

BLOCK = "\u2588"

# Standard ANSI colors via Rich's color(N) syntax — bypasses Textual theme
ANSI_RED = "color(1)"
ANSI_GREEN = "color(2)"
ANSI_YELLOW = "color(3)"
ANSI_BLUE = "color(4)"
ANSI_WHITE = "color(7)"
ANSI_BRIGHT_BLACK = "color(8)"
ANSI_BRIGHT_WHITE = "color(15)"
ANSI_BLACK = "color(0)"

_NAME_TO_ANSI = {
    "red": ANSI_RED,
    "green": ANSI_GREEN,
    "yellow": ANSI_YELLOW,
    "blue": ANSI_BLUE,
    "white": ANSI_WHITE,
}


def gpu_color(gpu: GpuInfo, current_user: str) -> str:
    """Return a logical color name for the GPU."""
    if gpu.is_drained:
        return "yellow"
    if gpu.user == current_user:
        return "white"
    if gpu.user is not None:
        vram_pct = (gpu.mem_used / gpu.mem_total * 100) if gpu.mem_total > 0 else 0
        return "red" if vram_pct > 5 else "green"
    return "blue"


def gpu_style(gpu: GpuInfo, current_user: str) -> str:
    """Return an ANSI Rich style string for the GPU."""
    return _NAME_TO_ANSI[gpu_color(gpu, current_user)]


def node_color(node: NodeInfo, current_user: str) -> str:
    """Return a logical color name for the node, based on actual GPU states."""
    if not node.gpus:
        if "drain" in node.state or "down" in node.state:
            return "yellow"
        return "blue"
    if any(g.user == current_user for g in node.gpus):
        return "white"
    if all(g.is_drained for g in node.gpus):
        return "yellow"
    free = sum(1 for g in node.gpus if g.user is None and not g.is_drained)
    used = sum(1 for g in node.gpus if g.user is not None)
    if free == len(node.gpus):
        return "blue"
    if used == len(node.gpus):
        return "red"
    # Mix of free and used — check for green GPUs (used but low VRAM usage)
    has_green = any(
        g.user is not None and not g.is_drained
        and (g.mem_used / g.mem_total * 100 if g.mem_total > 0 else 0) <= 5
        for g in node.gpus
    )
    if has_green or free > 0:
        return "green"
    return "red"


def node_style(node: NodeInfo, current_user: str) -> str:
    """Return an ANSI Rich style string for the node."""
    return _NAME_TO_ANSI[node_color(node, current_user)]


def node_sort_key(node: NodeInfo, current_user: str) -> tuple[int, str]:
    color = node_color(node, current_user)
    order = {"white": 0, "blue": 1, "green": 2, "yellow": 3, "red": 4}
    return (order.get(color, 5), node.name)


def render_vram_bar(pct: float, color: str, width: int = 16) -> Text:
    """color is a logical color name like 'red', 'blue', etc."""
    ansi = _NAME_TO_ANSI.get(color, color)
    filled = int(pct * width)
    empty = width - filled
    t = Text()
    t.append(BLOCK * filled, style=ansi)
    t.append("\u2591" * empty, style=ANSI_BRIGHT_BLACK)
    t.append(f" {pct * 100:3.0f}%", style=ansi)
    return t
