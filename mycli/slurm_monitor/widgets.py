"""TUI widgets for the slurm monitor."""

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


def _format_duration(start: datetime | None) -> str:
    if start is None:
        return ""
    delta = datetime.now() - start
    total_minutes = int(delta.total_seconds()) // 60
    if total_minutes < 60:
        return f"{total_minutes}m"
    hours = total_minutes // 60
    mins = total_minutes % 60
    if hours < 24:
        return f"{hours}h{mins:02d}m"
    days = hours // 24
    hours = hours % 24
    return f"{days}d{hours:02d}h{mins:02d}m"

from .colors import (
    ANSI_BLACK, ANSI_BLUE, ANSI_BRIGHT_BLACK, ANSI_BRIGHT_WHITE, ANSI_GREEN,
    ANSI_RED, ANSI_WHITE, ANSI_YELLOW,
    BLOCK, BLOCK_SMALL, gpu_color, gpu_style, node_sort_key, node_style, render_vram_bar,
)
from .data import ClusterState, NodeInfo


class PriorityWidget(Static):
    """Non-interactive priority jobs display."""

    MAX_DISPLAY = 10

    def render_content(self, state: ClusterState, width: int) -> Text:
        text = Text()
        label = Text(" QUEUE", style=f"bold on {ANSI_BRIGHT_BLACK}")
        label.pad_right(width)
        text.append_text(label)
        text.append("\n\n")

        all_jobs = state.pending_jobs
        # Current user's jobs first, then the rest
        user_jobs = [j for j in all_jobs if j.user == state.current_user]
        other_jobs = [j for j in all_jobs if j.user != state.current_user]
        ordered = user_jobs + other_jobs
        display = ordered[:self.MAX_DISPLAY]
        remaining = len(ordered) - len(display)

        for job in display:
            col1 = f"{job.partition} [N-{job.num_nodes} G-{job.gpu_count}]"
            col2 = job.user
            is_me = job.user == state.current_user
            style = f"bold {ANSI_BRIGHT_WHITE}" if is_me else ANSI_BRIGHT_WHITE
            gap = max(1, width - len(col1) - len(col2) - 3)
            line = Text()
            line.append(" " + col1, style=style)
            line.append(" " * gap)
            line.append(col2 + " ", style=style)
            text.append_text(line)
            text.append("\n")

        if remaining > 0:
            msg = f"... {remaining} more in queue  "
            pad = max(0, width - len(msg))
            line = Text()
            line.append(" " * pad)
            line.append(msg, style=ANSI_BLUE)
            text.append_text(line)

        text.append("\n")
        return text


class NodeListWidget(Widget, can_focus=True):
    """Interactive node list with GPU blocks."""

    selected: reactive[int] = reactive(0)
    scroll_offset: reactive[int] = reactive(0)

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sorted_nodes: list[NodeInfo] = []
        self.current_user: str = ""
        self.filter_mode: str = ""
        self.filter_text: str = ""

    def update_nodes(self, state: ClusterState):
        self.current_user = state.current_user
        all_nodes = list(state.nodes.values())
        filtered = all_nodes
        if self.filter_text:
            if self.filter_mode == "user":
                ft = self.filter_text.lower()
                filtered = [n for n in all_nodes
                            if any(ft in (j.user or "").lower() for j in n.jobs)
                            or any(ft in (g.user or "").lower() for g in n.gpus)]
            elif self.filter_mode == "node":
                filtered = [n for n in all_nodes
                            if self.filter_text.lower() in n.name.lower()]
            elif self.filter_mode == "partition":
                ft = self.filter_text.lower()
                filtered = [n for n in all_nodes
                            if ft in n.partition.lower()]
        filtered.sort(key=lambda n: node_sort_key(n, state.current_user))
        self.sorted_nodes = filtered
        if self.selected >= len(self.sorted_nodes):
            self.selected = max(0, len(self.sorted_nodes) - 1)
        self.refresh()

    def get_selected_node(self) -> NodeInfo | None:
        if 0 <= self.selected < len(self.sorted_nodes):
            return self.sorted_nodes[self.selected]
        return None

    def render(self) -> Text:
        width = self.size.width
        text = Text()

        label = Text(" LIST OF NODES", style=f"bold on {ANSI_BRIGHT_BLACK}")
        label.pad_right(width)
        text.append_text(label)
        text.append("\n\n")

        if not self.sorted_nodes:
            text.append("  No nodes", style=ANSI_BRIGHT_BLACK)
            return text

        visible_height = max(1, self.size.height - 3)
        if self.selected < self.scroll_offset:
            self.scroll_offset = self.selected
        elif self.selected >= self.scroll_offset + visible_height:
            self.scroll_offset = self.selected - visible_height + 1

        max_name = max(len(n.name) for n in self.sorted_nodes) if self.sorted_nodes else 8
        max_type = max(len(n.partition) for n in self.sorted_nodes) if self.sorted_nodes else 0
        max_gpus = max(n.total_gpus for n in self.sorted_nodes) if self.sorted_nodes else 4

        end = min(self.scroll_offset + visible_height, len(self.sorted_nodes))
        for i in range(self.scroll_offset, end):
            node = self.sorted_nodes[i]
            is_selected = (i == self.selected)
            bg = f" on {ANSI_BLACK}" if is_selected else ""

            nc = node_style(node, self.current_user)
            blocks = Text()
            for gpu in node.gpus:
                gc = gpu_style(gpu, self.current_user)
                blocks.append(BLOCK_SMALL, style=gc + bg)

            pad_blocks = max_gpus - len(node.gpus)
            if pad_blocks > 0:
                blocks.append(" " * pad_blocks, style=bg if bg else "")

            COL_GAP = "    "  # 4 spaces between columns
            type_str = node.partition.ljust(max_type) if max_type > 0 else ""
            type_col_width = (max_type + len(COL_GAP)) if max_type > 0 else 0
            name_pad = max_name - len(node.name)
            content_width = 1 + name_pad + len(node.name) + len(COL_GAP) + type_col_width + max_gpus
            left_pad = max(0, width - content_width - 1)
            line = Text()
            line.append(" " * left_pad, style=bg if bg else "")
            line.append(" " * name_pad, style=bg if bg else "")
            line.append(node.name, style=nc + bg)
            line.append(COL_GAP, style=bg if bg else "")
            if type_str:
                line.append(type_str, style=ANSI_BRIGHT_BLACK + bg)
                line.append(COL_GAP, style=bg if bg else "")
            line.append_text(blocks)
            # Fill to full width for selected-line background, leave room for slab
            used = left_pad + name_pad + len(node.name) + len(COL_GAP) + type_col_width + len(node.gpus) + pad_blocks
            remaining = max(0, width - used - 1)
            line.append(" " * remaining, style=bg if bg else "")
            if is_selected:
                line.append("\u2590", style=f"{ANSI_WHITE} on {ANSI_BLACK}")
            else:
                line.append(" ")
            text.append_text(line)
            if i < end - 1:
                text.append("\n")

        return text

    def move_up(self):
        if not self.sorted_nodes:
            return
        self.selected = (self.selected - 1) % len(self.sorted_nodes)
        self.refresh()

    def move_down(self):
        if not self.sorted_nodes:
            return
        self.selected = (self.selected + 1) % len(self.sorted_nodes)
        self.refresh()

    def move_first(self):
        if not self.sorted_nodes:
            return
        self.selected = 0
        self.refresh()

    def move_last(self):
        if not self.sorted_nodes:
            return
        self.selected = len(self.sorted_nodes) - 1
        self.refresh()


class NodeDetailWidget(Static):
    """Right panel showing GPU details for selected node."""

    def render_detail(self, node: NodeInfo | None, current_user: str) -> Text:
        text = Text()
        if node is None:
            text.append("No node selected", style=ANSI_BRIGHT_BLACK)
            return text

        text.append(f" {node.name}", style="bold")
        text.append(f"  ({node.state})\n", style=ANSI_BRIGHT_BLACK)
        text.append("\n")

        max_user = max((len(gpu.user or "free") for gpu in node.gpus), default=4)

        for gpu in node.gpus:
            gc_name = gpu_color(gpu, current_user)
            gc = gpu_style(gpu, current_user)
            user_str = gpu.user or "free"
            is_bold = gpu.user == current_user
            style_prefix = "bold " if is_bold else ""

            line = Text()
            line.append(f" GPU {gpu.index}: ", style=ANSI_BRIGHT_BLACK)
            line.append(user_str.ljust(max_user), style=style_prefix + gc)

            if gpu.mem_total > 0:
                pct = gpu.mem_used / gpu.mem_total
                line.append("  ")
                line.append_text(render_vram_bar(pct, gc_name))
            elif gpu.is_drained:
                line.append("  DRAINED", style=f"{ANSI_RED} dim")
            else:
                line.append("  ", style=ANSI_BRIGHT_BLACK)
                line.append_text(render_vram_bar(0.0, gc_name))

            duration = _format_duration(gpu.start_time)
            if duration:
                line.append(f"  {duration}", style=gc)

            text.append_text(line)
            text.append("\n")

        return text


class ShortcutsWidget(Static):
    """Fixed legend + shortcuts help at the bottom of the right panel."""

    GPU_LEGEND = [
        (ANSI_WHITE, "current user"),
        (ANSI_RED, "used (VRAM > 5%)"),
        (ANSI_GREEN, "used (VRAM \u2264 5%)"),
        (ANSI_BLUE, "free"),
        (ANSI_YELLOW, "drained"),
    ]

    NODE_LEGEND = [
        (ANSI_WHITE, "has your GPU"),
        (ANSI_RED, "fully used"),
        (ANSI_GREEN, "mixed"),
        (ANSI_BLUE, "all free"),
        (ANSI_YELLOW, "drained"),
    ]

    SHORTCUTS = [
        ("gg", "go to top"),
        ("G", "go to bottom"),
        ("j \u2193", "move down"),
        ("k \u2191", "move up"),
        ("/ n", "search node"),
        ("? u", "search user"),
        ("p", "search partition"),
        ("m", "open nvitop"),
        ("c", "copy nvitop cmd"),
        ("esc", "clear filter"),
        ("q", "quit"),
    ]

    def render_shortcuts(self, has_filter: bool = False) -> Text:
        text = Text()

        # Legend
        gpu_lines = self.GPU_LEGEND
        node_lines = self.NODE_LEGEND
        max_rows = max(len(gpu_lines), len(node_lines))
        node_desc_w = max(len(d) for _, d in node_lines)
        # Column widths: " Aa desc" for Node, then gap, " ▮ desc" for GPU
        node_col_w = 3 + node_desc_w  # "Aa " + desc

        node_header = " Node Colors".ljust(node_col_w + 1)
        text.append(node_header, style=ANSI_BRIGHT_BLACK)
        text.append("    ", style=ANSI_BRIGHT_BLACK)
        text.append("GPU Colors\n", style=ANSI_BRIGHT_BLACK)

        for i in range(max_rows):
            line = Text()
            if i < len(node_lines):
                color, desc = node_lines[i]
                line.append(" Aa ", style=color)
                line.append(desc.ljust(node_desc_w), style=ANSI_BLUE)
            else:
                line.append(" " * (node_col_w + 1))
            line.append("    ")
            if i < len(gpu_lines):
                color, desc = gpu_lines[i]
                line.append(f"{BLOCK_SMALL}{BLOCK_SMALL} ", style=color)
                line.append(desc, style=ANSI_BLUE)
            text.append_text(line)
            text.append("\n")

        text.append("\n")

        # Shortcuts
        shortcuts = [(k, d) for k, d in self.SHORTCUTS
                     if k != "esc" or has_filter]
        max_key = max((len(k) for k, _ in shortcuts), default=3)
        for key, desc in shortcuts:
            line = Text()
            line.append(f" {key:>{max_key}s}", style=ANSI_BLUE)
            line.append(" - ", style=ANSI_BRIGHT_BLACK)
            line.append(desc, style=ANSI_BRIGHT_BLACK)
            text.append_text(line)
            text.append("\n")
        return text


class StatusBarWidget(Static):
    """Status bar at bottom — shows commands or search input."""

    DEFAULT_CSS = """
    StatusBarWidget {
        height: 1;
    }
    """

    def render_commands(self, filter_mode: str = "", filter_text: str = "",
                        has_filter: bool = False) -> Text:
        if filter_mode:
            label = {"user": "user", "node": "node", "partition": "partition"}.get(filter_mode, filter_mode)
            t = Text(style=f"{ANSI_BLUE} on {ANSI_BRIGHT_BLACK}")
            t.append(f" Filter by {label}: {filter_text}\u2588")
            t.pad_right(200)
            return t

        t = Text(style=f"on {ANSI_BRIGHT_BLACK}")
        if has_filter:
            t.append(f" filtered: {filter_text}", style=ANSI_BLUE)
        t.pad_right(200)
        return t
