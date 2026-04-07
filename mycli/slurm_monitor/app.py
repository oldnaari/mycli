"""Slurm monitor TUI application."""

import os
import sys
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from .colors import ANSI_BRIGHT_BLACK
from .data import apply_nvidia_smi, fetch_nvidia_smi, refresh_slurm_state, ClusterState
from .demo import build_demo_state
from .widgets import (
    NodeDetailWidget, NodeListWidget, PriorityWidget, StatusBarWidget,
)


class SlurmMonitorApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #main-area {
        height: 1fr;
    }
    #left-panel {
        width: 40;
    }
    #priority {
        height: auto;
        max-height: 12;
    }
    #node-list {
        height: 1fr;
    }
    #node-detail {
        width: 1fr;
    }
    #nvidia-cmd {
        height: 1;
    }
    #status-bar {
        height: 1;
    }
    """

    def __init__(self, demo: bool = False, **kw):
        super().__init__(**kw)
        self.demo = demo
        self.cluster_state: ClusterState | None = None
        self.filter_mode: str = ""
        self.filter_text: str = ""
        self._stop_event = threading.Event()
        self._dirty = False
        self._filter_dirty = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield PriorityWidget(id="priority")
                yield NodeListWidget(id="node-list")
            yield NodeDetailWidget(id="node-detail")
        yield Static(id="nvidia-cmd")
        yield StatusBarWidget(id="status-bar")

    def on_mount(self) -> None:
        self.ansi_color = True
        self.current_user = os.environ.get("USER", "unknown")
        if self.demo:
            self.cluster_state = build_demo_state()
            self.update_all()
        else:
            # Timer polls for dirty flag — only way background data reaches the UI
            self.set_interval(2, self._check_dirty)
            # Slurm refresh thread (sinfo + squeue every 30s)
            t1 = threading.Thread(target=self._slurm_refresh_loop, daemon=True)
            t1.start()
            # GPU poll thread (nvidia-smi continuously)
            t2 = threading.Thread(target=self._gpu_poll_loop, daemon=True)
            t2.start()
        # Fast timer for filter debounce (always active)
        self.set_interval(0.2, self._check_filter_dirty)
        self.query_one("#node-list", NodeListWidget).focus()

    def on_unmount(self) -> None:
        self._stop_event.set()

    def _check_dirty(self) -> None:
        """Timer callback: re-render UI if background threads have new data."""
        if self._dirty:
            self._dirty = False
            self.update_all()

    def _check_filter_dirty(self) -> None:
        """Fast timer: apply pending filter changes."""
        if self._filter_dirty:
            self._filter_dirty = False
            nl = self.query_one("#node-list", NodeListWidget)
            nl.filter_text = self.filter_text
            nl.filter_mode = self.filter_mode
            if self.cluster_state:
                nl.update_nodes(self.cluster_state)
            self.update_detail()

    def _slurm_refresh_loop(self) -> None:
        """Background thread: periodically refresh sinfo + squeue."""
        while not self._stop_event.is_set():
            try:
                state = refresh_slurm_state(self.current_user)
                self.cluster_state = state
                self._dirty = True
            except Exception:
                pass
            self._stop_event.wait(30)

    def _gpu_poll_loop(self) -> None:
        """Background thread: continuously cycle through nodes fetching nvidia-smi."""
        while not self._stop_event.is_set():
            state = self.cluster_state
            if state is None or not state.nodes:
                self._stop_event.wait(1)
                continue
            node_names = list(state.nodes.keys())
            for node_name in node_names:
                if self._stop_event.is_set():
                    return
                if self.cluster_state is not state:
                    break
                try:
                    name, output = fetch_nvidia_smi(node_name)
                    if output is not None and self.cluster_state is not None:
                        apply_nvidia_smi(self.cluster_state, name, output)
                        self._dirty = True
                except Exception:
                    pass

    def _compute_left_width(self, state: ClusterState) -> int:
        nodes = list(state.nodes.values())
        if not nodes:
            return 40
        max_name = max(len(n.name) for n in nodes)
        max_type = max(len(n.gpu_type) for n in nodes)
        max_gpus = max(n.total_gpus for n in nodes)
        type_col = (max_type + 4) if max_type > 0 else 0
        min_width = max_name + 4 + type_col + max_gpus + 2
        half_screen = self.size.width // 2
        return max(min_width, half_screen)

    def update_all(self) -> None:
        state = self.cluster_state
        if state is None:
            return

        lp = self.query_one("#left-panel")
        needed = self._compute_left_width(state)
        lp.styles.width = needed

        pw = self.query_one("#priority", PriorityWidget)
        pw.update(pw.render_content(state, needed))

        nl = self.query_one("#node-list", NodeListWidget)
        nl.update_nodes(state)

        self.update_detail()
        self.update_status()

    def update_detail(self) -> None:
        nl = self.query_one("#node-list", NodeListWidget)
        nd = self.query_one("#node-detail", NodeDetailWidget)
        node = nl.get_selected_node()
        nd.update(nd.render_detail(node, self.current_user))

        cmd_widget = self.query_one("#nvidia-cmd", Static)
        if node:
            t = Text(style=ANSI_BRIGHT_BLACK)
            t.append(f" ssh {node.name} nvidia-smi")
            cmd_widget.update(t)
        else:
            cmd_widget.update(Text(" (no node selected)", style=ANSI_BRIGHT_BLACK))

    def update_status(self) -> None:
        sb = self.query_one("#status-bar", StatusBarWidget)
        sb.update(sb.render_commands(self.filter_mode, self.filter_text))

    def on_key(self, event) -> None:
        if self.filter_mode:
            if event.key == "escape":
                self.filter_mode = ""
                self.filter_text = ""
                self._filter_dirty = True
                self.update_status()
                event.prevent_default()
                return
            if event.key == "enter":
                self.filter_mode = ""
                self.update_status()
                event.prevent_default()
                return
            if event.key == "backspace":
                self.filter_text = self.filter_text[:-1]
            elif event.is_printable and event.character:
                self.filter_text += event.character
            # Only update status bar immediately; filter applied by debounce timer
            self._filter_dirty = True
            self.update_status()
            event.prevent_default()
            return

        nl = self.query_one("#node-list", NodeListWidget)

        if event.key in ("j", "down"):
            nl.move_down()
            self.update_detail()
            event.prevent_default()
        elif event.key in ("k", "up"):
            nl.move_up()
            self.update_detail()
            event.prevent_default()
        elif event.key in ("u", "question_mark"):
            self.filter_mode = "user"
            self.filter_text = ""
            self.update_status()
            event.prevent_default()
        elif event.key in ("n", "slash"):
            self.filter_mode = "node"
            self.filter_text = ""
            self.update_status()
            event.prevent_default()
        elif event.key == "q":
            self.exit()
            event.prevent_default()


def main():
    demo = "--demo" in sys.argv
    app = SlurmMonitorApp(demo=demo)
    app.run()
