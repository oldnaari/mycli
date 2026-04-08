"""Slurm monitor TUI application."""

import base64
import os
import subprocess
import sys
import threading

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from .colors import ANSI_BRIGHT_BLACK
from .data import apply_nvidia_smi, fetch_nvidia_smi, refresh_slurm_state, ClusterState
from .demo import build_demo_state
from .widgets import (
    NodeDetailWidget, NodeListWidget, PriorityWidget, ShortcutsWidget, StatusBarWidget,
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
    #right-panel {
        width: 1fr;
    }
    #node-detail {
        height: 1fr;
    }
    #shortcuts {
        height: auto;
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
        self._pending_g = False
        self._left_width: int = 40

    def compose(self) -> ComposeResult:
        with Horizontal(id="main-area"):
            with Vertical(id="left-panel"):
                yield PriorityWidget(id="priority")
                yield NodeListWidget(id="node-list")
            with Vertical(id="right-panel"):
                yield NodeDetailWidget(id="node-detail")
                yield ShortcutsWidget(id="shortcuts")
        yield StatusBarWidget(id="status-bar")

    def on_mount(self) -> None:
        self.ansi_color = True
        self.current_user = os.environ.get("USER", "unknown")
        if self.demo:
            self.cluster_state = build_demo_state()
            self.update_all()
        else:
            self.set_interval(3, self._check_dirty)
            t1 = threading.Thread(target=self._slurm_refresh_loop, daemon=True)
            t1.start()
            t2 = threading.Thread(target=self._gpu_poll_loop, daemon=True)
            t2.start()
        self.query_one("#node-list", NodeListWidget).focus()

    def on_unmount(self) -> None:
        self._stop_event.set()

    def _check_dirty(self) -> None:
        if self._dirty:
            self._dirty = False
            self.update_all()

    def _slurm_refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = refresh_slurm_state(self.current_user)
                old = self.cluster_state
                if old is not None:
                    for name, node in state.nodes.items():
                        old_node = old.nodes.get(name)
                        if old_node is None:
                            continue
                        for gpu in node.gpus:
                            if gpu.index < len(old_node.gpus):
                                old_gpu = old_node.gpus[gpu.index]
                                gpu.utilization = old_gpu.utilization
                                gpu.mem_used = old_gpu.mem_used
                                gpu.mem_total = old_gpu.mem_total
                self.cluster_state = state
                self._dirty = True
            except Exception:
                pass
            self._stop_event.wait(30)

    def _gpu_poll_loop(self) -> None:
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
        max_type = max(len(n.partition) for n in nodes)
        max_gpus = max(n.total_gpus for n in nodes)
        type_col = (max_type + 4) if max_type > 0 else 0
        min_width = max_name + 4 + type_col + max_gpus + 2
        half_screen = self.size.width // 2
        return max(min_width, half_screen)

    def update_all(self) -> None:
        state = self.cluster_state
        if state is None:
            return

        needed = self._compute_left_width(state)
        if needed != self._left_width:
            self._left_width = needed
            lp = self.query_one("#left-panel")
            lp.styles.width = needed

        pw = self.query_one("#priority", PriorityWidget)
        pw.update(pw.render_content(state, self._left_width))

        nl = self.query_one("#node-list", NodeListWidget)
        nl.update_nodes(state)

        self._update_detail_only()
        self._update_shortcuts()

    def _update_detail_only(self) -> None:
        """Update only the right panel + nvidia cmd. Lightweight."""
        nl = self.query_one("#node-list", NodeListWidget)
        nd = self.query_one("#node-detail", NodeDetailWidget)
        node = nl.get_selected_node()
        nd.update(nd.render_detail(node, self.current_user))

    def _update_shortcuts(self) -> None:
        nl = self.query_one("#node-list", NodeListWidget)
        sw = self.query_one("#shortcuts", ShortcutsWidget)
        sw.update(sw.render_shortcuts(bool(nl.filter_text)))

    def _update_status(self) -> None:
        sb = self.query_one("#status-bar", StatusBarWidget)
        nl = self.query_one("#node-list", NodeListWidget)
        sb.update(sb.render_commands(self.filter_mode, self.filter_text,
                                     bool(nl.filter_text)))

    def _apply_filter(self) -> None:
        """Apply current filter text to node list and refresh it."""
        nl = self.query_one("#node-list", NodeListWidget)
        nl.filter_text = self.filter_text
        nl.filter_mode = self.filter_mode
        if self.cluster_state:
            nl.update_nodes(self.cluster_state)

    def on_key(self, event) -> None:
        if self.filter_mode:
            if event.key == "escape":
                self.filter_mode = ""
                self.filter_text = ""
                self._apply_filter()
                self._update_status()
                self._update_shortcuts()
                self._update_detail_only()
                event.prevent_default()
                return
            if event.key == "enter":
                self.filter_mode = ""
                self._update_status()
                event.prevent_default()
                return
            if event.key == "backspace":
                self.filter_text = self.filter_text[:-1]
            elif event.is_printable and event.character:
                self.filter_text += event.character
            # Lightweight: only update status bar text per keystroke
            self._update_status()
            # Apply filter to node list directly (skip detail panel)
            self._apply_filter()
            event.prevent_default()
            return

        nl = self.query_one("#node-list", NodeListWidget)

        if event.key == "g":
            if self._pending_g:
                self._pending_g = False
                nl.move_first()
                self._update_detail_only()
            else:
                self._pending_g = True
            event.prevent_default()
            return
        self._pending_g = False

        if event.key in ("j", "down"):
            nl.move_down()
            self._update_detail_only()
            event.prevent_default()
        elif event.key in ("k", "up"):
            nl.move_up()
            self._update_detail_only()
            event.prevent_default()
        elif event.key == "G":
            nl.move_last()
            self._update_detail_only()
            event.prevent_default()
        elif event.key in ("u", "question_mark"):
            self.filter_mode = "user"
            self.filter_text = ""
            self._update_status()
            event.prevent_default()
        elif event.key in ("n", "slash"):
            self.filter_mode = "node"
            self.filter_text = ""
            self._update_status()
            event.prevent_default()
        elif event.key == "p":
            self.filter_mode = "partition"
            self.filter_text = ""
            self._update_status()
            event.prevent_default()
        elif event.key == "escape":
            if nl.filter_text:
                nl.filter_text = ""
                nl.filter_mode = ""
                self.filter_text = ""
                if self.cluster_state:
                    nl.update_nodes(self.cluster_state)
                self._update_detail_only()
                self._update_status()
                self._update_shortcuts()
            event.prevent_default()
        elif event.key == "c":
            node = nl.get_selected_node()
            if node:
                cmd = f"ssh {node.name} -t python3.11 -m nvitop"
                encoded = base64.b64encode(cmd.encode()).decode()
                osc = f"\x1b]52;c;{encoded}\x07"
                try:
                    with open("/dev/tty", "w") as tty:
                        tty.write(osc)
                        tty.flush()
                except OSError:
                    pass
            event.prevent_default()
        elif event.key == "m":
            node = nl.get_selected_node()
            if node:
                cmd = ["ssh", node.name, "-t", "python3.11", "-m", "nvitop"]
                with self.suspend():
                    subprocess.run(cmd)
            event.prevent_default()
        elif event.key == "q":
            self.exit()
            event.prevent_default()


def main():
    demo = "--demo" in sys.argv
    app = SlurmMonitorApp(demo=demo)
    app.run()
