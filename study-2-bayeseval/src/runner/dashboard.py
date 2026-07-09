"""Rich Live dashboard matching the WGD panel style."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class _TaskState:
    total: int
    done: int = 0
    errors: int = 0
    tok_in: int = 0
    tok_out: int = 0
    brier_sum: float = 0.0
    brier_n: int = 0
    progress_task_id: int | None = None


class Dashboard:
    def __init__(self, title: str, show_price: bool = False,
                 pricing: dict[str, tuple[float, float]] | None = None,
                 static: bool = False):
        self.title = title
        self.show_price = show_price
        self.pricing = pricing or {}
        self._tasks: dict[tuple[str, str], _TaskState] = {}
        self._start = time.monotonic()
        self._recent: deque[str] = deque(maxlen=6)
        columns = [
            TextColumn("[progress.description]{task.description:<40}"),
            BarColumn(bar_width=32),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[speed]:>5.1f} req/s[/cyan]"),
        ]
        if not static:
            columns.insert(0, SpinnerColumn())
            columns.append(TimeRemainingColumn())
            columns.append(TimeElapsedColumn())
        self._progress = Progress(*columns)
        self._overall_task = self._progress.add_task(
            "[bold white]Overall[/bold white]", total=0, speed=0.0
        )
        self._live: Live | None = None

    def register(self, domain: str, model: str, total: int) -> None:
        key = (domain, model)
        short = model.split("/")[-1]
        pid = self._progress.add_task(f"[dim]{domain} · {short}[/dim]", total=total, speed=0.0)
        self._tasks[key] = _TaskState(total=total, progress_task_id=pid)
        overall_total = sum(s.total for s in self._tasks.values())
        self._progress.update(self._overall_task, total=overall_total)

    def record(self, domain: str, model: str, *, error: bool,
               brier: float | None, question_id: str = "",
               answer: str | None = None, tok_in: int = 0, tok_out: int = 0) -> None:
        st = self._tasks[(domain, model)]
        st.done += 1
        st.tok_in += tok_in
        st.tok_out += tok_out
        if error:
            st.errors += 1
        if brier is not None and brier == brier:
            st.brier_sum += brier
            st.brier_n += 1

        short = model.split("/")[-1]
        if error:
            self._recent.append(
                f"[red]✗[/red] [dim]{question_id}[/dim] [cyan]{short}[/cyan] → [red]error[/red]"
            )
        else:
            ans_str = str(answer)[:30] if answer else "?"
            self._recent.append(
                f"[green]✓[/green] [dim]{question_id}[/dim] [cyan]{short}[/cyan] → [bold]{ans_str}[/bold]"
            )

        elapsed = time.monotonic() - self._start
        done_total = sum(s.done for s in self._tasks.values())
        speed = done_total / elapsed if elapsed > 0 else 0.0
        self._progress.update(self._overall_task, completed=done_total, speed=speed)
        self._progress.update(st.progress_task_id, completed=st.done, speed=speed)

    def _domain_panels(self) -> Columns:
        by_domain: dict[str, list[tuple[str, _TaskState]]] = {}
        for (domain, model), st in self._tasks.items():
            by_domain.setdefault(domain, []).append((model, st))
        panels = []
        for domain in sorted(by_domain):
            t = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
            t.add_column("Model", style="cyan", no_wrap=True)
            t.add_column("Done", justify="right")
            t.add_column("Err", justify="right")
            t.add_column("Tok In", justify="right")
            t.add_column("Tok Out", justify="right")
            t.add_column("Price" if self.show_price else "Brier", justify="right")
            for model, st in by_domain[domain]:
                short = model.split("/")[-1]
                err_str = f"[red]{st.errors}[/red]" if st.errors else "[dim]0[/dim]"
                if self.show_price:
                    in_rate, out_rate = self.pricing.get(model, (0, 0))
                    price = st.tok_in * in_rate + st.tok_out * out_rate
                    last_col = f"[green]${price:.3f}[/green]"
                else:
                    last_col = f"{st.brier_sum / st.brier_n:.4f}" if st.brier_n else "—"
                t.add_row(
                    short, str(st.done), err_str,
                    f"{st.tok_in:,}", f"{st.tok_out:,}", last_col,
                )
            panels.append(Panel(t, title=f"[bold]{domain}[/bold]", border_style="dim"))
        return Columns(panels, equal=True)

    def _recent_text(self) -> Group:
        if not self._recent:
            return Group(Text("Waiting for first response…", style="dim"))
        return Group(*[Text.from_markup(line) for line in list(self._recent)])

    def __rich__(self) -> Panel:
        n_models = len(set(m for _, m in self._tasks))
        n_domains = len(set(d for d, _ in self._tasks))
        total = sum(s.total for s in self._tasks.values())
        if self.show_price:
            total_price = sum(
                st.tok_in * self.pricing.get(model, (0, 0))[0]
                + st.tok_out * self.pricing.get(model, (0, 0))[1]
                for (_, model), st in self._tasks.items()
            )
            subtitle = (
                f"[dim]{n_models} model(s) · {n_domains} domain(s) "
                f"· {total} requests · [green]${total_price:.2f}[/green] total[/dim]"
            )
        else:
            subtitle = f"[dim]{n_models} model(s) · {n_domains} domain(s) · {total} requests[/dim]"
        return Panel(
            Group(
                self._progress,
                Text(""),
                self._domain_panels(),
                Text(""),
                Text("Recent", style="bold"),
                self._recent_text(),
            ),
            title=f"[bold cyan]{self.title}[/bold cyan]",
            border_style="cyan",
            subtitle=subtitle,
        )

    def __enter__(self):
        self._live = Live(self, refresh_per_second=4, auto_refresh=True)
        self._live.__enter__()
        return self

    def refresh(self) -> None:
        if self._live:
            self._live.refresh()

    def __exit__(self, *exc):
        self.refresh()
        assert self._live is not None
        self._live.__exit__(*exc)
