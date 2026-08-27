"""Progress reporting that stays legible wherever the output lands.

Three environments, three behaviours:
  - a real terminal: tqdm, redrawn in place;
  - a Jupyter/Colab/Kaggle *kernel* (stderr is not a tty, but a live widget
    renders): tqdm.auto's notebook bar;
  - everything else — `!python ...` subshells, log files, nohup, CI — plain
    newline-delimited lines every few seconds, greppable afterwards.
"""

import os
import sys
import time

# Flipped off by --no-progress. Module-level because every phase draws through
# bar() and threading a flag into each of them would be noise.
PROGRESS = True


def _in_notebook() -> bool:
    """True inside a Jupyter/Colab/Kaggle kernel (not a `!python` subprocess).

    stderr is not a tty there, so the isatty() check alone would fall back to
    plain lines — but tqdm.auto renders a live widget in a kernel, which is
    strictly better.
    """
    if "ipykernel" not in sys.modules:
        return False
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    # Colab's shell subclasses ZMQInteractiveShell under another name ("Shell"),
    # so walk the MRO rather than matching the leaf class name.
    return any(c.__name__ == "ZMQInteractiveShell" for c in type(shell).__mro__)


def _hosted_notebook_env() -> bool:
    """Colab or Kaggle, *including* their `!python ...` subshells.

    Colab runs `!` commands through a pty, so isatty() is True and the terminal
    branch would pick tqdm — but the cell renderer does not collapse tqdm's
    carriage-return redraws; they accumulate into one enormously wide line
    that scrolls off screen and reads as "there is no progress bar at all".
    These env vars are inherited by the subprocess, where plain periodic lines
    are legible in both frontends.
    """
    return any(k in os.environ for k in (
        "COLAB_RELEASE_TAG", "COLAB_BACKEND_VERSION",   # Colab
        "KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE",    # Kaggle
    ))


def _hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 \
        else f"{s // 60:d}:{s % 60:02d}"


class _LineProgress:
    """Newline-delimited progress, for when stderr is not a terminal.

    tqdm redraws in place with carriage returns. A terminal collapses those onto
    one line, but Colab's `!command` renderer, log files, nohup and CI do not —
    they concatenate every redraw into a single enormously wide line that scrolls
    off screen, which reads as "there is no progress bar". One plain line every
    few seconds is legible everywhere and greppable afterwards.
    """

    def __init__(self, total, desc, unit, initial=0, every=10.0):
        self.total, self.desc, self.unit = total, desc, unit
        self.n = self.initial = initial
        self.start = self.last = time.time()
        self.every = every
        self.postfix = ""
        self.shown = -1  # count at the last emit, so close() can skip a repeat

    def update(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last >= self.every:
            self._emit(now)

    def set_postfix(self, **kw):
        self.postfix = "  " + "  ".join(f"{k} {v}" for k, v in kw.items())
        self._emit(time.time())  # fresh numbers just arrived; show them now

    def _emit(self, now, final=False):
        self.last, self.shown = now, self.n
        elapsed = now - self.start
        # rate over *this run's* work only: on --resume, n starts at initial,
        # and counting that pre-done work would inflate the rate and shrink the
        # ETA by however far the previous session got.
        rate = (self.n - self.initial) / elapsed if elapsed > 0 else 0.0
        head = f"{self.desc:<12}" if self.desc else ""
        if self.total:
            eta = "" if final or rate <= 0 else f"  eta {_hms((self.total - self.n) / rate)}"
            body = (f"{100.0 * self.n / self.total:5.1f}%  {self.n}/{self.total} {self.unit}"
                    f"  {rate:.0f} {self.unit}/s  {_hms(elapsed)}{eta}")
        else:
            body = f"{self.n} {self.unit}  {rate:.0f} {self.unit}/s  {_hms(elapsed)}"
        print(head + body + self.postfix, file=sys.stderr, flush=True)

    def write(self, s):
        print(s, flush=True)

    def close(self):
        if self.n != self.shown:
            self._emit(time.time(), final=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class _Stub:
    """--no-progress: count silently, pass messages straight through."""

    n = 0

    def update(self, k=1):
        self.n += k

    def set_postfix(self, **kw):
        pass

    def write(self, s):
        print(s, flush=True)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def restart_timer(pb) -> None:
    """Re-stamp a reporter's clock to *now* — called once jit compilation is
    done, so every later rate and ETA describes steady-state training instead
    of amortising the first step's multi-minute compile across the whole run."""
    now = time.time()
    if isinstance(pb, _LineProgress):
        pb.start = pb.last = now
        pb.initial = pb.n
    elif hasattr(pb, "start_t"):  # tqdm
        pb.initial = pb.n
        pb.start_t = now
        pb.last_print_t = now
    # _Stub keeps no clock


def raw_terminal() -> bool:
    """True when stderr is a real terminal that renders carriage-return redraws.

    Gates third-party CR-based bars (e.g. the tokenizers Rust trainer's), which
    bar() cannot wrap: Colab's pty-backed `!` cells claim to be a tty but
    accumulate the redraws into one unreadable line.
    """
    return PROGRESS and sys.stderr.isatty() and not _hosted_notebook_env()


def bar(total=None, desc: str = "", unit: str = "it", initial: int = 0):
    """Progress reporter appropriate to where the output is going.

    Precedence: a live kernel gets tqdm's notebook widget; a real terminal
    gets tqdm — unless it is Colab/Kaggle's pty-backed `!` cell, whose fake
    tty cannot render redraws; everything else gets periodic plain lines.
    """
    if not PROGRESS:
        return _Stub()
    if _in_notebook() or (sys.stderr.isatty() and not _hosted_notebook_env()):
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, unit=unit, initial=initial,
                        dynamic_ncols=True, smoothing=0.05)
        except ImportError:
            pass
    return _LineProgress(total, desc, unit, initial)
