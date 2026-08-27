"""Progress reporting that stays legible wherever the output lands.

Three environments, three behaviours:
  - a real terminal: tqdm, redrawn in place;
  - a Jupyter/Colab/Kaggle *kernel* (stderr is not a tty, but a live widget
    renders): tqdm.auto's notebook bar;
  - everything else — `!python ...` subshells, log files, nohup, CI — plain
    newline-delimited lines every few seconds, greppable afterwards.
"""

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
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


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
        self.n = initial
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
        rate = self.n / elapsed if elapsed > 0 else 0.0
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


def bar(total=None, desc: str = "", unit: str = "it", initial: int = 0):
    """Progress reporter appropriate to where the output is going."""
    if not PROGRESS:
        return _Stub()
    if sys.stderr.isatty() or _in_notebook():
        try:
            from tqdm.auto import tqdm

            return tqdm(total=total, desc=desc, unit=unit, initial=initial,
                        dynamic_ncols=True, smoothing=0.05)
        except ImportError:
            pass
    return _LineProgress(total, desc, unit, initial)
