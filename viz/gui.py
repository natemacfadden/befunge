# =============================================================================
#    Copyright (C) 2026  Nate MacFadden
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Tkinter GUI for stepping through Befunge programs
# -----------------------------------------------------------------------------

import os, sys, tkinter as tk
from tkinter import filedialog, font as tkfont

import numpy as np

# Allow `python viz/gui.py` directly, in addition to import from befunge.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from befunge import (
    W, H, STACK_CAP, OUTPUT_CAP, str_to_grid, new_state, _run_core,
    _VISITED_DUMMY,
    S_SP, S_OUT_LEN, S_X, S_Y, S_DX, S_DY, S_STRING_MODE,
)


class Interpreter:
    """Drives befunge._run_core one step at a time so the GUI can show
    each instruction's effect. Carries no dispatch logic of its own."""

    def __init__(self, src=""):
        self.load(src)

    def load(self, src):
        self._grid    = str_to_grid(src)
        self._stack   = np.zeros(STACK_CAP,  dtype=np.int64)
        self._out_buf = np.zeros(OUTPUT_CAP, dtype=np.int32)
        self.state    = new_state()
        self.halted   = False
        self.error    = None

    def snapshot(self):
        sp = int(self.state[S_SP])
        n = int(self.state[S_OUT_LEN])
        return (self._grid.copy(),
                self._stack[:sp].copy(),
                self._out_buf[:n].copy(),
                self.state.copy(),
                self.halted,
                self.error)

    def restore(self, snap):
        g, s, ob, st, h, e = snap
        self._grid[:] = g
        self.state[:] = st
        self._stack[:len(s)] = s
        self._out_buf[:len(ob)] = ob
        self.halted = h
        self.error = e

    def step(self):
        if self.halted: return
        try:
            status = _run_core(self._grid, 1, self._stack, self._out_buf, self.state, _VISITED_DUMMY)
        except Exception as e:
            self.halted = True
            self.error = str(e)
            return
        if status == 0:
            self.halted = True
        elif status == 2:
            self.halted = True
            self.error = "p: stack underflow or value out of byte range"

    @property
    def grid_array(self):
        return self._grid

    @property
    def stack(self):
        return [int(v) for v in self._stack[:int(self.state[S_SP])]]

    @property
    def output(self):
        n = int(self.state[S_OUT_LEN])
        return ''.join(chr(int(c)) for c in self._out_buf[:n])

    @property
    def x(self): return int(self.state[S_X])
    @property
    def y(self): return int(self.state[S_Y])
    @property
    def dx(self): return int(self.state[S_DX])
    @property
    def dy(self): return int(self.state[S_DY])
    @property
    def string_mode(self): return bool(self.state[S_STRING_MODE])


class BefungeGrid(tk.Frame):
    """A W×H grid of cells rendered on a Canvas, optionally editable.
    Each cell is drawn independently so it never reflows on content change."""

    GRID_LINE      = '#d0d0d0'
    BG             = 'white'
    IP_COLOR       = '#ffe066'
    CURSOR_OUTLINE = '#3399ff'
    MOD_OUTLINE    = '#dd4444'  # cells touched by `p` get this border

    def __init__(self, parent, cols=W, rows=H, cell_w=9, cell_h=14,
                 font=None, editable=False, on_change=None):
        super().__init__(parent, bd=0, padx=0, pady=0)
        self.cols      = cols
        self.rows      = rows
        self.cell_w    = cell_w
        self.cell_h    = cell_h
        self.editable  = editable
        self.on_change = on_change
        self.font      = font or tkfont.nametofont("TkFixedFont")

        cw = cols * cell_w + 1
        ch = rows * cell_h + 1
        self.canvas = tk.Canvas(self, width=cw, height=ch, bg=self.BG,
                                bd=0, highlightthickness=0, takefocus=editable)
        self.canvas.pack()

        # Grid lines (offset by 0.5 for crisp 1px lines on retina/non-retina).
        for c in range(cols + 1):
            self.canvas.create_line(c * cell_w + 0.5, 0,
                                    c * cell_w + 0.5, rows * cell_h,
                                    fill=self.GRID_LINE)
        for r in range(rows + 1):
            self.canvas.create_line(0, r * cell_h + 0.5,
                                    cols * cell_w, r * cell_h + 0.5,
                                    fill=self.GRID_LINE)

        # Per-cell state. _chars holds what's logically there; _text_ids holds
        # canvas item ids for any cell that's currently drawn.
        self._chars         = [[' '] * cols for _ in range(rows)]
        self._text_ids      = {}
        self._ip_rect       = None
        self._cursor_rect   = None
        self._cursor        = (0, 0)
        self._modified_rect = None  # single moving outline for the latest `p`

        if editable:
            self.canvas.bind('<Button-1>', self._on_click)
            self.canvas.bind('<Key>', self._on_key)
            # Tk's <<Paste>> maps to the platform shortcut. Explicit bindings
            # are belt-and-suspenders in case the virtual event doesn't fire.
            self.canvas.bind('<<Paste>>',   self._on_paste)
            self.canvas.bind('<Command-v>', self._on_paste)
            self.canvas.bind('<Control-v>', self._on_paste)
            self._draw_cursor()

    # ---- public API --------------------------------------------------------

    def load_src(self, src):
        """Load a multi-line source string into the grid."""
        for y in range(self.rows):
            row = self._chars[y]
            for x in range(self.cols):
                row[x] = ' '
        for y, line in enumerate(src.splitlines()[:self.rows]):
            for x, ch in enumerate(line[:self.cols]):
                self._chars[y][x] = ch
        self._redraw_all()

    def dump_src(self):
        return '\n'.join(''.join(row) for row in self._chars)

    def update_from_array(self, arr, mark_changes=True):
        """Sync the canvas to the (H, W) int array. We sweep all 2000 cells
        every call — equality comparisons are nearly free — but only the
        cells whose content actually changed pay the expensive tkinter
        canvas-draw cost. Cells holding values outside chr()'s range render
        as a placeholder so we don't crash on display.

        When `mark_changes` is True (the default during stepping), the red
        outline is cleared and then re-added at any cell whose content
        changed this call. Effect: the outline shows for exactly one
        refresh after a `p`, then disappears on the next one if no further
        write happened. Pass `mark_changes=False` for a silent sync (e.g.,
        after `reset`) — the outline is left alone."""
        if mark_changes and self._modified_rect is not None:
            self.canvas.delete(self._modified_rect)
            self._modified_rect = None
        for y in range(self.rows):
            row = self._chars[y]
            arr_row = arr[y]
            for x in range(self.cols):
                v = int(arr_row[x])
                ch = chr(v) if 0 <= v < 0x110000 else '?'
                if row[x] != ch:
                    row[x] = ch
                    self._draw_cell(x, y)
                    if mark_changes:
                        self._mark_modified(x, y)

    def _mark_modified(self, x, y):
        # Single outline that follows the most recent write — moves rather
        # than accumulating, so the user always sees where `p` just landed.
        cx = x * self.cell_w
        cy = y * self.cell_h
        coords = (cx + 1, cy + 1, cx + self.cell_w, cy + self.cell_h)
        if self._modified_rect is None:
            self._modified_rect = self.canvas.create_rectangle(
                *coords, fill='', outline=self.MOD_OUTLINE, width=2)
        else:
            self.canvas.coords(self._modified_rect, *coords)

    def reset_modifications(self):
        if self._modified_rect is not None:
            self.canvas.delete(self._modified_rect)
            self._modified_rect = None

    def highlight_ip(self, x, y):
        cx = x * self.cell_w
        cy = y * self.cell_h
        coords = (cx + 1, cy + 1, cx + self.cell_w, cy + self.cell_h)
        if self._ip_rect is None:
            self._ip_rect = self.canvas.create_rectangle(
                *coords, fill=self.IP_COLOR, outline='')
            self.canvas.tag_lower(self._ip_rect)
        else:
            self.canvas.coords(self._ip_rect, *coords)

    def clear_ip(self):
        if self._ip_rect is not None:
            self.canvas.delete(self._ip_rect)
            self._ip_rect = None

    # ---- internal drawing --------------------------------------------------

    NONPRINT_COLOR = '#888888'  # bytes with a standard control-picture glyph
    NOGLYPH_COLOR  = '#dd4444'  # bytes with no standard glyph — call them out

    def _draw_cell(self, x, y):
        key = (x, y)
        if key in self._text_ids:
            self.canvas.delete(self._text_ids.pop(key))
        ch = self._chars[y][x]
        o = ord(ch)

        # Pick a glyph and color for this cell. Space stays invisible; other
        # non-printable bytes get a placeholder. Bytes with no standard glyph
        # at all (the C1 control area, 128–159) are flagged in red.
        #   - 0–31 : Unicode "control pictures" (␀ ␁ ␂ … ␟) — gray
        #   - 127  : '␡' (DEL picture) — gray
        #   - 128–159 : '·' (middle dot) — RED (no standard glyph)
        if o == 32:
            return                         # plain space, nothing to draw
        if 33 <= o < 127 or 160 <= o < 256:
            glyph, color = ch, 'black'     # printable ASCII / extended ASCII
        elif 0 <= o < 32:
            glyph, color = chr(0x2400 + o), self.NONPRINT_COLOR
        elif o == 127:
            glyph, color = '␡', self.NONPRINT_COLOR
        elif 128 <= o < 160:
            glyph, color = '·', self.NOGLYPH_COLOR
        else:
            glyph, color = '?', self.NOGLYPH_COLOR

        cx = x * self.cell_w + self.cell_w / 2
        cy = y * self.cell_h + self.cell_h / 2
        self._text_ids[key] = self.canvas.create_text(
            cx, cy, text=glyph, font=self.font, fill=color)

    def _redraw_all(self):
        for tid in self._text_ids.values():
            self.canvas.delete(tid)
        self._text_ids.clear()
        for y in range(self.rows):
            for x in range(self.cols):
                self._draw_cell(x, y)

    def _draw_cursor(self):
        x, y = self._cursor
        cx = x * self.cell_w
        cy = y * self.cell_h
        coords = (cx + 1, cy + 1, cx + self.cell_w, cy + self.cell_h)
        if self._cursor_rect is None:
            self._cursor_rect = self.canvas.create_rectangle(
                *coords, outline=self.CURSOR_OUTLINE, width=2)
        else:
            self.canvas.coords(self._cursor_rect, *coords)

    # ---- editing -----------------------------------------------------------

    def _on_click(self, event):
        x = int(event.x) // self.cell_w
        y = int(event.y) // self.cell_h
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self._cursor = (x, y)
            self._draw_cursor()
            self.canvas.focus_set()

    def _on_key(self, event):
        x, y = self._cursor
        ks = event.keysym
        if ks == 'Left':
            x = max(0, x - 1)
        elif ks == 'Right':
            x = min(self.cols - 1, x + 1)
        elif ks == 'Up':
            y = max(0, y - 1)
        elif ks == 'Down':
            y = min(self.rows - 1, y + 1)
        elif ks == 'BackSpace':
            x = max(0, x - 1)
            self._chars[y][x] = ' '
            self._draw_cell(x, y)
            if self.on_change: self.on_change()
        elif ks == 'Delete':
            self._chars[y][x] = ' '
            self._draw_cell(x, y)
            if self.on_change: self.on_change()
        elif ks == 'Return':
            x = 0
            y = min(self.rows - 1, y + 1)
        elif event.char and len(event.char) == 1 and 32 <= ord(event.char) < 127:
            self._chars[y][x] = event.char
            self._draw_cell(x, y)
            if self.on_change: self.on_change()
            x += 1
            if x >= self.cols:
                x = 0
                y = min(self.rows - 1, y + 1)
        self._cursor = (x, y)
        self._draw_cursor()

    def _on_paste(self, event=None):
        try:
            text = self.canvas.clipboard_get()
        except tk.TclError:
            return 'break'
        start_x, start_y = self._cursor
        x, y = start_x, start_y
        changed = False
        for ch in text:
            if ch == '\r':
                continue
            if ch == '\n':
                y += 1
                x = start_x
                continue
            if y >= self.rows:
                break
            if x >= self.cols:
                continue  # don't auto-wrap; let the line truncate
            if 32 <= ord(ch) < 127:
                self._chars[y][x] = ch
                self._draw_cell(x, y)
                changed = True
                x += 1
        if changed:
            self._cursor = (min(x, self.cols - 1), min(y, self.rows - 1))
            self._draw_cursor()
            if self.on_change:
                self.on_change()
        return 'break'


class App:
    def __init__(self, src=""):
        """`src` (optional) is a Befunge source string to preload into the
        editor — convenient for `App(df.iloc[i]['pruned_program']).run()`
        from a notebook."""
        self.interp         = Interpreter()
        self.history        = []
        self.running        = False
        self.delay          = 200
        self.steps_per_tick = 1

        self.root = tk.Tk()
        self.root.title("Befunge")

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=9)

        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        # Fixed-height header rows on both sides so the grids below them start
        # at the same Y. Tall enough to accommodate a tk.Button on macOS.
        HEADER_H = 32

        # LEFT: editor header
        top = tk.Frame(left, height=HEADER_H)
        top.pack(fill="x", anchor="w")
        top.pack_propagate(False)
        tk.Label(top, text="Editor", font=("Sans", 11, "bold")).pack(side="left")
        tk.Button(top, text="Load...", command=self.load_file).pack(side="right")
        tk.Button(top, text="Clear",   command=self.clear).pack(side="right", padx=(0, 4))
        self.editor_grid = BefungeGrid(left, W, H, font=mono, editable=True,
                                       on_change=self.reset)
        self.editor_grid.pack(anchor="w")

        # RIGHT: execution header (status moves up here so it shares the row
        # with the Execution label, matching the editor's label+buttons row).
        exec_top = tk.Frame(right, height=HEADER_H)
        exec_top.pack(fill="x", anchor="w")
        exec_top.pack_propagate(False)
        tk.Label(exec_top, text="Execution", font=("Sans", 11, "bold")).pack(side="left")
        self.status = tk.Label(exec_top, text="", font=("Sans", 10), anchor="w")
        self.status.pack(side="left", padx=(8, 0))

        self.display_grid = BefungeGrid(right, W, H, font=mono)
        self.display_grid.pack(anchor="w")

        tk.Label(right, text="Stack (bottom -> top)", font=("Sans", 10, "bold"),
                 anchor="w").pack(fill="x")
        self.stack_view = tk.Text(right, width=W, height=3, font=mono,
                                  wrap="word", state="disabled")
        self.stack_view.pack(fill="x")

        tk.Label(right, text="Output", font=("Sans", 10, "bold"),
                 anchor="w").pack(fill="x", pady=(4, 0))
        out_frame = tk.Frame(right)
        out_frame.pack(fill="x")
        self.output_view = tk.Text(out_frame, width=W, height=8, font=mono,
                                   wrap="char", state="disabled")
        self.output_view.pack(side="left", fill="both", expand=True)
        out_scroll = tk.Scrollbar(out_frame, command=self.output_view.yview)
        out_scroll.pack(side="right", fill="y")
        self.output_view.config(yscrollcommand=out_scroll.set)

        ctrl = tk.Frame(left)
        ctrl.pack(pady=6, anchor="w")
        tk.Button(ctrl, text="Reset",     command=self.reset).pack(side="left", padx=2)
        tk.Button(ctrl, text="Go",        command=self.go).pack(side="left", padx=2)
        tk.Button(ctrl, text="Stop",      command=self.stop).pack(side="left", padx=2)
        tk.Button(ctrl, text="Slower",    command=self.slower).pack(side="left", padx=(12, 2))
        tk.Button(ctrl, text="Faster",    command=self.faster).pack(side="left", padx=2)
        tk.Button(ctrl, text="Step Back", command=self.step_back).pack(side="left", padx=(12, 2))
        tk.Button(ctrl, text="Step Fwd",  command=self.step).pack(side="left", padx=2)
        self.speed_label = tk.Label(ctrl, text="", font=("Sans", 10), width=16)
        self.speed_label.pack(side="left", padx=4)

        self._update_speed_label()
        if src:
            self.editor_grid.load_src(src)
        self.reset()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Befunge", "*.bf"), ("All files", "*.*")])
        if not path: return
        with open(path) as f:
            src = f.read()
        self.editor_grid.load_src(src)
        self.reset()

    def clear(self):
        self.editor_grid.load_src("")
        self.reset()

    def reset(self):
        self.running = False
        src = self.editor_grid.dump_src()
        self.interp.load(src)
        self.history.clear()
        self.display_grid.reset_modifications()
        self.refresh(mark_changes=False)

    def step(self):
        if self.interp.halted: return
        self.history.append(self.interp.snapshot())
        if len(self.history) > 1000:
            self.history.pop(0)
        self.interp.step()
        self.refresh()

    def step_back(self):
        self.running = False
        if not self.history: return
        self.interp.restore(self.history.pop())
        self.refresh()

    def go(self):
        if self.interp.halted: return
        self.running = True
        self._tick()

    def stop(self):
        self.running = False

    def slower(self):
        if self.steps_per_tick > 1:
            self.steps_per_tick //= 2
        else:
            self.delay = min(2000, self.delay * 2)
        self._update_speed_label()

    def faster(self):
        if self.delay > 1:
            self.delay = max(1, self.delay // 2)
        else:
            self.steps_per_tick = min(100000, self.steps_per_tick * 2)
        self._update_speed_label()

    def _update_speed_label(self):
        rate = int(self.steps_per_tick * 1000 / self.delay)
        self.speed_label.config(text=f"{rate:,} steps/s")

    def _tick(self):
        if not self.running or self.interp.halted: return
        for _ in range(self.steps_per_tick):
            self.history.append(self.interp.snapshot())
            if len(self.history) > 1000:
                self.history.pop(0)
            self.interp.step()
            if self.interp.halted:
                break
        self.refresh()
        if not self.interp.halted:
            self.root.after(self.delay, self._tick)

    def refresh(self, mark_changes=True):
        self.display_grid.update_from_array(self.interp.grid_array, mark_changes=mark_changes)
        self.display_grid.highlight_ip(self.interp.x, self.interp.y)

        arrow = {(1,0): ">", (-1,0): "<", (0,1): "v", (0,-1): "^"}.get(
            (self.interp.dx, self.interp.dy), "?")
        mode = "STRING" if self.interp.string_mode else "normal"
        if self.interp.error:
            tail = f" [ERROR: {self.interp.error}]"
        elif self.interp.halted:
            tail = " [HALTED]"
        else:
            tail = ""
        self.status.config(
            text=f"IP: ({self.interp.x}, {self.interp.y}) {arrow}   mode: {mode}{tail}")

        self.stack_view.configure(state="normal")
        self.stack_view.delete("1.0", "end")
        self.stack_view.insert("1.0", " ".join(str(v) for v in self.interp.stack))
        self.stack_view.configure(state="disabled")

        self.output_view.configure(state="normal")
        self.output_view.delete("1.0", "end")
        self.output_view.insert("1.0", self.interp.output)
        self.output_view.see("end")
        self.output_view.configure(state="disabled")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
