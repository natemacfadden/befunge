import tkinter as tk
from tkinter import filedialog, simpledialog, font as tkfont
import random

W, H = 80, 25

class Interpreter:
    def __init__(self, src=""):
        self.load(src)

    def load(self, src):
        self.grid = [[' '] * W for _ in range(H)]
        for y, line in enumerate(src.splitlines()[:H]):
            for x, ch in enumerate(line[:W]):
                self.grid[y][x] = ch
        self.stack = []
        self.x, self.y, self.dx, self.dy = 0, 0, 1, 0
        self.string_mode = False
        self.output = ""
        self.halted = False

    def snapshot(self):
        return ([row[:] for row in self.grid], self.stack[:],
                self.x, self.y, self.dx, self.dy,
                self.string_mode, self.output, self.halted)

    def restore(self, snap):
        g, s, self.x, self.y, self.dx, self.dy, \
            self.string_mode, self.output, self.halted = snap
        self.grid = [row[:] for row in g]
        self.stack = s[:]

    def pop(self):
        return self.stack.pop() if self.stack else 0

    def step(self, ask_int=lambda: 0, ask_char=lambda: 0):
        if self.halted: return
        c = self.grid[self.y][self.x]
        s = self.stack

        if self.string_mode:
            if c == '"': self.string_mode = False
            else: s.append(ord(c))
        elif c.isdigit(): s.append(int(c))
        elif c == '+': s.append(self.pop() + self.pop())
        elif c == '*': s.append(self.pop() * self.pop())
        elif c == '-': a, b = self.pop(), self.pop(); s.append(b - a)
        elif c == '/': a, b = self.pop(), self.pop(); s.append(b // a if a else 0)
        elif c == '%': a, b = self.pop(), self.pop(); s.append(b % a if a else 0)
        elif c == '!': s.append(0 if self.pop() else 1)
        elif c == '`': a, b = self.pop(), self.pop(); s.append(1 if b > a else 0)
        elif c == '>': self.dx, self.dy = 1, 0
        elif c == '<': self.dx, self.dy = -1, 0
        elif c == '^': self.dx, self.dy = 0, -1
        elif c == 'v': self.dx, self.dy = 0, 1
        elif c == '?': self.dx, self.dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        elif c == '_': self.dx, self.dy = (1, 0) if self.pop() == 0 else (-1, 0)
        elif c == '|': self.dx, self.dy = (0, 1) if self.pop() == 0 else (0, -1)
        elif c == '"': self.string_mode = True
        elif c == ':': v = self.pop(); s += [v, v]
        elif c == '\\': a, b = self.pop(), self.pop(); s += [a, b]
        elif c == '$': self.pop()
        elif c == '.': self.output += str(self.pop()) + ' '
        elif c == ',': self.output += chr(self.pop() % 256)
        elif c == '#': self.x, self.y = (self.x + self.dx) % W, (self.y + self.dy) % H
        elif c == 'g': gy, gx = self.pop(), self.pop(); s.append(ord(self.grid[gy % H][gx % W]))
        elif c == 'p': py, px, v = self.pop(), self.pop(), self.pop(); self.grid[py % H][px % W] = chr(v % 256)
        elif c == '&': s.append(ask_int())
        elif c == '~': s.append(ask_char())
        elif c == '@': self.halted = True; return

        self.x, self.y = (self.x + self.dx) % W, (self.y + self.dy) % H


class App:
    def __init__(self):
        self.interp = Interpreter()
        self.history = []
        self.running = False
        self.delay = 200

        self.root = tk.Tk()
        self.root.title("Befunge")

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=11)

        left = tk.Frame(self.root)
        left.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        right = tk.Frame(self.root)
        right.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        # LEFT: editor
        top = tk.Frame(left)
        top.pack(fill="x")
        tk.Label(top, text="Editor", font=("Sans", 11, "bold")).pack(side="left")
        tk.Button(top, text="Load...", command=self.load_file).pack(side="right")
        self.editor = tk.Text(left, width=W, height=H, font=mono, wrap="none", undo=True)
        self.editor.pack()

        # RIGHT: display + stack + output + controls
        tk.Label(right, text="Execution", font=("Sans", 11, "bold")).pack(anchor="w")
        self.display = tk.Text(right, width=W, height=H, font=mono, wrap="none", state="disabled")
        self.display.pack()
        self.display.tag_configure("ip", background="yellow")
        self.display.tag_configure("string", background="#ffe0b0")

        self.status = tk.Label(right, text="", font=("Sans", 10), anchor="w")
        self.status.pack(fill="x", pady=(2, 4))

        tk.Label(right, text="Stack (bottom -> top)", font=("Sans", 10, "bold"), anchor="w").pack(fill="x")
        self.stack_view = tk.Text(right, width=W, height=3, font=mono, wrap="word", state="disabled")
        self.stack_view.pack()

        tk.Label(right, text="Output", font=("Sans", 10, "bold"), anchor="w").pack(fill="x", pady=(4, 0))
        self.output_view = tk.Text(right, width=W, height=4, font=mono, wrap="word", state="disabled")
        self.output_view.pack()

        ctrl = tk.Frame(right)
        ctrl.pack(pady=6)
        tk.Button(ctrl, text="Reset", command=self.reset).pack(side="left", padx=2)
        tk.Button(ctrl, text="Step Back", command=self.step_back).pack(side="left", padx=2)
        tk.Button(ctrl, text="Step Fwd", command=self.step).pack(side="left", padx=2)
        tk.Button(ctrl, text="Go", command=self.go).pack(side="left", padx=2)
        tk.Button(ctrl, text="Stop", command=self.stop).pack(side="left", padx=2)
        tk.Button(ctrl, text="Slower", command=self.slower).pack(side="left", padx=(12, 2))
        tk.Button(ctrl, text="Faster", command=self.faster).pack(side="left", padx=2)
        self.speed_label = tk.Label(ctrl, text="", font=("Sans", 10), width=10)
        self.speed_label.pack(side="left", padx=4)

        self._update_speed_label()
        self.refresh()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Befunge", "*.bf"), ("All files", "*.*")])
        if not path: return
        with open(path) as f:
            src = f.read()
        self.editor.delete("1.0", "end")
        self.editor.insert("1.0", src)
        self.reset()

    def reset(self):
        self.running = False
        src = self.editor.get("1.0", "end-1c")
        self.interp.load(src)
        self.history.clear()
        self.refresh()

    def _ask_int(self):
        v = simpledialog.askstring("Input", "Enter integer:", parent=self.root)
        try: return int(v)
        except (TypeError, ValueError): return 0

    def _ask_char(self):
        v = simpledialog.askstring("Input", "Enter character:", parent=self.root)
        return ord(v[0]) if v else 0

    def step(self):
        if self.interp.halted: return
        self.history.append(self.interp.snapshot())
        if len(self.history) > 1000:
            self.history.pop(0)
        self.interp.step(self._ask_int, self._ask_char)
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
        self.delay = min(2000, self.delay * 2)
        self._update_speed_label()

    def faster(self):
        self.delay = max(1, self.delay // 2)
        self._update_speed_label()

    def _update_speed_label(self):
        self.speed_label.config(text=f"{self.delay} ms/step")

    def _tick(self):
        if not self.running or self.interp.halted: return
        self.step()
        self.root.after(self.delay, self._tick)

    def refresh(self):
        text = "\n".join("".join(row) for row in self.interp.grid)
        self.display.configure(state="normal")
        self.display.delete("1.0", "end")
        self.display.insert("1.0", text)
        line = self.interp.y + 1
        col = self.interp.x
        self.display.tag_remove("ip", "1.0", "end")
        self.display.tag_add("ip", f"{line}.{col}", f"{line}.{col + 1}")
        self.display.configure(state="disabled")

        arrow = {(1,0): ">", (-1,0): "<", (0,1): "v", (0,-1): "^"}.get(
            (self.interp.dx, self.interp.dy), "?")
        mode = "STRING" if self.interp.string_mode else "normal"
        halted = " [HALTED]" if self.interp.halted else ""
        self.status.config(
            text=f"IP: ({self.interp.x}, {self.interp.y}) {arrow}   mode: {mode}{halted}")

        self.stack_view.configure(state="normal")
        self.stack_view.delete("1.0", "end")
        self.stack_view.insert("1.0", " ".join(str(v) for v in self.interp.stack))
        self.stack_view.configure(state="disabled")

        self.output_view.configure(state="normal")
        self.output_view.delete("1.0", "end")
        self.output_view.insert("1.0", self.interp.output)
        self.output_view.configure(state="disabled")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
