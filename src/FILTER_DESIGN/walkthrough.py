from manim import *
from manim_slides import Slide
import numpy as np

# ─── Palette ──────────────────────────────────────────────────────────────────

_BG = ManimColor("#04050e")
_ACCENT = ManimColor("#00d4ff")  # cyan  — headings, highlights
_GOLD = ManimColor("#ffd060")  # gold  — noisy signal
_GREEN = ManimColor("#00ff88")  # green — filtered signal / passband
_RED = ManimColor("#ff6b6b")  # coral — low-freq label

# ─── Filter back-end (identical logic to the utility function) ─────────────────


def _compute_ls_filter(
    lowcut: float = 1000, highcut: float = 4000, fs: float = 20050, order: int = 4
):
    """Return (h, a_coeffs) for a Type-I least-squares FIR bandpass filter."""
    num_loops = (order // 2) + 1
    omega_low = 2 * np.pi * lowcut / fs
    omega_high = 2 * np.pi * highcut / fs
    omega = np.linspace(0, np.pi, 1000)

    D = ((omega >= omega_low) & (omega <= omega_high)).astype(float)
    A = np.column_stack([np.cos(n * omega) for n in range(num_loops)])

    a = np.linalg.solve(A.T @ A, A.T @ D)

    h = np.zeros(order + 1)
    c = order // 2
    h[c] = a[0]
    h[c - 1] = h[c + 1] = a[1] / 2.0
    h[c - 2] = h[c + 2] = a[2] / 2.0
    return h, a


def _apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Direct-form FIR convolution (mirrors the utility function)."""
    y = np.zeros_like(x, dtype=float)
    for k, hk in enumerate(h):
        if k == 0:
            y += hk * x
        else:
            y[k:] += hk * x[:-k]
    return y


# ─── Main Animation Class ─────────────────────────────────────────────────────


class LeastSq(Slide):
    """
    Full Manim walkthrough of the Least-Squares Bandpass FIR Filter.
    Class is named 'LeastSq' for a shorter render command.
    """

    def setup(self):
        self.camera.background_color = _BG

    def construct(self):
        self.scene_title()
        self.scene_problem()
        self.scene_normalization()
        self.scene_desired_response()
        self.scene_design_matrix()
        self.scene_normal_equations()
        self.scene_reconstruct_taps()
        self.scene_convolution()
        self.scene_demo()

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def wipe(self, rt: float = 0.55):
        """Fade-out every mobject currently on screen."""
        mobs = list(self.mobjects)
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=rt)

    def step_banner(self, step_num: str, title: str) -> VGroup:
        """Top-edge step label with underbar."""
        lbl = Text(step_num, font_size=18, color=_ACCENT, weight=BOLD)
        name = Text(title, font_size=24, color=WHITE, weight=BOLD)
        row = VGroup(lbl, name).arrange(RIGHT, buff=0.24)
        bar = Line(
            row.get_left(), row.get_right(), color=_ACCENT, stroke_width=1.2
        ).next_to(row, DOWN, buff=0.07)
        return VGroup(row, bar).to_edge(UP, buff=0.22)

    def vmob_from_xy(self, ax: Axes, xs, ys, **kwargs) -> VMobject:
        """Build a VMobject polyline from x/y arrays via axis coordinates."""
        vm = VMobject(**kwargs)
        vm.set_points_as_corners([ax.c2p(float(x), float(y)) for x, y in zip(xs, ys)])
        return vm

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 0 — Title
    # ══════════════════════════════════════════════════════════════════════════

    def scene_title(self):
        t1 = Text("Least Squares", font_size=74, weight=BOLD)
        t2 = Text("Bandpass FIR Filter", font_size=52, color=_ACCENT)
        t3 = Text("A Step-by-Step Mathematical Walkthrough", font_size=26, color=GRAY)
        VGroup(t1, t2, t3).arrange(DOWN, buff=0.50)

        self.play(Write(t1), run_time=1.1)
        self.play(FadeIn(t2, shift=UP * 0.3))
        self.play(FadeIn(t3))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 1 — The Problem
    # ══════════════════════════════════════════════════════════════════════════

    def scene_problem(self):
        bnr = self.step_banner("Step 1", "The Problem")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        # Synthesise a noisy multi-component signal
        N = 280
        t = np.linspace(0, 0.014, N)
        rng = np.random.default_rng(seed=0)
        sig = (
            np.sin(2 * np.pi * 500 * t)
            + 0.85 * np.sin(2 * np.pi * 2500 * t)
            + 0.65 * np.sin(2 * np.pi * 8000 * t)
            + 0.35 * rng.standard_normal(N)
        )

        ax = Axes(
            x_range=[0, 0.014, 0.004],
            y_range=[-3.2, 3.2, 1],
            x_length=10.5,
            y_length=3.7,
            axis_config={"color": DARK_GRAY, "stroke_width": 1.2},
            tips=False,
        ).shift(DOWN * 0.55)
        xl = Text("Time (s)", font_size=17, color=DARK_GRAY).next_to(
            ax, DOWN, buff=0.20
        )
        yl = Text("Amplitude", font_size=17, color=DARK_GRAY).next_to(
            ax, LEFT, buff=0.18
        )

        curve = self.vmob_from_xy(ax, t, sig, color=_GOLD, stroke_width=1.5)

        tags = (
            VGroup(
                Text("500 Hz  (low)", font_size=17, color=_RED),
                Text("2500 Hz  (target ✓)", font_size=17, color=_GREEN),
                Text("8000 Hz  (high)", font_size=17, color="#ffaa44"),
                Text("+ Gaussian noise", font_size=17, color=GRAY),
            )
            .arrange(RIGHT, buff=0.60)
            .next_to(ax, DOWN, buff=0.52)
        )

        goal = Text(
            "Goal: isolate the 2500 Hz passband using a least-squares FIR filter",
            font_size=20,
            color=_ACCENT,
        ).next_to(tags, DOWN, buff=0.22)

        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(curve), run_time=2.0)
        self.play(FadeIn(tags, shift=UP * 0.10))
        self.play(Write(goal))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 2 — Frequency Normalization
    # ══════════════════════════════════════════════════════════════════════════

    def scene_normalization(self):
        bnr = self.step_banner("Step 2", "Frequency Normalization")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        fs, fl, fh = 20050, 1000, 4000
        wl = 2 * np.pi * fl / fs
        wh = 2 * np.pi * fh / fs

        # Symbolic formulas
        eq1 = MathTex(r"\omega_{low}  = \frac{2\pi\, f_{low}}{f_s}", font_size=40)
        eq2 = MathTex(r"\omega_{high} = \frac{2\pi\, f_{high}}{f_s}", font_size=40)
        VGroup(eq1, eq2).arrange(RIGHT, buff=2.8).shift(UP * 2.05)

        n1 = MathTex(
            rf"= \dfrac{{2\pi \times {fl}}}{{{fs}}} \approx {wl:.4f}\;\text{{rad}}",
            font_size=29,
            color=_ACCENT,
        ).next_to(eq1, DOWN, buff=0.30)
        n2 = MathTex(
            rf"= \dfrac{{2\pi \times {fh}}}{{{fs}}} \approx {wh:.4f}\;\text{{rad}}",
            font_size=29,
            color=_ACCENT,
        ).next_to(eq2, DOWN, buff=0.30)

        self.play(Write(eq1), Write(eq2))
        self.play(FadeIn(n1), FadeIn(n2))

        # Number-line visualisation  [0, π]
        nl = NumberLine(
            x_range=[0, np.pi, np.pi / 4],
            length=9.0,
            include_numbers=False,
            color=DARK_GRAY,
        ).shift(DOWN * 1.80)

        nl_lbls = VGroup(
            MathTex("0", font_size=21, color=GRAY).next_to(nl.n2p(0), DOWN, buff=0.09),
            MathTex(r"\pi/4", font_size=21, color=GRAY).next_to(
                nl.n2p(np.pi / 4), DOWN, buff=0.09
            ),
            MathTex(r"\pi/2", font_size=21, color=GRAY).next_to(
                nl.n2p(np.pi / 2), DOWN, buff=0.09
            ),
            MathTex(r"3\pi/4", font_size=21, color=GRAY).next_to(
                nl.n2p(3 * np.pi / 4), DOWN, buff=0.09
            ),
            MathTex(r"\pi", font_size=21, color=GRAY).next_to(
                nl.n2p(np.pi), DOWN, buff=0.09
            ),
        )
        passband_bar = Line(nl.n2p(wl), nl.n2p(wh), color=_ACCENT, stroke_width=7)
        dl = Dot(nl.n2p(wl), color=_GREEN, radius=0.10)
        dh = Dot(nl.n2p(wh), color=_RED, radius=0.10)
        ll = MathTex(r"\omega_{low}", font_size=19, color=_GREEN).next_to(
            dl, UP, buff=0.10
        )
        lh = MathTex(r"\omega_{high}", font_size=19, color=_RED).next_to(
            dh, UP, buff=0.10
        )

        self.play(Create(nl), FadeIn(nl_lbls))
        self.play(GrowFromCenter(dl), FadeIn(ll), GrowFromCenter(dh), FadeIn(lh))
        self.play(Create(passband_bar))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 3 — Desired Brick-Wall Response
    # ══════════════════════════════════════════════════════════════════════════

    def scene_desired_response(self):
        bnr = self.step_banner("Step 3", "Ideal Brick-Wall Response")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        wl = 2 * np.pi * 1000 / 20050
        wh = 2 * np.pi * 4000 / 20050

        # Definition
        defn = MathTex(
            r"D(\omega) = \begin{cases}"
            r" 1 & \omega_{low} \leq \omega \leq \omega_{high} \\"
            r" 0 & \text{otherwise}"
            r"\end{cases}",
            font_size=36,
        ).shift(UP * 2.75)
        self.play(Write(defn), run_time=1.5)

        # Plot
        ax = Axes(
            x_range=[0, np.pi, np.pi / 4],
            y_range=[-0.12, 1.45, 0.5],
            x_length=9.2,
            y_length=3.25,
            axis_config={"color": DARK_GRAY, "stroke_width": 1.2},
            tips=False,
        ).shift(DOWN * 1.00)
        xl = MathTex(r"\omega\;\text{(rad)}", font_size=20, color=DARK_GRAY).next_to(
            ax, RIGHT, buff=0.10
        )
        yl = (
            MathTex(r"D(\omega)", font_size=20, color=DARK_GRAY)
            .next_to(ax, UP, buff=0.10)
            .shift(LEFT * 3.6)
        )

        # Brick-wall outline
        bw_pts = [
            ax.c2p(0, 0),
            ax.c2p(wl, 0),
            ax.c2p(wl, 1),
            ax.c2p(wh, 1),
            ax.c2p(wh, 0),
            ax.c2p(np.pi, 0),
        ]
        brick_wall = VMobject(color=_ACCENT, stroke_width=3.0)
        brick_wall.set_points_as_corners(bw_pts)

        fill = Polygon(
            ax.c2p(wl, 0),
            ax.c2p(wl, 1),
            ax.c2p(wh, 1),
            ax.c2p(wh, 0),
            color=_ACCENT,
            fill_opacity=0.22,
            stroke_width=0,
        )
        pb_lbl = MathTex(r"\text{Passband}", font_size=22, color=_ACCENT).move_to(
            ax.c2p((wl + wh) / 2, 1.20)
        )
        wl_tick = MathTex(r"\omega_{low}", font_size=19, color=_GREEN).next_to(
            ax.c2p(wl, 0), DOWN, buff=0.10
        )
        wh_tick = MathTex(r"\omega_{high}", font_size=19, color=_RED).next_to(
            ax.c2p(wh, 0), DOWN, buff=0.10
        )

        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(fill), Create(brick_wall), run_time=1.2)
        self.play(FadeIn(pb_lbl), FadeIn(wl_tick), FadeIn(wh_tick))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 4 — Design Matrix A
    # ══════════════════════════════════════════════════════════════════════════

    def scene_design_matrix(self):
        bnr = self.step_banner("Step 4", "Cosine Basis Design Matrix  A")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        # Type-I FIR real-frequency-response representation
        resp_hdr = Text(
            "For a Type-I FIR filter the real frequency response is:",
            font_size=21,
            color=GRAY,
        ).shift(UP * 2.90)
        self.play(FadeIn(resp_hdr))

        resp = MathTex(
            r"H(\omega) = a_0 + 2\sum_{n=1}^{2} a_n \cos(n\,\omega)",
            font_size=38,
        ).next_to(resp_hdr, DOWN, buff=0.35)
        self.play(Write(resp), run_time=1.2)

        # Grid explanation
        sub = Text(
            "Evaluate over 1 000 uniformly-spaced points  " r"ω₀ … ω₉₉₉ ∈ [0, π]:",
            font_size=20,
            color=GRAY,
        ).next_to(resp, DOWN, buff=0.40)
        self.play(FadeIn(sub))

        A_def = MathTex(
            r"\mathbf{A}_{i,\,n} = \cos\!\bigl(n\,\omega_i\bigr),"
            r"\quad n \in \{0,\,1,\,2\},\quad i = 0,\ldots,999",
            font_size=30,
        ).next_to(sub, DOWN, buff=0.32)
        self.play(Write(A_def))

        # Matrix display
        mat = MathTex(
            r"\mathbf{A}_{\,1000\times3} = "
            r"\begin{bmatrix}"
            r"1 & \cos(\omega_0)   & \cos(2\omega_0)   \\"
            r"1 & \cos(\omega_1)   & \cos(2\omega_1)   \\"
            r"\vdots & \vdots      & \vdots            \\"
            r"1 & \cos(\omega_{999}) & \cos(2\omega_{999})"
            r"\end{bmatrix}",
            font_size=27,
        ).next_to(A_def, DOWN, buff=0.40)
        self.play(Write(mat), run_time=1.4)

        col_lbls = (
            VGroup(
                Text("n = 0  (constant)", font_size=17, color=_RED),
                Text("n = 1  (1st harm.)", font_size=17, color=_GREEN),
                Text("n = 2  (2nd harm.)", font_size=17, color=_ACCENT),
            )
            .arrange(RIGHT, buff=1.10)
            .next_to(mat, DOWN, buff=0.28)
        )
        self.play(FadeIn(col_lbls, shift=UP * 0.10))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 5 — Normal Equations
    # ══════════════════════════════════════════════════════════════════════════

    def scene_normal_equations(self):
        bnr = self.step_banner("Step 5", "Least Squares — Normal Equations")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        # Objective
        obj_hdr = Text(
            "Minimise the squared error between H(ω) and the ideal D(ω):",
            font_size=22,
            color=GRAY,
        ).shift(UP * 3.0)
        self.play(FadeIn(obj_hdr))

        obj = MathTex(
            r"\min_{\mathbf{a}}\;\left\|\mathbf{A}\mathbf{a} - \mathbf{D}\right\|^2",
            font_size=50,
        ).next_to(obj_hdr, DOWN, buff=0.34)
        self.play(Write(obj), run_time=0.95)

        # Gradient
        grad_hdr = Text(
            "Differentiate w.r.t. a and set the gradient to zero:",
            font_size=21,
            color=GRAY,
        ).next_to(obj, DOWN, buff=0.44)
        self.play(FadeIn(grad_hdr))

        grad = MathTex(
            r"\nabla_{\mathbf{a}}\left\|\mathbf{A}\mathbf{a}-\mathbf{D}\right\|^2"
            r"= 2\,\mathbf{A}^{T}\!\left(\mathbf{A}\mathbf{a}-\mathbf{D}\right)"
            r"= \mathbf{0}",
            font_size=32,
        ).next_to(grad_hdr, DOWN, buff=0.28)
        self.play(Write(grad), run_time=1.3)

        # Arrow → Normal Equations
        arr = Arrow(
            start=grad.get_bottom() + DOWN * 0.05,
            end=grad.get_bottom() + DOWN * 0.80,
            buff=0,
            color=YELLOW,
            stroke_width=2.8,
            tip_length=0.22,
        )
        self.play(GrowArrow(arr))

        ne = MathTex(
            r"\bigl(\mathbf{A}^T\mathbf{A}\bigr)\,\mathbf{a}"
            r"\;=\;\mathbf{A}^T\mathbf{D}",
            font_size=52,
            color=_ACCENT,
        ).next_to(arr, DOWN, buff=0.14)
        box = SurroundingRectangle(ne, color=_ACCENT, buff=0.18, corner_radius=0.08)
        self.play(Write(ne), run_time=0.95)
        self.play(Create(box))

        # Closed-form solution
        sol = MathTex(
            r"\therefore\quad \mathbf{a} ="
            r"\left(\mathbf{A}^T\mathbf{A}\right)^{-1}\!\mathbf{A}^T\mathbf{D}",
            font_size=36,
            color=YELLOW,
        ).next_to(ne, DOWN, buff=0.52)
        self.play(Write(sol), run_time=0.9)
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 6 — Reconstruct FIR Taps
    # ══════════════════════════════════════════════════════════════════════════

    def scene_reconstruct_taps(self):
        bnr = self.step_banner("Step 6", "Reconstructing the FIR Tap Coefficients")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        sub = Text(
            "Type I FIR  ·  odd length (5 taps)  ·  linear phase  ·  even symmetry",
            font_size=21,
            color=GRAY,
        ).shift(UP * 2.92)
        self.play(FadeIn(sub))

        # ── 5 tap boxes ──────────────────────────────────────────────────────
        box_colors = [_GOLD, _GREEN, _ACCENT, _GREEN, _GOLD]
        tap_boxes = (
            VGroup(
                *[
                    VGroup(
                        Square(
                            side_length=0.90,
                            color=c,
                            fill_color=c,
                            fill_opacity=0.15,
                            stroke_width=2.0,
                        ),
                        MathTex(f"h[{i}]", font_size=25),
                    ).arrange(DOWN, buff=0.08)
                    for i, c in enumerate(box_colors)
                ]
            )
            .arrange(RIGHT, buff=0.32)
            .shift(UP * 1.40)
        )
        self.play(FadeIn(tap_boxes))

        # ── Symmetry arrows ───────────────────────────────────────────────────
        da = DoubleArrow(
            start=tap_boxes[0].get_top() + UP * 0.28,
            end=tap_boxes[4].get_top() + UP * 0.28,
            buff=0,
            color=_GOLD,
            tip_length=0.17,
            stroke_width=2.2,
        )
        db = DoubleArrow(
            start=tap_boxes[1].get_top() + UP * 0.14,
            end=tap_boxes[3].get_top() + UP * 0.14,
            buff=0,
            color=_GREEN,
            tip_length=0.15,
            stroke_width=2.0,
        )
        sym_lbl = Text(
            "h[n] = h[4 − n]   (symmetric)", font_size=19, color=_GOLD
        ).next_to(da, UP, buff=0.07)
        self.play(Create(da), Create(db), FadeIn(sym_lbl))

        # ── Reconstruction formulas ───────────────────────────────────────────
        f0 = MathTex(r"h[2] \;=\; a_0", font_size=34, color=_ACCENT)
        f1 = MathTex(r"h[1] = h[3] \;=\; a_1/2", font_size=34, color=_GREEN)
        f2 = MathTex(r"h[0] = h[4] \;=\; a_2/2", font_size=34, color=_GOLD)
        VGroup(f0, f1, f2).arrange(DOWN, buff=0.34).shift(DOWN * 0.65)
        for f in (f0, f1, f2):
            self.play(Write(f), run_time=0.65)

        # ── Numerical result ──────────────────────────────────────────────────
        h, a = _compute_ls_filter()
        val = Text(
            f"Solved:  h = [{h[0]:.4f},  {h[1]:.4f},  {h[2]:.4f},"
            f"  {h[3]:.4f},  {h[4]:.4f}]",
            font_size=19,
            color=_ACCENT,
        ).next_to(f2, DOWN, buff=0.42)
        self.play(FadeIn(val, shift=UP * 0.12))
        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 7 — Convolution (Difference Equation)
    # ══════════════════════════════════════════════════════════════════════════

    def scene_convolution(self):
        bnr = self.step_banner(
            "Step 7", "Applying the Filter — Direct-Form Convolution"
        )
        self.play(FadeIn(bnr, shift=DOWN * 0.12))

        hdr = Text("FIR difference equation:", font_size=23, color=GRAY).shift(
            UP * 2.90
        )
        self.play(FadeIn(hdr))

        eq = MathTex(
            r"y[n] \;=\; \sum_{k=0}^{4} h[k]\cdot x[n-k]",
            font_size=50,
        ).next_to(hdr, DOWN, buff=0.36)
        self.play(Write(eq), run_time=1.1)

        exp = MathTex(
            r"y[n] = h[0]\,x[n]"
            r"+ h[1]\,x[n{-}1]"
            r"+ h[2]\,x[n{-}2]"
            r"+ h[3]\,x[n{-}3]"
            r"+ h[4]\,x[n{-}4]",
            font_size=27,
            color=_ACCENT,
        ).next_to(eq, DOWN, buff=0.46)
        self.play(Write(exp), run_time=1.3)

        win_hdr = Text(
            "Sliding 5-tap window across the discrete input signal:",
            font_size=20,
            color=GRAY,
        ).next_to(exp, DOWN, buff=0.44)
        self.play(FadeIn(win_hdr))

        # Sample boxes
        sample_boxes = (
            VGroup(
                *[
                    VGroup(
                        Square(
                            side_length=0.57,
                            color=DARK_GRAY,
                            fill_opacity=0.18,
                            stroke_width=1.4,
                        ),
                        MathTex(f"x[{i}]" if i < 9 else r"\cdots", font_size=15),
                    ).arrange(DOWN, buff=0.04)
                    for i in range(10)
                ]
            )
            .arrange(RIGHT, buff=0.07)
            .next_to(win_hdr, DOWN, buff=0.38)
        )
        self.play(FadeIn(sample_boxes))

        win = SurroundingRectangle(VGroup(*sample_boxes[:5]), color=_ACCENT, buff=0.05)
        win_lbl = MathTex(r"h[0\ldots4]", font_size=18, color=_ACCENT).next_to(
            win, UP, buff=0.08
        )
        self.play(Create(win), FadeIn(win_lbl))

        for s in range(1, 5):
            nw = SurroundingRectangle(
                VGroup(*sample_boxes[s : s + 5]), color=_ACCENT, buff=0.05
            )
            nl = MathTex(r"h[0\ldots4]", font_size=18, color=_ACCENT).next_to(
                nw, UP, buff=0.08
            )
            self.play(Transform(win, nw), Transform(win_lbl, nl), run_time=0.44)

        self.next_slide()
        self.wipe()

    # ══════════════════════════════════════════════════════════════════════════
    # Scene 8 — Live Demo with Scanning Sweep
    # ══════════════════════════════════════════════════════════════════════════

    def scene_demo(self):
        bnr = self.step_banner("Step 8", "Filter in Action — Scanning Demo")
        self.play(FadeIn(bnr, shift=DOWN * 0.12))
        self.next_slide()
        self.play(FadeOut(bnr))

        # ── Build signals ─────────────────────────────────────────────────────
        N = 340
        t = np.linspace(0, 0.017, N)
        rng = np.random.default_rng(seed=7)
        raw = (
            np.sin(2 * np.pi * 500 * t)
            + 0.90 * np.sin(2 * np.pi * 2500 * t)
            + 0.70 * np.sin(2 * np.pi * 8000 * t)
            + 0.40 * rng.standard_normal(N)
        )
        h, _ = _compute_ls_filter()
        filt = _apply_fir(raw, h)

        def _norm(arr: np.ndarray, lim: float = 1.90) -> np.ndarray:
            return arr / (np.max(np.abs(arr)) + 1e-9) * lim

        rn = _norm(raw)
        fn = _norm(filt)

        # ── Axes ──────────────────────────────────────────────────────────────
        ax = Axes(
            x_range=[0, N - 1, (N - 1) // 4],
            y_range=[-2.30, 2.30, 1],
            x_length=12.2,
            y_length=5.10,
            axis_config={"color": DARK_GRAY, "stroke_width": 1.2},
            tips=False,
        ).shift(DOWN * 0.08)
        xl = Text("Samples", font_size=17, color=DARK_GRAY).next_to(ax, DOWN, buff=0.14)
        yl = Text("Amplitude", font_size=17, color=DARK_GRAY).next_to(
            ax, LEFT, buff=0.14
        )
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.70)

        # ── Draw full noisy curve ─────────────────────────────────────────────
        init_curve = self.vmob_from_xy(
            ax,
            range(N),
            rn,
            color=_GOLD,
            stroke_width=1.6,
            stroke_opacity=0.88,
        )
        self.play(Create(init_curve), run_time=1.6)

        # ── Legend ────────────────────────────────────────────────────────────
        leg_noisy = VGroup(
            Line(ORIGIN, RIGHT * 0.55, color=_GOLD, stroke_width=3),
            Text("Noisy Input", font_size=17, color=_GOLD),
        ).arrange(RIGHT, buff=0.18)
        leg_filt = VGroup(
            Line(ORIGIN, RIGHT * 0.55, color=_GREEN, stroke_width=3),
            Text("Filtered Output", font_size=17, color=_GREEN),
        ).arrange(RIGHT, buff=0.18)
        legend = (
            VGroup(leg_noisy, leg_filt)
            .arrange(DOWN, buff=0.22, aligned_edge=LEFT)
            .to_corner(UR, buff=0.32)
            .shift(DOWN * 0.62)
        )
        self.play(FadeIn(leg_noisy))

        # ── Scan prompt ───────────────────────────────────────────────────────
        prompt = Text(
            "Applying Least-Squares Bandpass Filter  →  scanning left to right …",
            font_size=19,
            color=_ACCENT,
        ).to_edge(UP, buff=0.22)
        self.play(FadeIn(prompt))
        self.next_slide()

        # ── Scan tracker ──────────────────────────────────────────────────────
        tracker = ValueTracker(0.0)

        # ── Live composite curve ──────────────────────────────────────────────
        # Left  of scan line → FILTERED (green)
        # Right of scan line → NOISY    (gold)
        def _make_composite() -> VGroup:
            p = tracker.get_value()
            split = max(2, min(N - 2, int(p * N)))
            g = VGroup()

            # Filtered segment (left)
            lv = VMobject(color=_GREEN, stroke_width=2.7)
            lv.set_points_as_corners([ax.c2p(i, fn[i]) for i in range(split)])
            g.add(lv)

            # Noisy segment (right)
            rv = VMobject(color=_GOLD, stroke_width=1.6, stroke_opacity=0.88)
            rv.set_points_as_corners([ax.c2p(i, rn[i]) for i in range(split, N)])
            g.add(rv)

            return g

        composite = always_redraw(_make_composite)

        # ── Scan line ─────────────────────────────────────────────────────────
        def _scan_line() -> Line:
            x = tracker.get_value() * (N - 1)
            return Line(
                ax.c2p(x, -2.30),
                ax.c2p(x, 2.30),
                color=WHITE,
                stroke_width=2.5,
                stroke_opacity=0.90,
            )

        def _glow_dot() -> Dot:
            x = tracker.get_value() * (N - 1)
            return Dot(ax.c2p(x, 0.0), color=WHITE, radius=0.068)

        scan_line = always_redraw(_scan_line)
        glow_dot = always_redraw(_glow_dot)

        # Swap static curve → live composite + scan overlay
        self.remove(init_curve)
        self.add(composite, scan_line, glow_dot)

        # ── SWEEP ─────────────────────────────────────────────────────────────
        self.play(
            tracker.animate.set_value(1.0),
            run_time=8.0,
            rate_func=linear,
        )

        # ── Finish ────────────────────────────────────────────────────────────
        self.play(FadeOut(scan_line), FadeOut(glow_dot), FadeOut(prompt))
        self.play(FadeIn(leg_filt))

        done = Text(
            "✓  1000–4000 Hz passband preserved  ·  500 Hz, 8000 Hz and noise rejected",
            font_size=19,
            color=_GREEN,
        ).to_edge(DOWN, buff=0.28)
        self.play(Write(done))
        self.next_slide()
        self.wipe()
        self.play(Write(Text("Thank You", font_size=45)))
