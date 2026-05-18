from manim import *
import numpy as np

# Define signals
noisy_fn = lambda x: np.sin(x) + 0.5 * np.sin(3 * x) + 0.25 * np.sin(5 * x)
clean_fn = lambda x: np.sin(x) + 0.5 * np.sin(3 * x)


class SignalFilter(Scene):
    def construct(self):
        # 1. Setup Axes
        ax1 = (
            Axes(
                x_range=[0, 10, 1],
                y_range=[-1.5, 1.5, 0.5],
                axis_config={"include_numbers": True},
            )
            .scale(0.5)
            .to_edge(UP)
        )

        self.add(ax1)

        ax2 = (
            Axes(
                x_range=[0, 6, 1],
                y_range=[0, 1.5, 0.25],
                axis_config={"include_numbers": True},
            )
            .scale(0.5)
            .to_edge(DOWN)
        )

        self.add(ax2)

        dots = VGroup(
            Dot(ax2.c2p(1, 1), color=RED_C),
            Dot(ax2.c2p(3, 0.5), color=RED_C),
            Dot(ax2.c2p(5, 0.25), color=RED_C),
        )

        self.add(dots)

        filter_graph = ax2.plot(lambda x: np.where(x <= 4, 1, 0), color=GREEN_C)
        self.add(filter_graph)

        # 2. ValueTracker to sync everything
        # This represents the current x-coordinate of the filter line
        vt = ValueTracker(0)

        # 3. Define the signals as dynamic objects
        # Noisy signal: draws from the tracker value to the end (10)
        noisy_graph = always_redraw(
            lambda: ax1.plot(noisy_fn, x_range=[vt.get_value(), 10], color=BLUE)
        )

        # Filtered signal: draws from the start (0) up to the tracker value
        filtered_graph = always_redraw(
            lambda: ax1.plot(clean_fn, x_range=[0, vt.get_value()], color=YELLOW)
        )

        self.play(Create(noisy_graph), Create(filtered_graph))

        # 4. Define the scanning line
        # It stays attached to the tracker's x-coordinate
        filter_line = always_redraw(
            lambda: Line(
                ax1.c2p(vt.get_value(), -1.2),
                ax1.c2p(vt.get_value(), 1.2),
                color=GREEN,
                stroke_width=5,
            )
        )

        # Add a glow or a small label to the line for effect
        line_label = always_redraw(
            lambda: Text("FILTERING", font_size=14)
            .next_to(filter_line, UP, buff=0.1)
            .set_color(GREEN)
        )

        # 5. Initial state
        self.play(
            Create(noisy_graph),
            ReplacementTransform(filter_graph, filter_line),
            Create(line_label),
        )
        self.wait(1)

        # 6. The Animation
        # Moving the tracker from 0 to 10 drives all redraw functions
        self.play(vt.animate.set_value(10), run_time=3, rate_func=linear)

        # Fade out the line once done
        self.play(FadeOut(filter_line), FadeOut(line_label))
        self.wait(2)
