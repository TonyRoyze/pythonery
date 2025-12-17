from manim import *
import numpy as np

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class ScalingAnimation(Scene):
    def construct(self):
        # self.intro()
        self.decision_tree_example()
        self.knn_example()
        self.linear_regression_example()
        # self.conclusion()

    def intro(self):
        title = Text("When to Scale Your Data?", color=BLACK).scale(1.2)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

    def decision_tree_example(self):
        # Title
        title = Text("Case 1: Boundary", color=BLACK).scale(0.8).to_edge(UP)
        self.play(Write(title))

        # Number line 0 to 1 (Meters)
        ax_m = NumberLine(
            x_range=[0, 1, 0.1],
            length=10,
            color=WHITE,
            include_numbers=True,
            label_direction=DOWN,
        ).shift(UP * 1)
        
        label_m = Text("Meters", color=BLACK).scale(0.5).next_to(ax_m, RIGHT)
        
        # Data points
        # Class A (Red): 0.1, 0.2, 0.3, 0.7, 0.75
        # Class B (Blue): 0.85, 0.9, 0.95
        points_m = VGroup()
        for x in [0.1, 0.2, 0.3, 0.7, 0.75]:
            dot = Dot(ax_m.n2p(x), color=RED)
            points_m.add(dot)
        for x in [0.85, 0.9, 0.95]:
            dot = Dot(ax_m.n2p(x), color=BLUE)
            points_m.add(dot)

        self.play(Create(ax_m), Write(label_m), Create(points_m))

        # Split
        split_val_m = 0.8
        split_line_m = DashedLine(
            start=ax_m.n2p(split_val_m) + UP,
            end=ax_m.n2p(split_val_m) + DOWN,
            color=GREEN
        )
        split_text_m = MathTex("x < 0.8", color=GREEN).scale(0.5).next_to(split_line_m, UP)
        
        self.play(Create(split_line_m), Write(split_text_m))
        self.wait(1)

        # Transition to Centimeters
        # 0 to 100
        ax_cm = NumberLine(
            x_range=[0, 100, 10],
            length=10,
            color=WHITE,
            include_numbers=True,
            label_direction=DOWN,
        ).shift(DOWN * 2)
        
        label_cm = Text("Centimeters", color=BLACK).scale(0.5).next_to(ax_cm, RIGHT)

        # Map points to new axis
        points_cm = VGroup()
        for x in [10, 20, 30, 70, 75]:
            dot = Dot(ax_cm.n2p(x), color=RED)
            points_cm.add(dot)
        for x in [85, 90, 95]:
            dot = Dot(ax_cm.n2p(x), color=BLUE)
            points_cm.add(dot)
            
        self.play(
            TransformFromCopy(ax_m, ax_cm),
            TransformFromCopy(points_m, points_cm),
            Write(label_cm)
        )

        # New Split
        split_val_cm = 80
        split_line_cm = DashedLine(
            start=ax_cm.n2p(split_val_cm) + UP,
            end=ax_cm.n2p(split_val_cm) + DOWN,
            color=GREEN
        )
        split_text_cm = MathTex("x < 80", color=GREEN).scale(0.5).next_to(split_line_cm, UP)

        self.play(
            TransformFromCopy(split_line_m, split_line_cm),
            TransformFromCopy(split_text_m, split_text_cm)
        )
        
        self.play(
            FadeOut(title), FadeOut(ax_m), FadeOut(label_m), FadeOut(points_m), 
            FadeOut(split_line_m), FadeOut(split_text_m), FadeOut(ax_cm), 
            FadeOut(label_cm), FadeOut(points_cm), FadeOut(split_line_cm), 
            FadeOut(split_text_cm)
        )

    def knn_example(self):
        title = Text("Case 2: Distance", color=BLACK, font_size=36).to_edge(UP)
        self.play(Write(title))

        # Axes (Meters)
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": WHITE, "include_numbers": True},
        ).shift(LEFT * 3).scale(0.7)
        x_label = axes.get_x_axis_label("x (m)").scale(0.7).next_to(axes.x_axis, DOWN)
        y_label = axes.get_y_axis_label("y (m)").scale(0.7).next_to(axes.y_axis, LEFT).rotate(90 * DEGREES)

        self.play(Create(axes), Write(x_label), Write(y_label))

        # Points: A(1,1), B(1,2), C(2,2)
        pA = axes.c2p(1, 1)
        pB = axes.c2p(1, 2)
        pC = axes.c2p(2, 2)
        
        dotA = Dot(pA, color=RED).set_z_index(10)
        dotB = Dot(pB, color=RED).set_z_index(10)
        dotC = Dot(pC, color=BLUE).set_z_index(10) # Target or different class
        
        labelA = MathTex("A", color=WHITE).next_to(dotA, DL, buff=0.1).scale(0.7)
        labelB = MathTex("B", color=WHITE).next_to(dotB, UL, buff=0.1).scale(0.7)
        labelC = MathTex("C", color=WHITE).next_to(dotC, UR, buff=0.1).scale(0.7)

        self.play(FadeIn(dotA), FadeIn(dotB), FadeIn(dotC), Write(labelA), Write(labelB), Write(labelC))

        # Distances
        lineAB = Line(pA, pB, color=GRAY)
        lineBC = Line(pB, pC, color=GRAY)
        
        distAB = MathTex("d=1", color=BLACK).scale(0.7).next_to(lineAB, LEFT, buff=0.1)
        distBC = MathTex("d=1", color=BLACK).scale(0.7).next_to(lineBC, UP, buff=0.1)

        self.play(Create(lineAB), Create(lineBC), Write(distAB), Write(distBC))
        
        self.wait(1)

        axes_cm = Axes(
            x_range=[0, 250, 50],
            y_range=[0, 3, 1],
            x_length=6, # Stretched a bit
            y_length=5,
            axis_config={"color": WHITE, "include_numbers": True},
        ).shift(RIGHT * 3).scale(0.7)
        
        x_label_cm = axes_cm.get_x_axis_label("x (cm)").scale(0.7).next_to(axes_cm.x_axis, DOWN)
        y_label_cm = axes_cm.get_y_axis_label("y (m)").scale(0.7).next_to(axes_cm.y_axis, LEFT).rotate(90 * DEGREES)
        
        # New Points
        pA_new = axes_cm.c2p(100, 1)
        pB_new = axes_cm.c2p(100, 2)
        pC_new = axes_cm.c2p(200, 2)
        
        dotA_new = Dot(pA_new, color=RED).set_z_index(10)
        dotB_new = Dot(pB_new, color=RED).set_z_index(10)
        dotC_new = Dot(pC_new, color=BLUE).set_z_index(10)

        labelA_new = MathTex("A", color=WHITE).next_to(dotA_new, DL, buff=0.1).scale(0.7)
        labelB_new = MathTex("B", color=WHITE).next_to(dotB_new, UL, buff=0.1).scale(0.7)
        labelC_new = MathTex("C", color=WHITE).next_to(dotC_new, UR, buff=0.1).scale(0.7)
        
        # Move camera or fade replace? Let's fade replace to the right side
        self.play(
            FadeIn(axes_cm), FadeIn(x_label_cm), FadeIn(y_label_cm),
            FadeIn(labelA_new), FadeIn(labelB_new), FadeIn(labelC_new),
            FadeIn(dotA_new), FadeIn(dotB_new), FadeIn(dotC_new)
        )
        
        # New Distances
        lineAB_new = Line(pA_new, pB_new, color=GRAY)
        lineBC_new = Line(pB_new, pC_new, color=GRAY)
        
        distAB_new = MathTex("d=1", color=BLACK).scale(0.7).next_to(lineAB_new, LEFT, buff=0.1)
        distBC_new = MathTex("d=100", color=BLACK).scale(0.7).next_to(lineBC_new, UP, buff=0.1)
        
        self.play(Create(lineAB_new), Create(lineBC_new), Write(distAB_new), Write(distBC_new))
        
        self.wait(2)
        
        self.play(
            FadeOut(title), FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(dotA), FadeOut(dotB), FadeOut(dotC),
            FadeOut(labelA), FadeOut(labelB), FadeOut(labelC),
            FadeOut(lineAB), FadeOut(lineBC), FadeOut(distAB), FadeOut(distBC),
            FadeOut(axes_cm), FadeOut(x_label_cm), FadeOut(y_label_cm),
            FadeOut(dotA_new), FadeOut(dotB_new), FadeOut(dotC_new),
            FadeOut(labelA_new), FadeOut(labelB_new), FadeOut(labelC_new),
            FadeOut(lineAB_new), FadeOut(lineBC_new), FadeOut(distAB_new), FadeOut(distBC_new),
        )

    def linear_regression_example(self):
        title = Text("Case 3: Interpretability", color=BLACK).scale(0.8).to_edge(UP)
        self.play(Write(title))

        # Equation
        eq1 = MathTex(r"y = \beta_1 x_1 + \beta_2 x_2", color=BLACK).shift(UP * 2)
        self.play(Write(eq1))
        
        
        vals1 = MathTex(r"\text{If } x_1, x_2 \text{ in meters: } \beta_1 \approx \beta_2", color=BLACK).scale(0.8).next_to(eq1, DOWN)
        self.play(Write(vals1))
        
        
        arrow = Arrow(UP, DOWN, color=BLACK).scale(0.8).next_to(vals1, DOWN)
        text_change = MathTex(r"\text{Scale } x_2 \text{ to cm}", color=RED).scale(0.8).next_to(arrow, RIGHT)
        self.play(Create(arrow), Write(text_change))
        
        eq2 = MathTex(r"y = \beta_1 x_1 + \beta_2' (100 x_2)", color=BLACK).next_to(arrow, DOWN)
        self.play(Write(eq2))
        
        conclusion_text = MathTex(r"\Rightarrow \beta_2' = \frac{\beta_2}{100}", color=RED).next_to(eq2, DOWN)
        self.play(Write(conclusion_text))
        
        # note = Text("Coefficient shrinks, making variable look less important.", color=BLACK).scale(0.7).to_edge(DOWN)
        # self.play(Write(note))
        self.wait(2)
        
        # self.play(
        #     FadeOut(title), FadeOut(eq1), FadeOut(vals1), FadeOut(arrow), 
        #     FadeOut(text_change), FadeOut(eq2), FadeOut(conclusion_text)
        # )

    def conclusion(self):
        # Summary table or text
        t1 = Text("1. Distance-based (KNN, SVM, K-Means) -> SCALE", color=GREEN, font_size=30).shift(UP*1)
        t2 = Text("2. Tree-based (Random Forest, XGBoost) -> OPTIONAL", color=BLUE, font_size=30)
        t3 = Text("3. Linear Models (Regression, Logistic) -> SCALE for Interpretability", color=ORANGE, font_size=30).shift(DOWN*1)
        
        self.play(Write(t1))
        self.play(Write(t2))
        self.play(Write(t3))
        self.wait(3)
