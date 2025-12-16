from manim import *
import numpy as np

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class RidgeStandardization(Scene):
    def construct(self):
        # Title
        title = Text("Why Standardization Matters", color=BLACK).to_edge(UP)
        self.play(Write(title))
        
        # Setup Axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=7,
            y_length=7,
            axis_config={"color": BLACK},
        ).shift(DOWN * 0.5)
        
        labels = axes.get_axis_labels(x_label=r"\beta_1", y_label=r"\beta_2")
        self.play(Create(axes), Write(labels))
        
        # --- Part 1: Unstandardized Data ---
        # Scenario: x1 has large variance, x2 has small variance.
        # Cost contours are elongated along beta_2 axis (vertical).
        # Wait, if x1 is large, cost is sensitive to beta_1, so narrow in beta_1.
        # Ellipse equation: 10*x^2 + y^2 = C
        
        subtitle = Text("Unstandardized Data", font_size=30, color=RED).next_to(title, DOWN)
        self.play(FadeIn(subtitle))

        # OLS Solution (Center of ellipses)
        ols_point = axes.c2p(2, 2)
        ols_dot = Dot(ols_point, color=BLUE)
        ols_label = MathTex(r"\hat{\beta}_{OLS}", color=BLUE).next_to(ols_dot, UR, buff=0.1)
        
        self.play(FadeIn(ols_dot), Write(ols_label))
        
        # Draw Ellipses (MSE Contours)
        # Elongated vertically implies x1 has larger variance than x2?
        # If x1 is large, small change in beta1 makes big change in y -> big error.
        # So steep gradient in beta1 direction -> narrow contours in beta1.
        # So yes, narrow horizontal, wide vertical.
        
        ellipses = VGroup()
        for r in [0.5, 1.0, 1.5, 2.0]:
            ellipse = Ellipse(width=r*1, height=r*4, color=BLUE_D)
            ellipse.move_to(ols_point)
            ellipses.add(ellipse)
            
        self.play(Create(ellipses))
        
        # Draw Ridge Penalty (Circle centered at origin)
        # Constraint: beta1^2 + beta2^2 <= t
        penalty_circle = Circle(radius=1.5, color=RED)
        penalty_circle.move_to(axes.c2p(0, 0))
        
        penalty_label = MathTex(r"\|\beta\|^2 \leq t", color=RED).next_to(penalty_circle, DL, buff=0.1)
        
        self.play(Create(penalty_circle), Write(penalty_label))
        
        # Show Ridge Solution (Intersection)
        # Visually, the "wide" vertical ellipses will touch the circle near the top/bottom?
        # Or rather, since it's narrow horizontally, the gradient is strong horizontally.
        # It pulls the solution towards the vertical axis?
        # Let's look at the geometry.
        # Ellipse center at (2,2). Narrow width (x), large height (y).
        # It looks like a vertical oval.
        # Expanding from (2,2), the first time it hits the circle (centered at 0,0)...
        # Since it's tall, it reaches "down" faster than "left".
        # Actually, if it's tall, the level sets are far apart vertically.
        # The gradient is largest horizontally.
        # So the descent path from OLS to origin follows the steepest descent.
        # Steepest descent on a valley that is steep horizontally means it moves horizontally first.
        # So it reduces beta1 quickly, keeping beta2 large.
        # So the intersection is likely at small beta1, large beta2.
        # Let's approximate the intersection point visually.
        
        ridge_point_unstd_coords = axes.c2p(0.3, 1.45) # visually estimated intersection
        ridge_dot_unstd = Dot(ridge_point_unstd_coords, color=RED)
        ridge_label_unstd = MathTex(r"\hat{\beta}_{Ridge}", color=RED).next_to(ridge_dot_unstd, LEFT, buff=0.1)
        
        self.play(FadeIn(ridge_dot_unstd), Write(ridge_label_unstd))
        
        explanation_unstd = Text(
            "Large scale of x1 makes contours narrow.\nPenalty dominates x2 (small scale).",
            font_size=24, color=BLACK
        ).to_corner(DR)
        self.play(Write(explanation_unstd))
        self.wait(2)
        
        # --- Part 2: Standardized Data ---
        
        new_subtitle = Text("Standardized Data", font_size=30, color=GREEN).next_to(title, DOWN)
        
        self.play(
            FadeOut(subtitle),
            FadeIn(new_subtitle),
            FadeOut(explanation_unstd),
            FadeOut(ridge_dot_unstd),
            FadeOut(ridge_label_unstd)
        )
        
        # Transform Ellipses to Circles
        # Standardized -> Equal variance -> Circular contours
        circles = VGroup()
        for r in [0.5, 1.0, 1.5, 2.0]:
            circle = Circle(radius=r*2, color=GREEN_D) # Radius scaled to match visual size
            circle.move_to(ols_point)
            circles.add(circle)
            
        self.play(Transform(ellipses, circles))
        
        # New Ridge Solution
        # With circular contours, the solution lies on the line segment connecting OLS to Origin.
        # OLS is at (2,2). Origin (0,0). Line y=x.
        # Intersection with circle radius 1.5 (in plot units).
        # Wait, axes units vs pixel units.
        # Axes x_length=7 for range 8 (-4 to 4). So 1 unit = 7/8.
        # Circle radius 1.5 (pixels? no, Mobject units).
        # Axes.c2p(0,0) is center.
        # Let's just place the dot visually on the line connecting (0,0) and (2,2) on the circle.
        
        # The penalty circle was created with radius 1.5 in manim units.
        # We need to find the point on that circle closest to (2,2).
        # Angle is 45 degrees.
        # Point = (1.5 * cos 45, 1.5 * sin 45) relative to origin?
        # Wait, the penalty circle is in the axes coordinates frame?
        # No, I created it with Circle(radius=1.5).
        # I should probably use axes coordinates to be precise, but visual is fine.
        # Let's just put it at 45 degrees on the red circle.
        
        ridge_point_std_coords = penalty_circle.point_at_angle(PI/4)
        ridge_dot_std = Dot(ridge_point_std_coords, color=GREEN)
        ridge_label_std = MathTex(r"\hat{\beta}_{Ridge}", color=GREEN).next_to(ridge_dot_std, RIGHT, buff=0.1)
        
        self.play(FadeIn(ridge_dot_std), Write(ridge_label_std))
        
        explanation_std = Text(
            "Scales are equal.\nShrinkage is balanced.",
            font_size=24, color=BLACK
        ).to_corner(DR)
        self.play(Write(explanation_std))
        
        self.wait(2)
        
        # Conclusion
        conclusion = Text("Always standardize before Ridge!", font_size=40, color=BLACK).move_to(DOWN * 3)
        self.play(Write(conclusion))
        self.wait(2)

