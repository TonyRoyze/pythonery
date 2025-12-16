from manim import *
import numpy as np

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class RidgeMulticollinearityScene(Scene):
    def construct(self):

        # 2. OLS Formula
        ols_formula = MathTex(
            r"\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y",
            color=BLACK
        ).scale(1.0)
        self.play(Write(ols_formula))
        self.wait(1)
        
        # Move formula up
        self.play(ols_formula.animate.shift(UP * 2.5).scale(0.7))

        # 3. Define a multicollinear matrix X
        # Let's use a simple 3x2 matrix where col2 = 2 * col1
        X_val = r"\begin{pmatrix} a & ka \\ b & kb \end{pmatrix}"
        
        # Display X
        matrix_x = MathTex(X_val, color=BLACK).scale(0.8)
        matrix_x_label = MathTex("X =", color=BLACK).scale(0.8).next_to(matrix_x, LEFT)
        x_group = VGroup(matrix_x_label, matrix_x)
        
        self.play(FadeIn(x_group))
        
        # Explain multicollinearity
        # col1_rect = SurroundingRectangle(matrix_x.get_columns()[0], color=BLUE)
        # col2_rect = SurroundingRectangle(matrix_x.get_columns()[1], color=RED)
        
        # self.play(Create(col1_rect), Create(col2_rect))
        
        explanation = Text("Column 2 = k * Column 1 (Perfect Multicollinearity)", color=RED).scale(0.5).next_to(ols_formula, DOWN)
        self.play(Write(explanation))
        self.wait(1)
        
        # self.play(FadeOut(col1_rect), FadeOut(col2_rect), FadeOut(explanation))

        # 4. Calculate X^T X
        # XtX_val = np.array([["a", "ka"], ["b", "kb"]]) # [[14, 28], [28, 56]]
        XtX_val = r"\begin{pmatrix} a^2 + b^2 + c^2 & k(a^2 + b^2 + c^2) \\ k(a^2 + b^2 + c^2) & k^2(a^2 + b^2 + c^2) \end{pmatrix}"

        self.play(x_group.animate.move_to(LEFT * 5))

        
        
        matrix_xtx = MathTex(XtX_val, color=BLACK).scale(0.8)
        matrix_xtx_label = MathTex("X^T X =", color=BLACK).scale(0.8).next_to(matrix_xtx, LEFT)
        xtx_group = VGroup(matrix_xtx_label, matrix_xtx).next_to(x_group, RIGHT, buff=1.5)
        
        arrow = Arrow(x_group.get_right(), xtx_group.get_left(), color=BLACK)
        
        self.play(GrowArrow(arrow), FadeIn(xtx_group))
        self.wait(1)

        # 5. Show Determinant is 0
        # det_val = int(np.linalg.det(XtX_val)) # Should be 0
        
        det_text = MathTex(r"\det(X^T X) = k^2(a^2 + b^2 + c^2) - k^2(a^2 + b^2 + c^2) = 0", color=RED).scale(0.7).next_to(arrow, DOWN, buff=1.5).set_x(0)
        self.play(Write(det_text))
        self.wait(1)
        
        singular_text = Text("Not Invertible!", color=RED).scale(0.5).next_to(det_text, DOWN)
        self.play(Write(singular_text))
        self.wait(2)
        
        # Clear screen for Ridge
        self.play(
            FadeOut(x_group), FadeOut(xtx_group), FadeOut(arrow), 
            FadeOut(det_text), FadeOut(singular_text), FadeOut(ols_formula), FadeOut(explanation),
        )

        self.wait(3)

        # 6. Ridge Solution
        ridge_formula = MathTex(
            r"\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y",
            color=BLACK
        )
        self.play(Write(ridge_formula))


        self.play(ridge_formula.animate.shift(UP * 2.5).scale(0.7))
        self.wait(1)

        # Bring back X^T X
        matrix_xtx = MathTex(XtX_val, color=BLACK).scale(0.8)
        matrix_xtx_label = MathTex("X^T X =", color=BLACK).scale(0.8).next_to(matrix_xtx, LEFT)
        xtx_group = VGroup(matrix_xtx_label, matrix_xtx)
        xtx_group.set_x(0)
        self.play(FadeIn(xtx_group))
        
        # Show Lambda I
        # lambda_val = 1 # Example lambda
        # I_val = np.eye(2, dtype=int)
        # lambda_I_val = lambda_val * I_val

        # Show Resulting Matrix
        ridge_matrix_val = r"\begin{pmatrix} d + \lambda & k(d) \\ k(d) & k^2(d)+\lambda \end{pmatrix}" # [[15, 28], [28, 57]]
        
        matrix_lambda_I = MathTex(ridge_matrix_val, color=BLACK).scale(0.8)
        matrix_lambda_I_label = MathTex(r"X^T X + \lambda I =", color=BLACK).scale(0.8).next_to(matrix_lambda_I, LEFT)
        lambda_I_group = VGroup(matrix_lambda_I_label, matrix_lambda_I)
        d_definition = MathTex(r"; \text{where } d = a^2 + b^2 + c^2", color=BLACK).scale(0.6).next_to(lambda_I_group, RIGHT)
        
        self.play(
            Transform(
                xtx_group, 
                lambda_I_group, 
            ),
            FadeIn(d_definition)
        )
        self.wait(1)


        # lambda_label = MathTex(r"(\lambda = 1)", color=BLACK).scale(0.8).next_to(lambda_I_group, RIGHT, buff=0.5)
        # self.play(FadeIn(lambda_label))
        # self.play(lambda_I_group.animate.move_to(LEFT * 4))

            
        # matrix_ridge = Matrix(ridge_matrix_val, element_to_mobject_config={"color": BLACK}).scale(0.8)
        # matrix_ridge_label = MathTex(r"(X^T X + \lambda I) =", color=BLACK).scale(0.8).next_to(matrix_ridge, LEFT)
        # ridge_group = VGroup(matrix_ridge_label, matrix_ridge).next_to(lambda_I_group, RIGHT, buff=0.5)
        
        # equals_sign = MathTex("=", color=BLACK).scale(0.8).move_to((lambda_I_group.get_right() + ridge_group.get_left()) / 2)
        
        # self.play(Write(equals_sign), FadeIn(ridge_group))
        # self.wait(1)
        
        # Calculate new determinant
        # new_det = int(round(np.linalg.det(ridge_matrix_val))) # 15*57 - 28*28 = 855 - 784 = 71
        
        new_det_text = MathTex(
            fr"\det(X^T X + \lambda I) = (k^2d + \lambda)(d+\lambda) - (kd)^2 \neq 0",
            color=GREEN
        ).scale(0.6).next_to(lambda_I_group, DOWN, buff=1.5).set_x(0)
        
        self.play(Write(new_det_text))
        self.wait(1)
        
        invertible_text = Text("Invertible!", color=GREEN, font_size=24).next_to(new_det_text, DOWN)
        self.play(Write(invertible_text))
        self.wait(3)
