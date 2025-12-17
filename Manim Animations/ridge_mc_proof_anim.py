from manim import *

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8


class RidgeMCProof(Scene):
    def construct(self):
      ridge_sol = MathTex(r"\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y").set_color(BLACK)
      ridge_sol2 = MathTex(r"\hat{\beta}_{Ridge} = (X^T X + \alpha I)^{-1} X^T y").set_color(BLACK)
      variance_ols = MathTex(r"Var(\hat{\beta}_{OLS}) = \sigma^2 (X^TX)^{-1}= \sigma^2 V\Lambda^{-1}V^T").set_color(BLACK)
      variance_ridge = MathTex(r"Var(\hat{\beta}_{Ridge}) = \sigma^2 (X^TX + \lambda I)^{-1}= \sigma^2 V(\Lambda + \lambda I)^{-1}V^T").set_color(BLACK)
      self.play(Write(ridge_sol))
      self.play(TransformMatchingTex(ridge_sol, ridge_sol2, path_arc=PI / 18, transform_mismatches=True, key_map={"\lambda": "\alpha"}))

      self.wait(1)

      self.play(ridge_sol.animate.shift(UP * 2.5).scale(0.7))
      

      decomp = MathTex(r"X^TX = V\Lambda V^T").set_color(BLACK)
      self.play(Write(decomp))
      self.wait(1)

      VT = MathTex(r"V^TV = VV^T = I").set_color(BLACK).next_to(decomp, DOWN).scale(0.8)
      LambdaMatrix = MathTex(r"\Lambda = \begin{bmatrix} \lambda_1 & 0 & \cdots \\ 0 & \lambda_2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}").set_color(BLACK).next_to(VT, DOWN).scale(0.8)

      VT_group = VGroup(VT, LambdaMatrix)
      self.play(Write(VT_group))
      self.wait(2)

      self.play(VT_group.animate.shift(RIGHT * 5).scale(0.7))
      self.wait(1)
      
      self.play(FadeOut(decomp))
      ridge_decompLHS = MathTex(r"X^TX + \lambda I =").set_color(BLACK).set_x(-1)
      ridge_decompRHS = MathTex(r"V\Lambda V^T + \lambda I").set_color(BLACK).next_to(ridge_decompLHS, RIGHT)
      self.play(Write(ridge_decompLHS), Write(ridge_decompRHS))
      self.wait(1)

      ridge_decomp2 = MathTex(r"V\Lambda V^T + \lambda VV^T").set_color(BLACK).next_to(ridge_decompLHS, RIGHT)
      self.play(TransformMatchingTex(ridge_decompRHS, ridge_decomp2, path_arc=PI / 18, transform_mismatches=True, key_map={"\lambda I": "\lambda VV^T"}))
      self.wait(1)

      ridge_decomp3 = MathTex(r"V\Lambda V^T + V(\lambda I) V^T").set_color(BLACK).next_to(ridge_decompLHS, RIGHT)
      self.play(TransformMatchingTex(ridge_decomp2, ridge_decomp3, path_arc=PI / 18, transform_mismatches=True, key_map={"\lambda VV^T": "V(\lambda I) V^T"}))
      self.wait(1)

      ridge_decomp4 = MathTex(r"V(\Lambda + \lambda I) V^T").set_color(BLACK).next_to(ridge_decompLHS, RIGHT)
      self.play(TransformMatchingTex(ridge_decomp3, ridge_decomp4, path_arc=PI / 18, transform_mismatches=True, key_map={"V\Lambda V^T + V(\lambda I) V^T": "V(\Lambda + \lambda I) V^T"}))
      self.wait(1)
        
        # Eigenvalue transformation
        # arrow = Arrow(start=UP, end=DOWN, color=BLACK).next_to(decomp, DOWN)
        # change = MathTex(r"\lambda_j \rightarrow \lambda_j + \lambda").next_to(arrow, DOWN)
        # change.set_color(YELLOW)
        
        # self.play(Create(arrow), Write(change))
        # self.wait(2)
        
        # explanation = VGroup(
        #     Text("Small eigenvalues → significantly bigger", font_size=24),
        #     Text("Tiny eigenvalues → lifted away from zero", font_size=24)
        # ).arrange(DOWN, aligned_edge=LEFT).next_to(change, DOWN, buff=0.5)
        
        # self.play(Write(explanation))
        # self.wait(3)
        
        # self.play(
        #     FadeOut(header_2), FadeOut(decomp), FadeOut(arrow), 
        #     FadeOut(change), FadeOut(explanation)
        # )

        # # 3. Variance
        # header_3 = Text("3. Impact on Variance", font_size=32, color=BLACK).next_to(title, DOWN, buff=0.5)
        # self.play(Write(header_3))
        
        # var_intro = Text("Variance term depends on inverse eigenvalues:", font_size=24).next_to(header_3, DOWN)
        # self.play(Write(var_intro))
        
        # # Compare OLS vs Ridge
        # ols_var = MathTex(r"\text{OLS: } \frac{1}{\lambda_j}")
        # ridge_var = MathTex(r"\text{Ridge: } \frac{1}{\lambda_j + \lambda}")
        
        # ols_var.shift(LEFT * 3)
        # ridge_var.shift(RIGHT * 3)
        
        # self.play(Write(ols_var), Write(ridge_var))
        # self.wait(1)
        
        # # 4. Example
        # example_box = SurroundingRectangle(VGroup(ols_var, ridge_var), color=BLACK, buff=2) 
        # # Actually lets just clear and assume example values
        # self.play(FadeOut(ols_var), FadeOut(ridge_var), FadeOut(var_intro))
        
        # text_example = Text("Example: Multicollinearity case", font_size=28, color=BLACK).next_to(header_3, DOWN)
        # self.play(Write(text_example))
        
        # values = MathTex(r"\lambda_j = 0.01", r", \quad \lambda = 1")
        # values.next_to(text_example, DOWN)
        # self.play(Write(values))
        
        # # Calculations
        # calc_ols = MathTex(r"\text{OLS: } \frac{1}{0.01} = 100")
        # calc_ridge = MathTex(r"\text{Ridge: } \frac{1}{0.01 + 1} \approx 1")
        
        # calc_ols.scale(1.2).set_color(RED).move_to(LEFT * 3 + DOWN * 0.5)
        # calc_ridge.scale(1.2).set_color(GREEN).move_to(RIGHT * 3 + DOWN * 0.5)
        
        # self.play(Write(calc_ols))
        # self.wait(1)
        # self.play(Write(calc_ridge))
        # self.wait(1)
        
        # final_msg = Text("Variance explodes vs. Stabilized", font_size=32, color=BLACK).next_to(VGroup(calc_ols, calc_ridge), DOWN, buff=1)
        # self.play(Write(final_msg))
        # self.wait(3)

        # self.play(FadeOut(Group(*self.mobjects)))

        # # Summary
        # summary = Text("Ridge lifts tiny eigenvalues\nto prevent variance explosion.", font_size=36, color=BLACK)
        # self.play(Write(summary))
        # self.wait(3)
