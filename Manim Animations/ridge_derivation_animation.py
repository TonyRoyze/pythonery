from manim import *
import numpy as np

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class RidgeDerivationAnimation(Scene):
    def construct(self):
        self.show_solution_steps()
        self.wait(2)

    def show_solution_steps(self):
        substrings = [
            r"J_{OLS}(\beta)",
            r"J_{Ridge}(\beta)",
            r"\|y - X\beta\|^2",
            r"\lambda\|\beta\|^2",
            r"(y - X\beta)^T(y - X\beta)",
            r"y^T y",
            r"- 2 \beta^T X^T y",
            r"\beta^T X^T X \beta",
            r"\lambda \beta^T \beta",
            r"\frac{\partial J}{\partial \beta}",
            r"(X^T X + \lambda I)",
            r"\hat{\beta}_{Ridge}",
        ]

        def create_equation(tex_string):
            return MathTex(
                tex_string,
                substrings_to_isolate=substrings,
                color=BLACK
            ).scale(1.1)

        def highlight_parts(mobject, parts, run_time=0.6):
            if not parts:
                return
            self.play(
                *[Indicate(mobject.get_parts_by_tex(part), color=BLACK) for part in parts],
                run_time=run_time
            )

        step_label = None

        def update_step_label(content):
            nonlocal step_label
            if isinstance(content, str):
                new_label = Text(content, font_size=30, color=BLACK)
            else:
                new_label = content.copy()
                if isinstance(new_label, MathTex):
                    new_label.scale(0.7)
                new_label.set_color(BLACK)
            new_label. move_to (UP * 1)
            if step_label is None:
                step_label = new_label
                self.play(FadeIn(step_label))
            else:
                self.play(Transform(step_label, new_label))

        # 1. OLS objective
        # update_step_label("Start with the OLS objective")
        ols_eq = create_equation(r"J_{OLS}(\beta) = \|y - X\beta\|^2")
        ols_eq.move_to(ORIGIN)
        self.play(Write(ols_eq))
        highlight_parts(ols_eq, [r"J_{OLS}(\beta)"])
        self.wait(0.2)

        # 2. Ridge objective (shift left for emphasis)
        update_step_label("Introduce the ridge penalty term")
        ridge_eq = create_equation(r"J_{Ridge}(\beta) = \|y - X\beta\|^2")
        self.play(
            TransformMatchingTex(
                ols_eq,
                ridge_eq,
                transform_mismatches=True,
                key_map={
                    r"J_{OLS}(\beta)": r"J_{Ridge}(\beta)",
                },
                path_arc=PI / 18,
            )
        )
        ridge_eq2 = create_equation(r"J_{Ridge}(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2")
        self.play(
            TransformMatchingTex(
                ridge_eq,
                ridge_eq2,
                transform_mismatches=True,
                path_arc=PI / 18,
            )
        )
        highlight_parts(ridge_eq2, [r"J_{Ridge}(\beta)", r"\lambda\|\beta\|^2"])
        self.wait(0.2)

        # 3. Expand norm
        update_step_label("Expand the squared norm")
        expanded_eq = create_equation(r"J_{Ridge}(\beta) = (y - X\beta)^T(y - X\beta) + \lambda \beta^T \beta")
        # expanded_eq.move_to(ridge_eq2.get_center())
        self.play(
            TransformMatchingTex(
                ridge_eq2,
                expanded_eq,
                transform_mismatches=True,
                key_map={
                    r"\|y - X\beta\|^2": r"(y - X\beta)^T(y - X\beta)",
                    r"\lambda\|\beta\|^2": r"\lambda \beta^T \beta",
                },
                path_arc=PI / 18,
            )
        )

        # 4. Expand quadratic terms
        update_step_label("Distribute and collect quadratic terms")
        highlight_parts(expanded_eq, [r"(y - X\beta)^T(y - X\beta)"])
        self.wait(0.2)
        quadratic_eq = create_equation(
            r"J_{Ridge}(\beta) = y^T y - 2 \beta^T X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta"
        )
        quadratic_eq.move_to(expanded_eq.get_center())
        self.play(
            TransformMatchingTex(
                expanded_eq,
                quadratic_eq,
                transform_mismatches=True,
                path_arc=PI / 18,
            )
        )
        highlight_parts(quadratic_eq, [r"J_{Ridge}(\beta)", r"y^T y"])
        self.wait(0.2)

        # 5. Gradient
        update_step_label(create_equation(r"\text{Differentiating with respect } \beta"))
        gradient_eq1 = create_equation(
            r"\frac{\partial J}{\partial \beta} = 0 - 2 \beta^T X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta"
        )
        # gradient_eq1.move_to(quadratic_eq.get_center())
        self.play(
            TransformMatchingTex(
                quadratic_eq,
                gradient_eq1,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"J_{Ridge}(\beta)" : r"\frac{\partial J}{\partial \beta}",
                    r"y^T y" : r"0"
                }
            )
        )
        highlight_parts(gradient_eq1, [r"- 2 \beta^T X^T y"])
        self.wait(0.2)


        gradient_eq2 = create_equation(
            r"\frac{\partial J}{\partial \beta} = 0 - 2 X^T y + \beta^T X^T X \beta + \lambda \beta^T \beta"
        )
        # gradient_eq2.move_to(gradient_eq1.get_center())
        self.play(
            TransformMatchingTex(
                gradient_eq1,
                gradient_eq2,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"- 2 \beta^T X^T y" : r"- 2 X^T y"
                }
            )
        )
        highlight_parts(gradient_eq2, [r"\beta^T X^T X \beta"])
        self.wait(0.2)

        gradient_eq3 = create_equation(
            r"\frac{\partial J}{\partial \beta} = 0 - 2 X^T y + 2X^T X \beta + \lambda \beta^T \beta"
        )
        # gradient_eq3.move_to(gradient_eq2.get_center())
        self.play(
            TransformMatchingTex(
                gradient_eq2,
                gradient_eq3,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"+ \beta^T X^T X \beta" : r"+ 2X^T X \beta"
                }
            )
        )
        highlight_parts(gradient_eq3, [r"\lambda \beta^T \beta"])
        self.wait(0.2)        


        gradient_eq4 = create_equation(
            r"\frac{\partial J}{\partial \beta} = 0 - 2 X^T y + 2X^T X \beta + 2\lambda \beta"
        )
        # gradient_eq4.move_to(gradient_eq3.get_center())
        self.play(
            TransformMatchingTex(
                gradient_eq3,
                gradient_eq4,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"+ \lambda \beta^T \beta" : r"+ 2\lambda \beta"
                }
            )
        )
        self.wait(0.2)   

        update_step_label("Collect terms")
        gradient_eq5 = create_equation(
            r"\frac{\partial J}{\partial \beta} = 2(X^T X + \lambda I)\beta - 2X^T y"
        )
        # gradient_eq5.move_to(gradient_eq4.get_center())
        self.play(
            TransformMatchingTex(
                gradient_eq4,
                gradient_eq5,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"+ 2X^T X \beta + 2\lambda \beta" : r"2(X^T X + \lambda I)\beta",
                    r"0 - 2X^T y" : r"- 2X^T y"
                }
            )
        )
        highlight_parts(gradient_eq5, [r"\frac{\partial J}{\partial \beta}"])
        self.wait(0.2)

        # 6. Normal equation
        update_step_label("Set the gradient to zero")
        normal_eq1 = create_equation(r"0 = 2(X^T X + \lambda I)\beta - 2X^T y")
        normal_eq1.move_to(gradient_eq5.get_center())
        self.play(
            TransformMatchingTex(
                gradient_eq5,
                normal_eq1,
                transform_mismatches=True,
                path_arc=PI / 18,
                key_map = {
                    r"\frac{\partial J}{\partial \beta}" : r"0"   
                }
            )
        )
        highlight_parts(normal_eq1, [r"(X^T X + \lambda I)"])
        self.wait(0.2)


        normal_eq2 = create_equation(r"(X^T X + \lambda I)\beta = X^T y")
        normal_eq2.move_to(normal_eq1.get_center())
        self.play(
            TransformMatchingTex(
                normal_eq1,
                normal_eq2,
                transform_mismatches=True,      
                path_arc=PI / 18,
                key_map = {
                    r"(X^T X + \lambda I)\beta" : r"(X^T X + \lambda I)\beta"
                }
            )
        )
        highlight_parts(normal_eq2, [r"(X^T X + \lambda I)\beta"])
        self.wait(0.2)

        # 7. Closed-form solution
        update_step_label(create_equation(r"\text{Solve for } (\hat{\beta}_{Ridge})"))
        solution_eq = create_equation(r"\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y")
        solution_eq.move_to(normal_eq2.get_center())
        self.play(
            TransformMatchingTex(
                normal_eq2,
                solution_eq,
                transform_mismatches=True,      
                path_arc=PI / 18,
                key_map = {
                    r"(X^T X + \lambda I)\beta" : r"\hat{\beta}_{Ridge}"
                }   
            )
        )
        highlight_parts(solution_eq, [r"\hat{\beta}_{Ridge}"])

        solution_box = SurroundingRectangle(solution_eq, color=BLACK, buff=0.2)
        self.play(Create(solution_box))
