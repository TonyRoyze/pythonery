from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class RidgeGeneralization(Scene):
    def construct(self):
        # 1. Setup Axes
        axes = Axes(
            x_range=[-1, 7, 1],
            y_range=[-1, 16, 2],
            x_length=10,
            y_length=6,
            axis_config={"color": WHITE},
        ).add_coordinates()
        
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        self.play(Create(axes), Write(labels))
        
        # 2. Generate Data
        X_train = np.array([1, 2, 3, 4, 5])
        y_train = np.array([3, 6, 8, 9, 12])

        table_data = [["X", "Y"]]
        for x, y in zip(X_train, y_train):
            table_data.append([str(x), str(y)])

        train_data_table = Table(
            table_data,
            include_outer_lines=True,
            line_config={"stroke_color": WHITE},
            h_buff=1,
            v_buff=0.5,
            fill_color=WHITE,
            fill_opacity=1,
        )
        train_data_table.scale(0.5)
        train_data_table.to_corner(UR)

        self.play(FadeIn(train_data_table))
        self.wait(1)

        
        train_dots = VGroup()
        for x, y in zip(X_train, y_train):
            dot = Dot(axes.c2p(x, y), color=BLUE)
            train_dots.add(dot)        
        
        self.play(FadeIn(train_dots))

        self.wait(1)
        
        # 4. Fit Models
        ols_slope, ols_intercept = 2.100, 1.300
        ridge_slope, ridge_intercept = 1.99494949, 0.62121212
        
        # 5. Visualize OLS
        ols_curve = axes.plot(
            lambda x: ols_slope * x + ols_intercept,
            color=RED,
            x_range=[0, 6]
        )
        
        self.play(FadeIn(ols_curve))
        self.wait(2)
        
        # 6. Visualize Ridge
        ridge_curve = axes.plot(
            lambda x: ridge_slope * x + ridge_intercept,
            color=YELLOW,
            x_range=[0, 6]
        )
        
        self.play( FadeIn(ridge_curve))
        self.wait(2)

        X_test = np.array([1.5, 3.5, 4.5])
        y_test = np.array([4, 8, 10])

        test_data_table_data = [["X", "Y"]]
        for x, y in zip(X_test, y_test):
            test_data_table_data.append([str(x), str(y)])

        test_data_table = Table(
            test_data_table_data,
            include_outer_lines=True,
            line_config={"stroke_color": WHITE},
            h_buff=1,
            v_buff=0.5,
            fill_color=WHITE,
            fill_opacity=1
        )
        test_data_table.scale(0.5)
        test_data_table.to_corner(UR)

        self.play(
          Transform(
            train_data_table, 
            test_data_table,
            transform_mismatches=True,
            path_arc=PI / 18
          )
        )
        self.wait(1)
        
        # 7. Show Unseen Data
        test_dots = VGroup()
        for x, y in zip(X_test, y_test):
            dot = Dot(axes.c2p(x, y), color=GREEN)
            test_dots.add(dot)
            
        self.play(FadeIn(test_dots))
        self.wait(1)

        
        
        # 8. Highlight Generalization
        # Show lines from test points to curves to visualize error
        
        ols_errors = VGroup()
        ridge_errors = VGroup()
        
        for x, y in zip(X_test, y_test):
            # OLS Error line
            y_pred_ols = ols_slope * x + ols_intercept
            line_ols = DashedLine(
                start=axes.c2p(x, y),
                end=axes.c2p(x, y_pred_ols),
                color=RED,
                stroke_opacity=0.5
            )
            ols_errors.add(line_ols)
            
            # Ridge Error line
            y_pred_ridge = ridge_slope * x + ridge_intercept
            line_ridge = DashedLine(
                start=axes.c2p(x, y),
                end=axes.c2p(x, y_pred_ridge),
                color=YELLOW,
                stroke_opacity=0.5
            )
            ridge_errors.add(line_ridge)

        rmse_ols = 1.089
        rmse_ridge = 0.684

        rmse_ols_label = MathTex(r"RMSE_{OLS} \approx ", rmse_ols, color=BLACK).scale(0.6).next_to(test_data_table, DOWN, aligned_edge=RIGHT)
        rmse_ridge_label = MathTex(r"RMSE_{Ridge} \approx ", rmse_ridge, color=BLACK).scale(0.6).next_to(rmse_ols_label, DOWN, aligned_edge=RIGHT)

        self.play(FadeIn(ols_errors))
        self.play(FadeIn(rmse_ols_label))
        self.wait(1)
        
        self.play(FadeIn(ridge_errors))
        self.play(FadeIn(rmse_ridge_label))
        self.wait(3)

        # 9. Outlier Scene
        self.play(
            FadeOut(test_dots),
            FadeOut(ols_errors),
            FadeOut(ridge_errors),
            FadeOut(rmse_ols_label),
            FadeOut(rmse_ridge_label),
            FadeOut(test_data_table),
            FadeOut(train_data_table)
        )
        
        # Identify the point to move (last one)
        outlier_dot = train_dots[-1]
        new_y = 2 # Make it a significant outlier (drop from 12 to 2)
        
        # Animate point moving
        self.play(
            outlier_dot.animate.move_to(axes.c2p(5, new_y)),
            run_time=2
        )
        
        # Fit Models
        ols_slope_new, ols_intercept_new = 0.100, 5.300
        ridge_slope_new, ridge_intercept_new = 1.153, 0.823
        
        # New curves
        ols_curve_new = axes.plot(
            lambda x: ols_slope_new * x + ols_intercept_new,
            color=RED,
            x_range=[0, 6]
        )
        
        ridge_curve_new = axes.plot(
            lambda x: ridge_slope_new * x + ridge_intercept_new,
            color=YELLOW,
            x_range=[0, 6]
        )
        
        # Animate transformation
        self.play(
            Transform(ols_curve, ols_curve_new),
            run_time=2
        )

        self.wait(1)

        self.play(
            Transform(ridge_curve, ridge_curve_new),
            run_time=2
        )