from manim import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge
import os

# Configuration matching ridge_derivation_animation.py
config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

# Color palette for features (using Manim colors)
FEATURE_COLORS = [
    BLUE, RED, GREEN, YELLOW, PURPLE,
    TEAL, PINK, ORANGE, MAROON, GOLD
]


def load_and_preprocess_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "credit.csv")
    
    df = pd.read_csv(csv_path)
    
    X = df.drop("Balance", axis=1)
    y = df["Balance"]
    
    if X.columns[0] == "Unnamed: 0":
        X = X.drop(X.columns[0], axis=1)
    
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_processed = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_processed[col] = le.fit_transform(X[col].astype(str))
    
    X_processed = X_processed.apply(pd.to_numeric, errors="coerce")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_processed.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_processed.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X_processed.columns.tolist()


class RidgeTracePlotScene(Scene):
    """Animate the ridge trace plot showing coefficient paths vs lambda"""
    
    def construct(self):
        X_train_scaled, _, y_train, _, feature_names = load_and_preprocess_data()
        
        lambdas = np.logspace(-2, 4, 100)
        
        coefs = []
        for alpha in lambdas:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_scaled, y_train)
            coefs.append(ridge.coef_)
        coefs = np.array(coefs)
        
        axes = Axes(
            x_range=[-2, 4, 1],
            y_range=[-300, 500, 100],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2},
            x_axis_config={
                "numbers_to_include": np.arange(-2, 5),
                "numbers_with_elongated_ticks": np.arange(-2, 5),
            },
            y_axis_config={
                "numbers_to_include": np.array([-300, 300, 500]),
            },
            tips=False,
        ).shift(DOWN * 0.5)
        
        # X-axis is log scale, so we need to transform
        log_lambdas = np.log10(lambdas)
        
        # Show axes
        self.play(Create(axes))
        
        # Axis labels
        x_label = axes.get_x_axis_label(
            MathTex(r"\log_{10}(\lambda)", color=BLACK), 
            edge=RIGHT, direction=RIGHT, buff=0.3
        ).scale(0.6)
        y_label = axes.get_y_axis_label(
            Text("Standardized Coefficients", color=BLACK),
            edge=UP, direction=UP, buff=0.3
        ).scale(0.5)
        self.play(Write(x_label), Write(y_label))
        self.wait(0.5)
        
        # Plot each coefficient path
        coefficient_lines = []
        coefficient_labels = []
        
        for i, (feature, color) in enumerate(zip(feature_names, FEATURE_COLORS[:len(feature_names)])):
            
            # Create points for this feature
            points = [
                axes.coords_to_point(log_lambdas[j], coefs[j, i])
                for j in range(len(lambdas))
            ]
            
            # Create line
            line = VMobject()
            line.set_points_as_corners(points)
            main_labels = ["Income", "Limit", "Rating", "Student"]
            if feature in main_labels:
                line.set_stroke(color=color, width=3)
            else:
                line.set_stroke(color=GREY, width=1)
            
            coefficient_lines.append(line)
            
            self.play(Create(line), run_time=0.5)
        
        # Add legend
        legend = VGroup()
        for i, (feature, color) in enumerate(zip(feature_names, FEATURE_COLORS[:len(feature_names)])):
            main_labels = ["Income", "Limit", "Rating", "Student"]
            if feature in main_labels:
                label = Text(feature, color=color).scale(0.5)
                legend.add(label)
        legend.arrange(DOWN, buff=0.2)
        legend.to_corner(UR, buff=2.0)
        self.play(FadeIn(legend))
        
        # Final wait
        self.wait(2)

        lambda_tracker = ValueTracker(-2)
        
        def get_vertical_line():
            x_val = lambda_tracker.get_value()
            line = Line(
                axes.c2p(x_val, -300),
                axes.c2p(x_val, 500),
                color=GRAY,
                stroke_width=2
            )
            return line
            
        vertical_line = always_redraw(get_vertical_line)
        
        # Label for lambda
        lambda_label = always_redraw(lambda: 
            MathTex(f"\\log(\\lambda) = {lambda_tracker.get_value():.2f}", color=BLACK).scale(0.5)
            .next_to(vertical_line, UP, buff=0.1)
        )

        self.play(Create(vertical_line), Write(lambda_label))
        
        # Animate scanning
        self.play(lambda_tracker.animate.set_value(4), run_time=5, rate_func=linear)
        
        # Move back to a "good" lambda (where coefficients stabilize but before they shrink too much)
        # Around log(lambda) = 2 or 3
        self.play(lambda_tracker.animate.set_value(1.0303041555), run_time=2)
        
        self.wait(2)


class CrossValidationScene(Scene):
    """Animate the cross-validation error curve with optimal lambda highlight"""
    
    def construct(self):
        # Load data
        X_train_scaled, _, y_train, _, _ = load_and_preprocess_data()
        
        # Perform cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_lambdas = np.logspace(-1, 2, 100)
        cv_scores_mean = []
        cv_scores_std = []
        
        for alpha in cv_lambdas:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(
                ridge, X_train_scaled, y_train,
                cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1
            )
            cv_scores_mean.append(-np.mean(scores))
            cv_scores_std.append(np.std(scores))
        
        cv_scores_mean = np.array(cv_scores_mean)
        cv_scores_std = np.array(cv_scores_std)
        
        # Find both optimal lambdas:
        # 1. minimal CV error ("min lambda")
        # 2. 1-standard-error rule ("1SE lambda": largest lambda within 1 std of min error)

        # -- Minimal CV error --
        min_cv_error = np.min(cv_scores_mean)
        min_cv_error_idx = np.argmin(cv_scores_mean)
        min_lambda = cv_lambdas[min_cv_error_idx]
        min_cv_std = cv_scores_std[min_cv_error_idx]

        # -- 1SE rule: largest lambda with error within 1 std of min error --
        within_1sd = np.where(cv_scores_mean <= min_cv_error + min_cv_std)[0]
        lambda_1se_idx = within_1sd[-1]  # pick largest lambda index within 1sd
        lambda_1se = cv_lambdas[lambda_1se_idx]
        cv_error_1se = cv_scores_mean[lambda_1se_idx]
        
        log_cv_lambdas = np.log10(cv_lambdas)
        log_min_lambda = np.log10(min_lambda)
        log_lambda_1se = np.log10(lambda_1se)
        
        # Create axes
        axes = Axes(
            x_range=[-1, 2, 1],
            y_range=[0, 40000, 10000],
            x_length=10,
            y_length=5,
            axis_config={"color": WHITE, "stroke_width": 2},
            x_axis_config={
                "numbers_to_include": np.arange(-1, 2.5, 1),
            },
            y_axis_config={
                "numbers_to_include": np.array([20000, 30000, 40000])
            },
            tips=False,
        ).shift(DOWN * 0.5)
 
        
        # Title
        # title = Text("Cross-Validation Error vs Lambda", 
        #             font_size=36, color=BLACK).to_edge(UP, buff=0.3)
        # self.play(Write(title))
        # self.wait(0.5)
        
        # Show axes
        self.play(Create(axes))
        
        # Axis labels
        x_label = axes.get_x_axis_label(
            MathTex(r"\log_{10}(\lambda)", color=BLACK), 
            edge=RIGHT, direction=RIGHT, buff=0.3
        ).scale(0.6)
        y_label = axes.get_y_axis_label(
            Text("CV MSE", color=BLACK),
            edge=UP, direction=UP, buff=0.3
        ).scale(0.5)

        self.play(Write(x_label), Write(y_label))
        self.wait(0.5)
        
                
        # Create error bars and points
        error_bars = VGroup()
        points = VGroup()
        log_cv_lambdas = np.log10(cv_lambdas)
        
        for i in range(len(cv_lambdas)):
            x_coord = axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i])[0]
            y_coord_mean = axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i])[1]
            y_coord_upper = axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i] + cv_scores_std[i])[1]
            y_coord_lower = axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i] - cv_scores_std[i])[1]
            
            # Error bar
            error_bar = Line(
                [x_coord, y_coord_lower, 0],
                [x_coord, y_coord_upper, 0],
                color=BLUE, stroke_width=1
            )
            
            # Point
            point = Dot(
                axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i]),
                color=BLUE, radius=0.03
            )
            
            error_bars.add(error_bar)
            points.add(point)
        
        # Animate error bars and points appearing
        self.play(
            LaggedStart(    
                *[Create(bar) for bar in error_bars],
                lag_ratio=0.01,
                run_time=3
            )
        )
        self.play(
            LaggedStart(
                *[FadeIn(point) for point in points],
                lag_ratio=0.01,
                run_time=2
            )
        )
        
        # Create smooth curve
        curve_points = [
            axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i])
            for i in range(len(cv_lambdas))
        ]
        curve = VMobject()
        curve.set_points_as_corners(curve_points)
        curve.set_stroke(color=BLUE, width=3)
        self.play(Create(curve), run_time=2)
        
        # Highlight optimal lambda
        # Highlight min lambda (Green)
        min_x = axes.coords_to_point(log_min_lambda, 0)[0]
        min_y_min = axes.coords_to_point(log_min_lambda, axes.y_range[0])[1]
        min_y_max = axes.coords_to_point(log_min_lambda, axes.y_range[1])[1]
        
        min_line = DashedLine(
            [min_x, min_y_min, 0],
            [min_x, min_y_max, 0],
            color=GREEN, stroke_width=3
        )
        
        min_label = MathTex(
            f"\\lambda_{{min}} = {min_lambda:.2f}",
            color=GREEN
        ).scale(0.7).next_to(min_line, UP, buff=0.2).shift(RIGHT * 0.5)
        
        self.play(Create(min_line), Write(min_label))
        
        min_point = Dot(
            axes.coords_to_point(log_min_lambda, min_cv_error),
            color=GREEN, radius=0.08
        )
        self.play(FadeIn(min_point))

        # Highlight 1SE lambda (Red)
        se_x = axes.coords_to_point(log_lambda_1se, 0)[0]
        se_y_min = axes.coords_to_point(log_lambda_1se, axes.y_range[0])[1]
        se_y_max = axes.coords_to_point(log_lambda_1se, axes.y_range[1])[1]
        
        se_line = DashedLine(
            [se_x, se_y_min, 0],
            [se_x, se_y_max, 0],
            color=RED, stroke_width=3
        )
        
        se_label = MathTex(
            f"\\lambda_{{1se}} = {lambda_1se:.2f}",
            color=RED
        ).scale(0.7).next_to(se_line, UP, buff=0.2).shift(RIGHT * 0.5)
        
        self.play(Create(se_line), Write(se_label))
        
        se_point = Dot(
            axes.coords_to_point(log_lambda_1se, cv_error_1se),
            color=RED, radius=0.08
        )
        self.play(FadeIn(se_point))
              
        # Final wait
        self.wait(2)


class CoefficientShrinkageScene(Scene):
    """Animate coefficient magnitudes at different lambda values"""
    
    def construct(self):
        # Load data
        X_train_scaled, _, y_train, _, feature_names = load_and_preprocess_data()
        
        # Selected lambda values
        selected_lambdas = [0.1, 1, 10, 100, 1000]
        lambda_labels = [f"λ={lam}" for lam in selected_lambdas]
        
        # Calculate coefficients for each lambda
        coef_magnitudes = []
        ols_coefs = None
        
        # OLS (lambda = 0)
        ridge_ols = Ridge(alpha=0.0)
        ridge_ols.fit(X_train_scaled, y_train)
        ols_coefs = np.abs(ridge_ols.coef_)
        
        for lam in selected_lambdas:
            ridge = Ridge(alpha=lam)
            ridge.fit(X_train_scaled, y_train)
            coef_magnitudes.append(np.abs(ridge.coef_))
        
        coef_magnitudes = np.array(coef_magnitudes)
        max_coef = max(ols_coefs.max(), coef_magnitudes.max())
        
        # Create axes
        axes = Axes(
            x_range=[-0.5, len(feature_names) - 0.5, 1],
            y_range=[0, max_coef * 1.1, max_coef * 0.2],
            x_length=12,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2},
            x_axis_config={
                "numbers_to_include": [],
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, max_coef * 1.1, max_coef * 0.2),
            },
            tips=False,
        ).shift(DOWN * 0.5)
        
        # Title
        title = Text("Coefficient Magnitudes at Different Lambda Values", 
                    font_size=32, color=BLACK).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.5)
        
        # Show axes
        self.play(Create(axes))
        
        # Axis labels
        y_label = axes.get_y_axis_label(
            Text("|Coefficient|", color=BLACK, font_size=24),
            edge=LEFT, direction=LEFT, buff=0.3
        )
        self.play(Write(y_label))
        
        # Feature labels on x-axis
        feature_labels = VGroup()
        x_positions = np.arange(len(feature_names))
        for i, feature in enumerate(feature_names):
            label = Text(feature, font_size=14, color=BLACK).rotate(PI/4)
            label.move_to(axes.coords_to_point(i, -max_coef * 0.15))
            feature_labels.add(label)
        
        self.play(FadeIn(feature_labels))
        self.wait(0.5)
        
        # Explanation
        explanation = Text(
            "As lambda increases, coefficients shrink towards zero",
            font_size=24, color=BLACK
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(explanation))
        self.wait(1)
        
        # Create bars for OLS first
        ols_bars = VGroup()
        ols_label = Text("OLS (λ=0)", font_size=16, color=BLACK).to_edge(LEFT, buff=0.5)
        self.play(FadeIn(ols_label))
        
        for i, feature in enumerate(feature_names):
            bar = Rectangle(
                width=0.15,
                height=axes.coords_to_point(0, ols_coefs[i])[1] - axes.coords_to_point(0, 0)[1],
                color=BLUE, fill_opacity=0.7, stroke_color=BLUE, stroke_width=2
            )
            bar.move_to(axes.coords_to_point(i, ols_coefs[i] / 2))
            ols_bars.add(bar)
        
        self.play(LaggedStart(*[Create(bar) for bar in ols_bars], lag_ratio=0.1, run_time=2))
        self.wait(1)
        
        # Create bars for each lambda value
        bar_groups = []
        lambda_colors = [RED, GREEN, YELLOW, PURPLE, ORANGE]
        width = 0.15
        spacing = width * 1.2
        
        for idx, (lam, color) in enumerate(zip(selected_lambdas, lambda_colors)):
            bars = VGroup()
            x_offset = (idx + 1) * spacing - (len(selected_lambdas) + 1) * spacing / 2
            
            for i in range(len(feature_names)):
                bar = Rectangle(
                    width=width,
                    height=axes.coords_to_point(0, coef_magnitudes[idx, i])[1] - axes.coords_to_point(0, 0)[1],
                    color=color, fill_opacity=0.7, stroke_color=color, stroke_width=2
                )
                bar.move_to(axes.coords_to_point(i + x_offset, coef_magnitudes[idx, i] / 2))
                bars.add(bar)
            
            bar_groups.append(bars)
            
            # Lambda label
            lam_label = Text(f"λ={lam}", font_size=18, color=color).to_edge(
                LEFT, buff=0.5 + (idx + 1) * 0.4
            )
            
            # Animate bars appearing
            self.play(
                LaggedStart(*[Create(bar) for bar in bars], lag_ratio=0.1, run_time=1.5),
                FadeIn(lam_label),
                run_time=2
            )
            self.wait(0.5)
        
        # Comparison text
        comparison_text = Text(
            "Ridge regression shrinks all coefficients, but larger ones shrink more",
            font_size=16, color=BLACK
        ).next_to(explanation, UP, buff=0.2)
        self.play(FadeIn(comparison_text))
        self.wait(2)
        
        # Show shrinkage percentage for largest coefficient
        largest_idx = np.argmax(ols_coefs)
        largest_feature = feature_names[largest_idx]
        shrinkage_pct = (ols_coefs[largest_idx] - coef_magnitudes[-1, largest_idx]) / ols_coefs[largest_idx] * 100
        
        shrinkage_text = Text(
            f"{largest_feature} shrinks by {shrinkage_pct:.1f}% from λ=0 to λ=1000",
            font_size=16, color=RED
        ).next_to(comparison_text, UP, buff=0.2)
        self.play(FadeIn(shrinkage_text))
        self.wait(2)
        
        # Final wait
        self.wait(2)


class BiasVarianceTradeoffScene(Scene):
    """Show the bias-variance trade-off with dual plots"""
    
    def construct(self):
        # Load data
        X_train_scaled, _, y_train, _, feature_names = load_and_preprocess_data()
        
        # Cross-validation data
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_lambdas = np.logspace(-1, 2, 100)
        cv_scores_mean = []
        cv_scores_std = []
        
        for alpha in cv_lambdas:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(
                ridge, X_train_scaled, y_train,
                cv=kfold, scoring="neg_mean_squared_error", n_jobs=-1
            )
            cv_scores_mean.append(-np.mean(scores))
            cv_scores_std.append(np.std(scores))
        
        cv_scores_mean = np.array(cv_scores_mean)
        cv_scores_std = np.array(cv_scores_std)
        
        # Find optimal lambda
        min_cv_error = np.min(cv_scores_mean)
        min_cv_std = cv_scores_std[np.argmin(cv_scores_mean)]
        within_1sd = np.where(cv_scores_mean <= min_cv_error + min_cv_std)[0]
        optimal_idx = within_1sd[-1]
        optimal_lambda = cv_lambdas[optimal_idx]
        
        log_cv_lambdas = np.log10(cv_lambdas)
        log_optimal_lambda = np.log10(optimal_lambda)
        
        # Title
        title = Text("Bias-Variance Trade-off in Ridge Regression", 
                    font_size=36, color=BLACK).to_edge(UP, buff=0.2)
        self.play(Write(title))
        self.wait(0.5)
        
        # Create two subplots side by side
        # Left plot: CV error
        left_axes = Axes(
            x_range=[-1, 2, 0.5],
            y_range=[10000, 14000, 1000],
            x_length=5.5,
            y_length=3,
            axis_config={"color": BLACK, "stroke_width": 2},
            x_axis_config={"numbers_to_include": np.arange(-1, 2.5, 0.5)},
            y_axis_config={"numbers_to_include": np.arange(10000, 15000, 1000)},
            tips=False,
        ).shift(LEFT * 3.5 + DOWN * 0.5)
        
        # Right plot: Coefficient magnitudes
        selected_lambdas = [0.1, 1, 10, 100, 1000]
        coef_magnitudes = []
        for lam in selected_lambdas:
            ridge = Ridge(alpha=lam)
            ridge.fit(X_train_scaled, y_train)
            coef_magnitudes.append(np.abs(ridge.coef_))
        coef_magnitudes = np.array(coef_magnitudes)
        max_coef = coef_magnitudes.max()
        
        right_axes = Axes(
            x_range=[-0.5, len(feature_names) - 0.5, 1],
            y_range=[0, max_coef * 1.1, max_coef * 0.2],
            x_length=5.5,
            y_length=3,
            axis_config={"color": BLACK, "stroke_width": 2},
            x_axis_config={"numbers_to_include": []},
            y_axis_config={"numbers_to_include": np.arange(0, max_coef * 1.1, max_coef * 0.2)},
            tips=False,
        ).shift(RIGHT * 3.5 + DOWN * 0.5)
        
        # Show axes
        self.play(Create(left_axes), Create(right_axes))
        
        # Left plot labels
        left_x_label = left_axes.get_x_axis_label(
            MathTex(r"\log_{10}(\lambda)", color=BLACK, font_size=16),
            edge=DOWN, direction=DOWN, buff=0.2
        )
        left_y_label = left_axes.get_y_axis_label(
            Text("CV MSE", color=BLACK, font_size=18),
            edge=LEFT, direction=LEFT, buff=0.2
        )
        left_title = Text("Cross-Validation Error", font_size=22, color=BLACK).next_to(
            left_axes, UP, buff=0.1
        )
        
        # Right plot labels
        right_y_label = right_axes.get_y_axis_label(
            Text("|Coefficient|", color=BLACK, font_size=18),
            edge=LEFT, direction=LEFT, buff=0.2
        )
        right_title = Text("Coefficient Magnitudes", font_size=22, color=BLACK).next_to(
            right_axes, UP, buff=0.1
        )
        
        self.play(
            Write(left_x_label), Write(left_y_label), Write(left_title),
            Write(right_y_label), Write(right_title)
        )
        self.wait(0.5)
        
        # Plot CV error curve
        cv_curve_points = [
            left_axes.coords_to_point(log_cv_lambdas[i], cv_scores_mean[i])
            for i in range(len(cv_lambdas))
        ]
        cv_curve = VMobject()
        cv_curve.set_points_as_corners(cv_curve_points)
        cv_curve.set_stroke(color=BLUE, width=3)
        self.play(Create(cv_curve), run_time=2)
        
        # Highlight optimal lambda
        optimal_x = left_axes.coords_to_point(log_optimal_lambda, 0)[0]
        optimal_y_min = left_axes.coords_to_point(log_optimal_lambda, left_axes.y_range[0])[1]
        optimal_y_max = left_axes.coords_to_point(log_optimal_lambda, left_axes.y_range[1])[1]
        optimal_line = DashedLine(
            [optimal_x, optimal_y_min, 0],
            [optimal_x, optimal_y_max, 0],
            color=RED, stroke_width=2
        )
        self.play(Create(optimal_line))
        
        # Plot coefficient magnitudes
        lambda_colors = [RED, GREEN, YELLOW, PURPLE, ORANGE]
        width = 0.12
        spacing = width * 1.2
        
        for idx, (lam, color) in enumerate(zip(selected_lambdas, lambda_colors)):
            bars = VGroup()
            x_offset = (idx - 2) * spacing
            
            for i in range(len(feature_names)):
                bar = Rectangle(
                    width=width,
                    height=right_axes.coords_to_point(0, coef_magnitudes[idx, i])[1] - right_axes.coords_to_point(0, 0)[1],
                    color=color, fill_opacity=0.7, stroke_color=color, stroke_width=1.5
                )
                bar.move_to(right_axes.coords_to_point(i + x_offset, coef_magnitudes[idx, i] / 2))
                bars.add(bar)
            
            self.play(
                LaggedStart(*[Create(bar) for bar in bars], lag_ratio=0.1, run_time=0.8),
                run_time=1
            )
        
        # Explanation text
        explanation = Text(
            "Small λ → Low bias, high variance (overfitting)",
            font_size=16, color=BLACK
        ).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(explanation))
        self.wait(1)
        
        explanation2 = Text(
            "Large λ → High bias, low variance (underfitting)",
            font_size=16, color=BLACK
        ).next_to(explanation, DOWN, buff=0.2)
        self.play(FadeIn(explanation2))
        self.wait(1)
        
        explanation3 = Text(
            f"Optimal λ = {optimal_lambda:.2f} → Balances bias and variance",
            font_size=16, color=RED
        ).next_to(explanation2, DOWN, buff=0.2)
        self.play(FadeIn(explanation3))
        self.wait(1)
        
        # Mathematical explanation
        formula = MathTex(
            r"\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}",
            color=BLACK, font_size=28
        ).next_to(explanation3, DOWN, buff=0.3)
        self.play(Write(formula))
        self.wait(2)
        
        # Final wait
        self.wait(2)

