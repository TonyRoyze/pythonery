from manim import *

config.background_color = "#d7e6fa"
config.pixel_width = 1920
config.pixel_height = 1080
config.frame_width = 14.2
config.frame_height = 8

class CrossValidation(Scene):
    def construct(self):
        # Title
        title = Text("K-Fold Cross Validation", color=BLACK).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # 1. Show the full dataset
        dataset_width = 10
        dataset_height = 1
        dataset = Rectangle(width=dataset_width, height=dataset_height, color=BLUE, fill_opacity=0.5, fill_color=BLUE)
        dataset_label = Text("Full Dataset", color=BLACK, font_size=24).next_to(dataset, UP)
        
        self.play(Create(dataset), Write(dataset_label))
        self.wait(1)

        # 2. Split into K=5 folds
        k = 5
        fold_width = dataset_width / k
        folds = VGroup()
        
        for i in range(k):
            fold = Rectangle(width=fold_width, height=dataset_height, color=WHITE, fill_opacity=0.5, fill_color=BLUE)
            # Position folds next to each other
            if i == 0:
                fold.move_to(dataset.get_left() + np.array([fold_width/2, 0, 0]))
            else:
                fold.next_to(folds[-1], RIGHT, buff=0)
            folds.add(fold)

        self.play(ReplacementTransform(dataset, folds), FadeOut(dataset_label))
        self.wait(1)

        # Labels for folds
        fold_labels = VGroup()
        for i, fold in enumerate(folds):
            label = Text(f"Fold {i+1}", color=BLACK, font_size=20).move_to(fold.get_center())
            fold_labels.add(label)
        
        self.play(Write(fold_labels))
        self.wait(1)

        # 3. Iterate through folds
        scores = []
        score_labels = VGroup()
        
        # Position for score table
        score_title = Text("Scores:", color=BLACK, font_size=30).to_corner(UR).shift(DOWN * 1)
        self.play(Write(score_title))

        for i in range(k):
            # Highlight Test Fold
            test_fold = folds[i]
            train_folds = [f for j, f in enumerate(folds) if j != i]
            
            # Animation: Change colors
            self.play(
                test_fold.animate.set_fill(RED, opacity=0.8),
                *[f.animate.set_fill(BLUE, opacity=0.8) for f in train_folds],
                run_time=0.5
            )
            
            # Labels
            test_label = Text("Test", color=RED, font_size=24).next_to(test_fold, DOWN)
            train_label = Text("Train", color=BLUE, font_size=24).next_to(folds, DOWN)
            # Adjust train label to be centered under the whole group roughly, or just indicate generally
            # Actually, let's just label the test fold specifically
            
            self.play(FadeIn(test_label))
            self.wait(0.5)
            
            # Simulate training/eval
            # ... (maybe a small animation of processing)
            
            # Show score
            score_val = 0.80 + (i * 0.02) # Dummy scores
            score_text = Text(f"Fold {i+1}: {score_val:.2f}", color=BLACK, font_size=24)
            score_text.next_to(score_title, DOWN).shift(DOWN * i * 0.5)
            
            self.play(Write(score_text))
            scores.append(score_val)
            score_labels.add(score_text)
            
            self.wait(0.5)
            
            # Reset colors for next iteration (except the last one, we can leave it or reset)
            if i < k - 1:
                self.play(
                    test_fold.animate.set_fill(BLUE, opacity=0.5),
                    FadeOut(test_label)
                )
            else:
                self.play(FadeOut(test_label))

        # 4. Average Score
        avg_score = sum(scores) / k
        avg_line = Line(start=score_labels.get_left(), end=score_labels.get_right(), color=BLACK).next_to(score_labels, DOWN)
        avg_text = Text(f"Average: {avg_score:.2f}", color=BLACK, font_size=24, weight=BOLD).next_to(avg_line, DOWN)
        
        self.play(Create(avg_line), Write(avg_text))
        self.wait(2)

        # Cleanup
        self.play(FadeOut(folds), FadeOut(fold_labels), FadeOut(score_labels), FadeOut(score_title), FadeOut(avg_line), FadeOut(avg_text), FadeOut(title))
