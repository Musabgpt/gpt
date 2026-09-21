from src.microcoder.promotion import PromotionPolicy, decide_promotion


def test_promotes_real_improvement_without_regression():
    d = decide_promotion(
        {"val_loss": 1.0, "anchor_score": 0.80},
        {"val_loss": 0.9, "anchor_score": 0.81},
    )
    assert d.promote


def test_rejects_train_style_loss_win_with_anchor_regression():
    d = decide_promotion(
        {"val_loss": 1.0, "anchor_score": 0.80},
        {"val_loss": 0.9, "anchor_score": 0.75},
    )
    assert not d.promote
    assert any("anchor regression" in r for r in d.reasons)


def test_thresholds_are_explicit():
    d = decide_promotion(
        {"val_loss": 1.0, "anchor_score": 0.80},
        {"val_loss": 0.995, "anchor_score": 0.79},
        PromotionPolicy(min_val_loss_improvement=0.01, max_anchor_regression=0.02),
    )
    assert not d.promote
