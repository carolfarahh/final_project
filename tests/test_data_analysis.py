def test_data_describe_basic():
    df = pd.DataFrame({
        "group": ["A", "A", "B"],
        "value": [1, 3, 2]
    })

    desc = data_describe(df, "group", "value")

    assert "mean" in desc.columns
    assert desc.loc["A", "mean"] == 2

def test_levene_equal_variance():
    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "value": np.random.normal(0, 1, 20)
    })

    stat, p = levene_test(df, "group", "value")
    assert 0 <= p <= 1

def test_anova_detects_difference():
    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "value": np.concatenate([np.ones(10), np.ones(10) * 5])
    })

    f, p = anova(df, "group", "value")
    assert p < 0.05

def test_welch_anova_runs():
    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10,
        "value": np.concatenate([
            np.random.normal(0, 1, 10),
            np.random.normal(0, 5, 10)
        ])
    })

    f, p = welch_anova(df, "group", "value")
    assert 0 <= p <= 1

def test_tukey_output():
    df = pd.DataFrame({
        "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        "value": np.concatenate([np.ones(10), np.ones(10)*2, np.ones(10)*3])
    })

    result = tukey(df, "group", "value")

    assert not result.empty
    assert {"A", "B", "C"}.issubset(set(result["group1"]) | set(result["group2"]))
