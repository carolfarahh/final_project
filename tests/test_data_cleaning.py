def test_stage_filter_basic():
    df = pd.DataFrame({
        "stage": ["I", "II", "III"],
        "value": [1, 2, 3]
    })

    out = stage_filter(df, "stage", ["I", "III"])

    assert len(out) == 2
    assert set(out["stage"]) == {"I", "III"}

def test_stage_filter_empty_result():
    df = pd.DataFrame({"stage": ["I", "II"]})
    out = stage_filter(df, "stage", ["IV"])
    assert out.empty

def test_gene_filter():
    df = pd.DataFrame({
        "gene": ["HTT", "TP53", "HTT"],
        "value": [1, 2, 3]
    })

    out = gene_filter(df, "gene", ["HTT"])
    assert len(out) == 2
    assert (out["gene"] == "HTT").all()

def test_transform_to_float_basic():
    s = pd.Series(["1", "2.5", "bad"])
    out = transform_to_float(s)

    assert out.dtype == float
    assert np.isnan(out.iloc[2])

def test_transform_to_float_preserves_nan():
    s = pd.Series([1, None, "3"])
    out = transform_to_float(s)
    assert np.isnan(out.iloc[1])

def test_isolation_forest_removes_extreme_outlier():
    df = pd.DataFrame({"x": [1, 1, 1, 1, 1000]})
    out = remove_outlier_isolate_forest(df, "x", random_state=42)

    assert 1000 not in out["x"].values

def test_isolation_forest_no_outliers():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = remove_outlier_isolate_forest(df, "x", random_state=42)

    assert len(out) == len(df)
