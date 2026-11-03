from pricing.notes import NUM, check, with_fit

ROWS = with_fit([
    {"segment": "18-20", "exposure": 1078.9, "claims": 219, "observed_frequency": 0.2030, "lgb_frequency": 0.2112,
     "observed_pure_premium": 736.0, "lgb_pure_premium": 414.0, "lgb_relativity": 2.87},
    {"segment": "61-70", "exposure": 9310.3, "claims": 576, "observed_frequency": 0.0619, "lgb_frequency": 0.0605,
     "observed_pure_premium": 99.0, "lgb_pure_premium": 119.0, "lgb_relativity": 0.82},
    {"segment": "71+", "exposure": 5912.9, "claims": 340, "observed_frequency": 0.0575, "lgb_frequency": 0.0620,
     "observed_pure_premium": 181.0, "lgb_pure_premium": 122.0, "lgb_relativity": 0.84},
])


def test_number_regex_skips_labels():
    assert NUM.findall("R82 and 18-20 at 1.32, 12,995 euros") == ["18", "20", "1.32", "12,995"]


def test_correct_note_passes():
    note = "The 18-20 segment has the highest relativity at 2.87. The 61-70 segment has the lowest relativity at 0.82."
    assert check(note, ROWS) == (True, [])


def test_wrong_lowest_segment_is_rejected():
    ok, problems = check("The 71+ segment has the lowest relativity at 0.84.", ROWS)
    assert not ok and "lowest is 61-70" in problems


def test_number_from_another_row_is_rejected():
    # 0.0605 is the 61-70 model frequency, not a 71+ number.
    ok, problems = check("For 71+ the model frequency is 0.0605.", ROWS)
    assert not ok and "0.0605" in problems


def test_invented_percentage_is_rejected():
    ok, problems = check("The model overestimates 18-20 by 4.5%.", ROWS)
    assert not ok and "4.5" in problems


def test_reversed_direction_is_rejected():
    assert with_fit(ROWS)[2]["model_fit"] == "model above observed"
    ok, problems = check("For 71+ the model frequency is lower than observed.", ROWS)
    assert not ok and any("71+" in p for p in problems)
