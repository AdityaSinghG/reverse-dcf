import pytest

from src import extract_pdf as ex


@pytest.mark.parametrize("tok,val,note", [
    ("20,211", 20211.0, "CLEAN"), ("(876)", -876.0, "CLEAN"), ("( 132)", -132.0, "CLEAN"),
    ("1.396", 1396.0, "GLYPH-CORRECTED"), ("37.779", 37779.0, "GLYPH-CORRECTED"), ("-", 0.0, "CLEAN"),
    ("0.10", 0.10, "CLEAN")])
def test_parse_token(tok, val, note):
    assert ex.parse_token(tok) == (val, note)


def test_numeric_tokens_line_and_label_digits_are_not_columns():
    line = "Total ex1>enses 20,314 17,406 7.433 55,145"
    assert ex.numeric_tokens(line, line.index("20,314")) == ["20,314", "17,406", "7.433", "55,145"]
    # a digit inside the label ("B2B") must not become a column: the spec label has to swallow it
    seg = "Hyperpure supplies (B2B business) 1.034 978 2,295 5.366"
    assert len(ex.numeric_tokens(seg, seg.index("(B2B"))) == 5           # guard trips -> NOT FOUND, never misread
    assert len(ex.numeric_tokens(seg, seg.index("1.034"))) == 4


def test_dash_is_zero_but_only_as_a_standalone_token():
    line = "Cost of materials consumed 47 39 - 54"
    assert ex.numeric_tokens(line) == ["47", "39", "-", "54"]
