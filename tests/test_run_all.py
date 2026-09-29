import pytest

from src import company_bundle as cb, run_all


def test_run_all_refuses_incomplete_company_and_prints_no_model_numbers(capsys):
    if cb.check_readiness("eternal").ready:
        pytest.skip("Eternal inputs are complete")
    code = run_all.run("eternal")
    out = capsys.readouterr().out
    assert code == 2 and "CANNOT BUILD eternal" in out and "STOPPED" in out
    for forbidden in ("WACC ", "value/share", "IMPLIED", "MONTE CARLO", "P(value", "RESULT: ALL"):
        assert forbidden not in out, forbidden


def test_run_all_fixture_runs_every_stage_and_labels_output_as_test(capsys):
    from pathlib import Path
    from src import build_excel as be
    if not Path(be.SOFFICE).exists():
        pytest.skip("LibreOffice not installed")
    code = run_all.run(None, fixture=True)
    out = capsys.readouterr().out
    assert code == 0
    assert "SYNTHETIC TEST DATA" in out and "[TEST DATA]" in out
    for stage in ("[1] DATA", "[2] WACC", "[3] FORWARD", "[4] IMPLIED", "[5] MONTE CARLO", "[6] PLOTS", "[7] EXCEL",
                  "[8] CHECKS TAB", "[9] AUDIT"):
        assert stage in out, stage
    assert "ALL STAGES PASSED" in out


def test_cli_usage_errors():
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "src.run_all"], capture_output=True, text=True)
    assert r.returncode != 0 and "usage" in (r.stderr + r.stdout).lower()
