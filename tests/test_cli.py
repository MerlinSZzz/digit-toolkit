"""Sensor-free tests for the command-line interface."""

from digit.cli import build_parser, main


def test_parser_builds_all_commands():
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = actions[0].choices if actions else {}
    for name in (
        "check",
        "view",
        "record",
        "play",
        "snapshot",
        "reference",
        "led",
        "fps",
        "reset",
        "eval",
        "train",
        "predict",
        "selftest",
    ):
        assert name in commands


def test_selftest_runs_without_sensor(capsys):
    assert main(["selftest"]) == 0
    out = capsys.readouterr().out
    assert "selftest" in out
