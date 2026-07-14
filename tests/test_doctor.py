from __future__ import annotations

from types import SimpleNamespace

from dapple.cli import doctor


def test_python_support_floor():
    assert doctor.check_python((3, 11, 0)).ok
    assert doctor.check_python((3, 12, 9)).ok
    result = doctor.check_python((3, 10, 14))
    assert not result.ok
    assert "3.11 or 3.12" in result.detail
    assert not doctor.check_python((3, 13, 0)).ok


def test_required_imports_cover_core_package_and_current_dependencies():
    modules = {requirement.module for requirement in doctor.CORE_IMPORTS}
    assert "dapple.data" in modules
    assert "dapple.analysis" in modules
    assert "pandas" in modules
    assert "dask" not in modules
    assert "zarr" not in modules
    assert "psutil" not in modules
    assert "magicgui" not in modules


def test_cli_entrypoints_validate_targets_and_launchers(monkeypatch, tmp_path):
    installed = {
        name: SimpleNamespace(value=target)
        for name, target in doctor.CLI_ENTRYPOINTS.items()
    }
    for name in installed:
        (tmp_path / name).write_text("launcher", encoding="utf-8")

    monkeypatch.setattr(doctor, "_console_entry_points", lambda: installed)
    monkeypatch.setattr(doctor, "_candidate_script_directories", lambda: (tmp_path,))
    monkeypatch.setattr(doctor, "_resolve_target", lambda _target: lambda: None)

    result = doctor.check_cli_entrypoints()
    assert result.ok
    assert "4 command" in result.detail


def test_cli_entrypoints_report_missing_wrapper(monkeypatch, tmp_path):
    installed = {
        name: SimpleNamespace(value=target)
        for name, target in doctor.CLI_ENTRYPOINTS.items()
    }
    monkeypatch.setattr(doctor, "_console_entry_points", lambda: installed)
    monkeypatch.setattr(doctor, "_candidate_script_directories", lambda: (tmp_path,))
    monkeypatch.setattr(doctor, "_resolve_target", lambda _target: lambda: None)

    result = doctor.check_cli_entrypoints()
    assert not result.ok
    assert "launcher" in result.detail


def test_check_import_reports_version():
    requirement = doctor.ImportRequirement("Example", "example", "example-dist")
    result = doctor.check_import(
        requirement,
        importer=lambda _name: SimpleNamespace(__version__="fallback"),
        version_getter=lambda _name: "1.2.3",
    )
    assert result.ok
    assert result.detail == "1.2.3"


def test_check_import_turns_exception_into_failure():
    requirement = doctor.ImportRequirement("Example", "example", "example-dist")

    def fail_import(_name: str):
        raise ImportError("broken binary")

    result = doctor.check_import(requirement, importer=fail_import)
    assert not result.ok
    assert result.required
    assert "broken binary" in result.detail


def test_manifest_inventory_requires_cohort_widget(monkeypatch):
    contributions = SimpleNamespace(
        readers=[SimpleNamespace(command=f"reader.{index}") for index in range(3)],
        widgets=[SimpleNamespace(command=f"widget.{index}") for index in range(7)],
    )
    manifest = SimpleNamespace(
        display_name="DAPPLE",
        contributions=contributions,
    )
    manager = SimpleNamespace(
        discover=lambda: None,
        get_manifest=lambda _name: manifest,
    )
    fake_npe2 = SimpleNamespace(
        PluginManager=SimpleNamespace(instance=lambda: manager),
    )
    monkeypatch.setattr(
        doctor.importlib,
        "import_module",
        lambda name: fake_npe2 if name == "npe2" else None,
    )

    result = doctor.check_manifest()
    assert not result.ok
    assert "CohortWidget" in result.detail


def test_manifest_target_resolves_cohort_widget_as_optional_headless(monkeypatch):
    contributions = SimpleNamespace(
        commands=[
            SimpleNamespace(
                id="dapple.open_cohort",
                python_name="dapple.widgets.cohort:CohortWidget",
            )
        ],
        readers=[],
        widgets=[SimpleNamespace(command="dapple.open_cohort")],
    )
    manifest = SimpleNamespace(contributions=contributions)
    manager = SimpleNamespace(
        discover=lambda: None,
        get_manifest=lambda _name: manifest,
    )
    fake_npe2 = SimpleNamespace(
        PluginManager=SimpleNamespace(instance=lambda: manager),
    )
    cohort_widget = type("CohortWidget", (), {})
    monkeypatch.setattr(
        doctor.importlib,
        "import_module",
        lambda name: fake_npe2 if name == "npe2" else None,
    )
    monkeypatch.setattr(doctor, "_resolve_target", lambda _target: cohort_widget)

    results = doctor.check_manifest_targets(require_gui=False)
    assert len(results) == 1
    assert results[0].ok
    assert not results[0].required


def test_main_returns_zero_when_required_checks_pass(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda *, require_gui: [
            doctor.CheckResult("core", True, "ok"),
            doctor.CheckResult("gui", require_gui, "ok" if require_gui else "skipped"),
        ],
    )
    assert doctor.main([]) == 0
    output = capsys.readouterr().out
    assert "[PASS] core" in output
    assert "DAPPLE is ready" in output


def test_main_returns_nonzero_on_required_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "run_checks",
        lambda *, require_gui: [
            doctor.CheckResult("required import", False, "missing", required=True)
        ],
    )
    assert doctor.main([]) == 1
    output = capsys.readouterr().out
    assert "[FAIL] required import" in output
    assert "1 required failure" in output


def test_headless_allows_optional_gui_warning(monkeypatch, capsys):
    seen: list[bool] = []

    def fake_checks(*, require_gui: bool):
        seen.append(require_gui)
        return [doctor.CheckResult("Qt binding", False, "not installed", required=require_gui)]

    monkeypatch.setattr(doctor, "run_checks", fake_checks)
    assert doctor.main(["--headless"]) == 0
    assert seen == [False]
    assert "[WARN] Qt binding" in capsys.readouterr().out
