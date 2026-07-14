"""Validate a DAPPLE installation without opening a graphical window.

``dapple-doctor`` is intentionally implemented with standard-library imports at
module import time.  That lets it explain a partially installed environment
instead of crashing on the first missing scientific or GUI dependency.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import sysconfig
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """One environment check rendered by :func:`print_report`."""

    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class ImportRequirement:
    """A module import and the distribution that supplies its version."""

    name: str
    module: str
    distribution: str


CORE_IMPORTS: tuple[ImportRequirement, ...] = (
    ImportRequirement("DAPPLE package", "dapple", "dapple"),
    ImportRequirement("DAPPLE data model", "dapple.data", "dapple"),
    ImportRequirement("DAPPLE I/O", "dapple.io", "dapple"),
    ImportRequirement("DAPPLE operators", "dapple.ops", "dapple"),
    ImportRequirement("DAPPLE pipeline", "dapple.pipeline", "dapple"),
    ImportRequirement("DAPPLE cohort", "dapple.cohort", "dapple"),
    ImportRequirement("DAPPLE spatial analysis", "dapple.analysis", "dapple"),
    ImportRequirement("NumPy", "numpy", "numpy"),
    ImportRequirement("SciPy", "scipy", "scipy"),
    ImportRequirement("pandas", "pandas", "pandas"),
    ImportRequirement("scikit-image", "skimage", "scikit-image"),
    ImportRequirement("scikit-learn", "sklearn", "scikit-learn"),
    ImportRequirement("pyimzML", "pyimzml", "pyimzml"),
    ImportRequirement("tifffile", "tifffile", "tifffile"),
    ImportRequirement("imagecodecs", "imagecodecs", "imagecodecs"),
    ImportRequirement("netCDF4", "netCDF4", "netCDF4"),
    ImportRequirement("lxml", "lxml", "lxml"),
    ImportRequirement("napari", "napari", "napari"),
    ImportRequirement("npe2", "npe2", "npe2"),
)

GUI_IMPORTS: tuple[ImportRequirement, ...] = (
    ImportRequirement("Qt abstraction", "qtpy", "QtPy"),
    ImportRequirement("plotting widgets", "pyqtgraph", "pyqtgraph"),
    ImportRequirement("DAPPLE widgets", "dapple.widgets", "dapple"),
    ImportRequirement("DAPPLE cohort widget", "dapple.widgets.cohort", "dapple"),
)

CLI_ENTRYPOINTS: dict[str, str] = {
    "dapple-apply-spec": "dapple.cli.apply_spec:main",
    "dapple-cohort-align": "dapple.cli.cohort_align:main",
    "dapple-doctor": "dapple.cli.doctor:main",
    "dapple-analyze-patterns": "dapple.cli.analyze_patterns:main",
}


def check_python(version_info: Sequence[int] | None = None) -> CheckResult:
    """Check the package's tested Python 3.11--3.12 range."""

    current = tuple((version_info or sys.version_info)[:3])
    version = ".".join(str(part) for part in current)
    if (3, 11, 0) <= current < (3, 13, 0):
        return CheckResult("Python", True, f"{version} (supported; requires 3.11 or 3.12)")
    return CheckResult(
        "Python",
        False,
        f"{version} is unsupported; DAPPLE requires Python 3.11 or 3.12",
    )


def check_import(
    requirement: ImportRequirement,
    *,
    required: bool = True,
    importer: Callable[[str], object] = importlib.import_module,
    version_getter: Callable[[str], str] = metadata.version,
) -> CheckResult:
    """Import one module and report its installed distribution version."""

    try:
        module = importer(requirement.module)
    except Exception as exc:  # noqa: BLE001 - this command diagnoses broken imports
        return CheckResult(
            requirement.name,
            False,
            f"cannot import {requirement.module}: {type(exc).__name__}: {exc}",
            required=required,
        )

    try:
        version = version_getter(requirement.distribution)
    except metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "imported; version unavailable"))
    except Exception as exc:  # noqa: BLE001 - version metadata is diagnostic only
        version = f"imported; version lookup failed: {type(exc).__name__}: {exc}"
    return CheckResult(requirement.name, True, version, required=required)


def check_qt_binding(*, required: bool = True) -> CheckResult:
    """Ask QtPy to load a binding without constructing a QApplication."""

    try:
        qtpy = importlib.import_module("qtpy")
        importlib.import_module("qtpy.QtCore")
        api_name = str(getattr(qtpy, "API_NAME", "unknown Qt binding"))
    except Exception as exc:  # noqa: BLE001 - report every binding/load failure
        return CheckResult(
            "Qt binding",
            False,
            f"not usable: {type(exc).__name__}: {exc}",
            required=required,
        )
    return CheckResult("Qt binding", True, api_name, required=required)


def _resolve_target(
    python_name: str,
    *,
    importer: Callable[[str], object] = importlib.import_module,
) -> object:
    """Resolve a ``module:attribute`` target without invoking it."""

    module_name, separator, attribute_path = python_name.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"invalid Python target {python_name!r}")
    target = importer(module_name)
    for attribute in attribute_path.split("."):
        target = getattr(target, attribute)
    return target


def _console_entry_points() -> dict[str, object]:
    """Return installed console-script entry points keyed by command name."""

    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group="console_scripts")
    else:  # pragma: no cover - compatibility with older importlib-metadata
        selected = entry_points.get("console_scripts", ())
    return {entry_point.name: entry_point for entry_point in selected}


def _candidate_script_directories() -> tuple[Path, ...]:
    """Find script directories for both this interpreter and DAPPLE's install."""

    candidates = {
        Path(sysconfig.get_path("scripts")),
        Path(sys.executable).resolve().parent,
    }
    try:
        install_root = Path(metadata.distribution("dapple").locate_file("")).resolve()
    except Exception:  # noqa: BLE001 - missing metadata is reported by the caller
        pass
    else:
        # A wheel normally lives in <venv>/Lib/site-packages (Windows) or
        # <venv>/lib/pythonX.Y/site-packages (POSIX). These candidates also cover
        # an editable installation whose doctor is run through another interpreter.
        for ancestor in (install_root, *tuple(install_root.parents)[:4]):
            candidates.add(ancestor / "Scripts")
            candidates.add(ancestor / "bin")
    return tuple(sorted(candidates, key=lambda path: str(path).casefold()))


def _find_script_wrapper(name: str, directories: Sequence[Path]) -> Path | None:
    suffixes = ("", ".exe", "-script.py", ".cmd")
    for directory in directories:
        for suffix in suffixes:
            candidate = directory / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def check_cli_entrypoints() -> CheckResult:
    """Validate every declared CLI target and its installed launcher wrapper."""

    try:
        installed = _console_entry_points()
        script_directories = _candidate_script_directories()
        details: list[str] = []
        for name, expected_target in CLI_ENTRYPOINTS.items():
            entry_point = installed.get(name)
            if entry_point is None:
                raise LookupError(f"missing console entry point {name!r}")
            actual_target = str(getattr(entry_point, "value", ""))
            if actual_target != expected_target:
                raise ValueError(
                    f"{name!r} targets {actual_target!r}, expected {expected_target!r}"
                )
            target = _resolve_target(actual_target)
            if not callable(target):
                raise TypeError(f"{actual_target!r} is not callable")
            wrapper = _find_script_wrapper(name, script_directories)
            if wrapper is None:
                raise FileNotFoundError(f"installed launcher for {name!r} was not found")
            details.append(name)
    except Exception as exc:  # noqa: BLE001 - diagnose broken package metadata/launchers
        return CheckResult(
            "CLI entry points",
            False,
            f"validation failed: {type(exc).__name__}: {exc}",
        )
    return CheckResult(
        "CLI entry points",
        True,
        f"{len(details)} command target(s) and launcher(s) resolved",
    )


def check_manifest() -> CheckResult:
    """Discover DAPPLE and validate its expected contribution inventory."""

    try:
        npe2 = importlib.import_module("npe2")
        manager = npe2.PluginManager.instance()
        manager.discover()
        manifest = manager.get_manifest("dapple")
        if manifest is None:
            raise LookupError("npe2 did not return a 'dapple' manifest")
        contributions = manifest.contributions
        n_readers = len(contributions.readers or ())
        n_widgets = len(contributions.widgets or ())
        widget_commands = {widget.command for widget in contributions.widgets or ()}
        if "dapple.open_cohort" not in widget_commands:
            raise LookupError("CohortWidget contribution 'dapple.open_cohort' is missing")
        if n_readers != 3 or n_widgets != 7:
            raise ValueError(
                f"expected 3 readers and 7 widgets, found {n_readers} and {n_widgets}"
            )
        detail = (
            f"{manifest.display_name} discovered "
            f"({n_readers} reader(s), {n_widgets} widget(s))"
        )
    except Exception as exc:  # noqa: BLE001 - plugin discovery can fail in many layers
        return CheckResult(
            "napari plugin manifest",
            False,
            f"discovery failed: {type(exc).__name__}: {exc}",
        )
    return CheckResult("napari plugin manifest", True, detail)


def check_manifest_targets(*, require_gui: bool = True) -> list[CheckResult]:
    """Resolve every reader/widget command target declared in the npe2 manifest."""

    try:
        npe2 = importlib.import_module("npe2")
        manager = npe2.PluginManager.instance()
        manager.discover()
        manifest = manager.get_manifest("dapple")
        if manifest is None:
            raise LookupError("npe2 did not return a 'dapple' manifest")
        contributions = manifest.contributions
        commands = {command.id: command for command in contributions.commands or ()}
        reader_ids = {reader.command for reader in contributions.readers or ()}
        widget_ids = {widget.command for widget in contributions.widgets or ()}
    except Exception as exc:  # noqa: BLE001 - discovery can fail in many layers
        return [
            CheckResult(
                "napari command targets",
                False,
                f"discovery failed: {type(exc).__name__}: {exc}",
            )
        ]

    results: list[CheckResult] = []
    for command_id in sorted(reader_ids | widget_ids):
        is_widget = command_id in widget_ids
        required = require_gui or not is_widget
        try:
            command = commands[command_id]
            python_name = str(command.python_name)
            target = _resolve_target(python_name)
            if command_id == "dapple.open_cohort" and getattr(target, "__name__", "") != "CohortWidget":
                raise TypeError(f"{python_name!r} did not resolve to CohortWidget")
            detail = python_name
            ok = True
        except Exception as exc:  # noqa: BLE001 - report any broken contribution target
            detail = f"cannot resolve: {type(exc).__name__}: {exc}"
            ok = False
        results.append(
            CheckResult(
                f"napari target {command_id}",
                ok,
                detail,
                required=required,
            )
        )
    return results


def run_checks(*, require_gui: bool = True) -> list[CheckResult]:
    """Run all installation checks in a stable, user-facing order."""

    results = [check_python()]
    results.extend(check_import(requirement) for requirement in CORE_IMPORTS)
    results.append(check_cli_entrypoints())
    results.append(check_qt_binding(required=require_gui))
    results.extend(
        check_import(requirement, required=require_gui) for requirement in GUI_IMPORTS
    )
    results.append(check_manifest())
    results.extend(check_manifest_targets(require_gui=require_gui))
    return results


def print_report(results: Sequence[CheckResult]) -> None:
    """Print a compact report suitable for terminals and support requests."""

    print("DAPPLE installation doctor")
    print("==========================")
    for result in results:
        if result.ok:
            state = "PASS"
        elif result.required:
            state = "FAIL"
        else:
            state = "WARN"
        print(f"[{state}] {result.name}: {result.detail}")

    failures = sum(not result.ok and result.required for result in results)
    warnings = sum(not result.ok and not result.required for result in results)
    print("")
    print(f"Summary: {failures} required failure(s), {warnings} optional warning(s).")
    if failures:
        print("Repair the failed items, then run dapple-doctor again.")
    else:
        print("DAPPLE is ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dapple-doctor",
        description=(
            "Check Python support, required imports, the Qt binding, and napari "
            "plugin/command discovery."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Treat Qt/widget import failures as optional warnings. Core imports and "
            "the npe2 manifest are still required."
        ),
    )
    args = parser.parse_args(argv)

    results = run_checks(require_gui=not args.headless)
    print_report(results)
    return 1 if any(not result.ok and result.required for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
