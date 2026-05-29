"""
Tests for Track B — DependencyManager hardening.

Covers pip-failure categorization, requirements.txt installation, the
install-confirmation hook, and cumulative install tracking. A FakeExecutor
stands in for DockerExecutor so nothing here needs Docker or a real pip.
"""

import pytest

from agent.dependency_manager import DependencyManager, InstallResult


# --------------------------------------------------------------------------- #
# Fake executor                                                                 #
# --------------------------------------------------------------------------- #

class FakeExecutor:
    """Minimal DockerExecutor stand-in.

    `script` maps a package name (or "-r:<path>" for requirements files) to a
    (ok, stdout, stderr) triple. Unknown packages succeed by default.
    """

    def __init__(self, script=None, persistent=True):
        self.persistent = persistent
        self.script = script or {}
        self.install_calls = []
        self.req_calls = []

    def install_packages_detailed(self, packages):
        self.install_calls.append(list(packages))
        # One package per call in DependencyManager's loop.
        name = packages[0] if packages else ""
        return self.script.get(name, (True, "Successfully installed", ""))

    def install_requirements_file_detailed(self, relative_path="requirements.txt"):
        self.req_calls.append(relative_path)
        return self.script.get(f"-r:{relative_path}", (True, "ok", ""))


# --------------------------------------------------------------------------- #
# Failure categorization                                                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stderr,expected", [
    ("ERROR: Could not find a version that satisfies the requirement foo\n"
     "ERROR: No matching distribution found for foo", "not_found"),
    ("WARNING: Retrying ... Temporary failure in name resolution\n"
     "Could not fetch URL https://pypi.org/simple/foo/", "network"),
    ("ERROR: Could not install packages due to an OSError: "
     "[Errno 13] Permission denied: '/usr/lib/python3'", "permission"),
    ("Building wheel for numpy (pyproject.toml) ... error\n"
     "error: command 'gcc' failed with exit status 1", "build_error"),
    ("something unexpected happened", "unknown"),
    ("", "unknown"),
])
def test_classify_pip_failure(stderr, expected):
    assert DependencyManager.classify_pip_failure(stderr) == expected


def test_install_failure_sets_categorized_reason():
    dm = DependencyManager(FakeExecutor(script={
        "foo": (False, "", "ERROR: No matching distribution found for foo"),
    }))
    result = dm.install_packages(["foo"])
    assert result.success is False
    assert result.failure_reason == "not_found"
    assert "foo" in result.stderr or "distribution" in result.stderr


def test_install_network_failure_reason():
    dm = DependencyManager(FakeExecutor(script={
        "bar": (False, "", "Temporary failure in name resolution"),
    }))
    result = dm.install_packages(["bar"])
    assert result.failure_reason == "network"


def test_not_persistent_still_reports_not_persistent():
    dm = DependencyManager(FakeExecutor(persistent=False))
    result = dm.install_packages(["foo"])
    assert result.success is False
    assert result.failure_reason == "not_persistent"


# --------------------------------------------------------------------------- #
# requirements.txt support                                                      #
# --------------------------------------------------------------------------- #

def test_install_from_requirements_success():
    ex = FakeExecutor()
    dm = DependencyManager(ex)
    result = dm.install_from_requirements("requirements.txt")
    assert result.success is True
    assert ex.req_calls == ["requirements.txt"]


def test_install_from_requirements_categorizes_failure():
    dm = DependencyManager(FakeExecutor(script={
        "-r:requirements.txt": (False, "", "No matching distribution found for ghost"),
    }))
    result = dm.install_from_requirements()
    assert result.success is False
    assert result.failure_reason == "not_found"


def test_install_from_requirements_requires_persistent():
    dm = DependencyManager(FakeExecutor(persistent=False))
    result = dm.install_from_requirements()
    assert result.success is False
    assert result.failure_reason == "not_persistent"


# --------------------------------------------------------------------------- #
# Confirmation UX                                                               #
# --------------------------------------------------------------------------- #

def test_confirm_decline_skips_install():
    ex = FakeExecutor()
    dm = DependencyManager(ex, ask_before_install=True, confirm_fn=lambda pkgs: False)
    result = dm.install_packages(["requests"])
    assert result.success is False
    assert result.failure_reason == "user_declined"
    assert ex.install_calls == []          # never touched the executor


def test_confirm_accept_proceeds():
    ex = FakeExecutor()
    dm = DependencyManager(ex, ask_before_install=True, confirm_fn=lambda pkgs: True)
    result = dm.install_packages(["requests"])
    assert result.success is True
    assert ex.install_calls  # install happened


def test_no_confirmation_by_default():
    ex = FakeExecutor()
    dm = DependencyManager(ex)              # ask_before_install defaults False
    result = dm.install_packages(["requests"])
    assert result.success is True
    assert ex.install_calls


# --------------------------------------------------------------------------- #
# Cumulative install tracking (persist-to-memory feed)                          #
# --------------------------------------------------------------------------- #

def test_installed_packages_accumulates_across_calls():
    ex = FakeExecutor()
    dm = DependencyManager(ex)
    dm.install_packages(["requests"])
    dm.install_packages(["pyyaml"])        # maps to PyYAML
    names = {n.lower() for n in dm.installed_packages}
    assert "requests" in names
    assert "pyyaml" in names


def test_installed_packages_excludes_failures():
    dm = DependencyManager(FakeExecutor(script={
        "ghost": (False, "", "No matching distribution found for ghost"),
    }))
    dm.install_packages(["requests"])
    dm.install_packages(["ghost"])
    names = {n.lower() for n in dm.installed_packages}
    assert "requests" in names
    assert "ghost" not in names


def test_reset_attempt_count_clears_installed():
    dm = DependencyManager(FakeExecutor())
    dm.install_packages(["requests"])
    dm.reset_attempt_count()
    assert dm.installed_packages == []
