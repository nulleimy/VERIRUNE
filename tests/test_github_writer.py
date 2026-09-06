import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from goverdocs.github_writer import (
    GitHubBranchWriteAmbiguousError,
    GitHubBranchWriteError,
    execute_github_branch_write,
)
from goverdocs.writer_boundary import issue_write_grant

HEAD = "a" * 40
OTHER = "b" * 40
CHANGE = "c" * 64
BRANCH = "feat/r13-2"
REPOSITORY = "nulleimy/OATHDO"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_sha(value: bytes) -> str:
    return hashlib.sha1(value).hexdigest()


def _operation(
    *,
    action: str = "create",
    target: str = "docs/output.md",
) -> dict[str, Any]:
    return {
        "event": "architecture_change",
        "rule_id": "DOC-EVT-011",
        "document_type": "architecture",
        "action": action,
        "target": target,
        "write_policy": "append-only" if action == "append" else "approval-required",
        "approval_required": True,
        "severity": "high",
        "priority": 80,
    }


def _gate_report(operation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "PASS",
        "evaluation_date": "2026-08-22",
        "input": {
            "digest": "d" * 64,
            "change_digest": CHANGE,
            "changed_files": ["src/example.py"],
            "repository": REPOSITORY,
            "pull_request": 80,
            "head_sha": HEAD,
        },
        "trust": {"trusted_verifiers": ["github-rest-source-v1"]},
        "policy_digests": {"decision_matrix": "e" * 64},
        "events": [],
        "obligations": [
            {
                "event": "architecture_change",
                "rule_id": "DOC-EVT-011",
                "severity": "high",
                "priority": 80,
                "required_evidence": [],
                "approval_required": True,
                "approval_roles": ["project-owner"],
                "actions": [operation],
            }
        ],
        "evidence_inputs": [],
        "approval_inputs": [],
        "validation_issues": [],
        "evidence_gaps": [],
        "rationale": ["all detected obligations are satisfied"],
    }


class FakeGitHubBranchWriter:
    def __init__(
        self,
        *,
        files: dict[str, bytes] | None = None,
        modes: dict[str, str] | None = None,
        branch: str = BRANCH,
        default_branch: str = "main",
        head_repo: str = REPOSITORY,
    ) -> None:
        initial_files = dict(files or {})
        initial_modes = {
            path: (modes or {}).get(path, "100644") for path in initial_files
        }
        self.branch = branch
        self.default_branch = default_branch
        self.head_repo = head_repo
        self.ref = HEAD
        self.pull_head = HEAD
        self.files_by_commit: dict[str, dict[str, bytes]] = {HEAD: initial_files}
        self.modes_by_commit: dict[str, dict[str, str]] = {HEAD: initial_modes}
        self.commit_trees: dict[str, str] = {HEAD: "1" * 40}
        self.tree_snapshots: dict[str, tuple[dict[str, bytes], dict[str, str]]] = {
            "1" * 40: (dict(initial_files), dict(initial_modes))
        }
        self.blobs: dict[str, bytes] = {}
        self.pending_trees: dict[str, tuple[str, str, str, str]] = {}
        self.commits: dict[str, dict[str, str]] = {}
        self.force_values: list[bool] = []
        self.advance_before_update = False
        self.break_post_ref = False
        self.break_post_content = False
        self.ref_updated = False
        self.tree_truncated = False
        self.non_directory_ancestor: str | None = None

        for content in initial_files.values():
            self.blobs[_git_sha(b"initial:" + content)] = content

    def get_pull(self, repository: str, pull_request: int) -> object:
        return {
            "state": "open",
            "merged": False,
            "head": {
                "ref": self.branch,
                "sha": self.pull_head,
                "repo": {"full_name": self.head_repo},
            },
        }

    def get_repository(self, repository: str) -> object:
        return {"default_branch": self.default_branch}

    def get_ref(self, repository: str, branch: str) -> object:
        if self.break_post_ref and self.ref_updated:
            return {"object": {"sha": OTHER}}
        return {"object": {"sha": self.ref}}

    def get_commit(self, repository: str, commit_sha: str) -> object:
        return {"tree": {"sha": self.commit_trees[commit_sha]}}

    def _tree_entries(
        self,
        files: dict[str, bytes],
        modes: dict[str, str],
    ) -> list[dict[str, str]]:
        entries: dict[str, dict[str, str]] = {}
        for path, content in files.items():
            parts = path.split("/")
            for index in range(1, len(parts)):
                ancestor = "/".join(parts[:index])
                entries.setdefault(
                    ancestor,
                    {
                        "path": ancestor,
                        "mode": "040000",
                        "type": "tree",
                        "sha": _git_sha(f"tree:{ancestor}".encode()),
                    },
                )
            blob_sha = _git_sha(b"blob:" + content)
            self.blobs[blob_sha] = content
            entries[path] = {
                "path": path,
                "mode": modes[path],
                "type": "blob",
                "sha": blob_sha,
            }
        if self.non_directory_ancestor is not None:
            entries[self.non_directory_ancestor] = {
                "path": self.non_directory_ancestor,
                "mode": "100644",
                "type": "blob",
                "sha": _git_sha(b"non-directory-ancestor"),
            }
            self.blobs[entries[self.non_directory_ancestor]["sha"]] = b"ancestor"
        return sorted(entries.values(), key=lambda item: item["path"])

    def get_tree(self, repository: str, tree_sha: str, *, recursive: bool) -> object:
        files, modes = self.tree_snapshots[tree_sha]
        return {
            "truncated": self.tree_truncated,
            "tree": self._tree_entries(files, modes),
        }

    def get_blob(self, repository: str, blob_sha: str) -> object:
        content = self.blobs[blob_sha]
        if self.break_post_content and self.ref_updated:
            content = b"tampered\n"
        return {
            "encoding": "base64",
            "content": base64.b64encode(content).decode("ascii"),
        }

    def create_blob(self, repository: str, content: bytes) -> object:
        sha = _git_sha(b"created-blob:" + content)
        self.blobs[sha] = content
        return {"sha": sha}

    def create_tree(
        self,
        repository: str,
        *,
        base_tree_sha: str,
        path: str,
        blob_sha: str,
        mode: str,
    ) -> object:
        sha = _git_sha(f"{base_tree_sha}:{path}:{blob_sha}:{mode}".encode())
        self.pending_trees[sha] = (base_tree_sha, path, blob_sha, mode)
        return {"sha": sha}

    def create_commit(
        self,
        repository: str,
        *,
        message: str,
        tree_sha: str,
        parent_sha: str,
    ) -> object:
        sha = _git_sha(f"{message}:{tree_sha}:{parent_sha}".encode())
        base_tree_sha, path, blob_sha, mode = self.pending_trees[tree_sha]
        assert base_tree_sha == self.commit_trees[parent_sha]
        files = dict(self.files_by_commit[parent_sha])
        modes = dict(self.modes_by_commit[parent_sha])
        files[path] = self.blobs[blob_sha]
        modes[path] = mode
        self.files_by_commit[sha] = files
        self.modes_by_commit[sha] = modes
        self.commit_trees[sha] = tree_sha
        self.tree_snapshots[tree_sha] = (dict(files), dict(modes))
        self.commits[sha] = {"parent": parent_sha, "tree": tree_sha}
        return {"sha": sha}

    def update_ref(
        self,
        repository: str,
        *,
        branch: str,
        new_sha: str,
        force: bool,
    ) -> object:
        self.force_values.append(force)
        if self.advance_before_update:
            self.ref = OTHER
            self.pull_head = OTHER
        parent = self.commits[new_sha]["parent"]
        if force or self.ref != parent:
            raise RuntimeError("non-fast-forward")
        self.ref = new_sha
        self.pull_head = new_sha
        self.ref_updated = True
        return {"object": {"sha": new_sha}}


def _execute(
    writer: FakeGitHubBranchWriter,
    operation: dict[str, Any],
    *,
    content: str,
    expected_before_sha256: str | None = None,
    execution_operation: dict[str, Any] | None = None,
    branch: str = BRANCH,
    head_sha: str = HEAD,
    repository: str = REPOSITORY,
    pull_request: int = 80,
) -> dict[str, Any]:
    report = _gate_report(operation)
    grant = issue_write_grant(report)
    return execute_github_branch_write(
        writer,
        grant,
        report,
        repository=repository,
        pull_request=pull_request,
        head_sha=head_sha,
        change_digest=CHANGE,
        branch=branch,
        operation=execution_operation or operation,
        content=content,
        expected_before_sha256=expected_before_sha256,
    )


def test_create_updates_only_pr_head_branch_and_receipt_is_schema_valid() -> None:
    writer = FakeGitHubBranchWriter()
    receipt = _execute(writer, _operation(), content="hello\n")

    assert writer.files_by_commit[writer.ref]["docs/output.md"] == b"hello\n"
    assert writer.modes_by_commit[writer.ref]["docs/output.md"] == "100644"
    assert writer.force_values == [False]
    assert receipt["branch"] == BRANCH
    assert receipt["old_head_sha"] == HEAD
    assert receipt["new_head_sha"] == writer.ref
    assert receipt["file_mode"] == "100644"
    assert receipt["pre_state"] == {"exists": False, "sha256": None}
    assert receipt["post_state"] == {
        "exists": True,
        "sha256": _sha256(b"hello\n"),
    }
    assert receipt["receipt_id"].startswith("github-branch-write-execution-v1:")

    schema = json.loads(
        Path("schemas/github-branch-write-execution-receipt.schema.json").read_text(
            encoding="utf-8"
        )
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(receipt),
        key=lambda item: list(item.path),
    )
    assert errors == []


def test_update_requires_exact_before_state_and_preserves_executable_mode() -> None:
    writer = FakeGitHubBranchWriter(
        files={"docs/output.md": b"before\n"},
        modes={"docs/output.md": "100755"},
    )
    receipt = _execute(
        writer,
        _operation(action="update"),
        content="after\n",
        expected_before_sha256=_sha256(b"before\n"),
    )

    assert writer.files_by_commit[writer.ref]["docs/output.md"] == b"after\n"
    assert writer.modes_by_commit[writer.ref]["docs/output.md"] == "100755"
    assert receipt["pre_state"]["sha256"] == _sha256(b"before\n")
    assert receipt["file_mode"] == "100755"


def test_append_preserves_existing_content() -> None:
    writer = FakeGitHubBranchWriter(files={"docs/output.md": b"first\n"})
    _execute(
        writer,
        _operation(action="append"),
        content="second\n",
        expected_before_sha256=_sha256(b"first\n"),
    )

    assert writer.files_by_commit[writer.ref]["docs/output.md"] == b"first\nsecond\n"


def test_append_rejects_result_that_is_not_utf8_text() -> None:
    writer = FakeGitHubBranchWriter(files={"docs/output.md": b"\xff"})
    with pytest.raises(GitHubBranchWriteError, match="would not produce UTF-8"):
        _execute(
            writer,
            _operation(action="append"),
            content="text\n",
            expected_before_sha256=_sha256(b"\xff"),
        )


def test_update_or_create_supports_missing_and_existing_targets() -> None:
    missing = FakeGitHubBranchWriter()
    _execute(
        missing,
        _operation(action="update_or_create"),
        content="created\n",
    )
    assert missing.files_by_commit[missing.ref]["docs/output.md"] == b"created\n"

    existing = FakeGitHubBranchWriter(files={"docs/output.md": b"old\n"})
    _execute(
        existing,
        _operation(action="update_or_create"),
        content="new\n",
        expected_before_sha256=_sha256(b"old\n"),
    )
    assert existing.files_by_commit[existing.ref]["docs/output.md"] == b"new\n"


def test_literal_main_is_forbidden_even_if_default_branch_is_elsewhere() -> None:
    writer = FakeGitHubBranchWriter(branch="main", default_branch="trunk")
    with pytest.raises(GitHubBranchWriteError, match="main or the repository default"):
        _execute(writer, _operation(), content="x\n", branch="main")


def test_arbitrary_default_branch_is_forbidden() -> None:
    writer = FakeGitHubBranchWriter(branch="release", default_branch="release")
    with pytest.raises(GitHubBranchWriteError, match="default branch"):
        _execute(writer, _operation(), content="x\n", branch="release")


def test_arbitrary_non_pr_branch_is_rejected() -> None:
    writer = FakeGitHubBranchWriter()
    with pytest.raises(GitHubBranchWriteError, match="does not match"):
        _execute(writer, _operation(), content="x\n", branch="other")


def test_fork_head_is_rejected() -> None:
    writer = FakeGitHubBranchWriter(head_repo="other/OATHDO")
    with pytest.raises(GitHubBranchWriteError, match="fork"):
        _execute(writer, _operation(), content="x\n")


def test_stale_pr_head_is_rejected() -> None:
    writer = FakeGitHubBranchWriter()
    writer.pull_head = OTHER
    with pytest.raises(GitHubBranchWriteError, match="PR head moved"):
        _execute(writer, _operation(), content="x\n")


def test_stale_branch_ref_is_rejected() -> None:
    writer = FakeGitHubBranchWriter()
    writer.ref = OTHER
    with pytest.raises(GitHubBranchWriteError, match="branch ref"):
        _execute(writer, _operation(), content="x\n")


def test_race_at_ref_update_fails_without_force() -> None:
    writer = FakeGitHubBranchWriter()
    writer.advance_before_update = True

    with pytest.raises(GitHubBranchWriteError, match="compare-and-swap"):
        _execute(writer, _operation(), content="x\n")

    assert writer.ref == OTHER
    assert writer.force_values == [False]


def test_post_ref_verification_failure_is_explicitly_ambiguous() -> None:
    writer = FakeGitHubBranchWriter()
    writer.break_post_ref = True

    with pytest.raises(GitHubBranchWriteAmbiguousError, match="post-write branch"):
        _execute(writer, _operation(), content="x\n")

    assert writer.ref_updated is True


def test_post_content_verification_failure_is_explicitly_ambiguous() -> None:
    writer = FakeGitHubBranchWriter()
    writer.break_post_content = True

    with pytest.raises(GitHubBranchWriteAmbiguousError, match="post-write content"):
        _execute(writer, _operation(), content="x\n")

    assert writer.ref_updated is True


@pytest.mark.parametrize(
    "target",
    [
        "/absolute.md",
        "../escape.md",
        "docs/../escape.md",
        "docs/*.md",
        ".git/config",
        "docs\\windows.md",
    ],
)
def test_unsafe_targets_fail_closed(target: str) -> None:
    with pytest.raises(GitHubBranchWriteError):
        _execute(FakeGitHubBranchWriter(), _operation(target=target), content="x\n")


def test_non_directory_tree_ancestor_fails_closed() -> None:
    writer = FakeGitHubBranchWriter()
    writer.non_directory_ancestor = "docs"
    with pytest.raises(GitHubBranchWriteError, match="non-directory Git tree ancestor"):
        _execute(writer, _operation(), content="x\n")


def test_truncated_recursive_tree_fails_closed() -> None:
    writer = FakeGitHubBranchWriter()
    writer.tree_truncated = True
    with pytest.raises(GitHubBranchWriteError, match="truncated"):
        _execute(writer, _operation(), content="x\n")


def test_unsupported_git_file_mode_fails_closed() -> None:
    writer = FakeGitHubBranchWriter(
        files={"docs/output.md": b"target\n"},
        modes={"docs/output.md": "120000"},
    )
    with pytest.raises(GitHubBranchWriteError, match="unsupported Git file mode"):
        _execute(
            writer,
            _operation(action="update"),
            content="after\n",
            expected_before_sha256=_sha256(b"target\n"),
        )


def test_operation_widening_and_noncanonical_shape_are_rejected() -> None:
    operation = _operation()
    widened = copy.deepcopy(operation)
    widened["target"] = "docs/other.md"
    with pytest.raises(GitHubBranchWriteError, match="outside the authorized grant scope"):
        _execute(
            FakeGitHubBranchWriter(),
            operation,
            content="x\n",
            execution_operation=widened,
        )

    extra = copy.deepcopy(operation)
    extra["extra"] = "not canonical"
    with pytest.raises(GitHubBranchWriteError, match="exact canonical operation shape"):
        _execute(
            FakeGitHubBranchWriter(),
            operation,
            content="x\n",
            execution_operation=extra,
        )


def test_wrong_pre_state_and_missing_pre_state_fail_closed() -> None:
    writer = FakeGitHubBranchWriter(files={"docs/output.md": b"before\n"})
    with pytest.raises(GitHubBranchWriteError, match="pre-state"):
        _execute(
            writer,
            _operation(action="update"),
            content="after\n",
            expected_before_sha256="f" * 64,
        )

    with pytest.raises(GitHubBranchWriteError, match="requires expected_before"):
        _execute(
            FakeGitHubBranchWriter(files={"docs/output.md": b"before\n"}),
            _operation(action="append"),
            content="after\n",
        )


def test_create_cannot_overwrite_and_update_requires_existing_target() -> None:
    with pytest.raises(GitHubBranchWriteError, match="cannot overwrite"):
        _execute(
            FakeGitHubBranchWriter(files={"docs/output.md": b"before\n"}),
            _operation(action="create"),
            content="after\n",
        )

    with pytest.raises(GitHubBranchWriteError, match="requires an existing"):
        _execute(
            FakeGitHubBranchWriter(),
            _operation(action="update"),
            content="after\n",
            expected_before_sha256="f" * 64,
        )


def test_unsupported_action_and_noop_update_fail_closed() -> None:
    with pytest.raises(GitHubBranchWriteError, match="unsupported"):
        _execute(
            FakeGitHubBranchWriter(),
            _operation(action="supersede"),
            content="x\n",
        )

    before = b"same\n"
    with pytest.raises(GitHubBranchWriteError, match="would not change"):
        _execute(
            FakeGitHubBranchWriter(files={"docs/output.md": before}),
            _operation(action="update"),
            content="same\n",
            expected_before_sha256=_sha256(before),
        )


def test_unsafe_repository_and_boolean_pull_request_fail_closed() -> None:
    operation = _operation()
    with pytest.raises(GitHubBranchWriteError, match="safe owner/name"):
        _execute(
            FakeGitHubBranchWriter(),
            operation,
            content="x\n",
            repository="nulleimy/OATHDO?x=1",
        )

    with pytest.raises(GitHubBranchWriteError, match="positive integer"):
        _execute(
            FakeGitHubBranchWriter(),
            operation,
            content="x\n",
            pull_request=True,
        )
