from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .writer_boundary import WriteGrantError, authorize_operation


class GitHubBranchWriteError(RuntimeError):
    pass


class GitHubBranchWriteAmbiguousError(GitHubBranchWriteError):
    """The branch ref moved, but final verification could not prove the result."""


class GitHubBranchWriter(Protocol):
    def get_pull(self, repository: str, pull_request: int) -> object: ...
    def get_repository(self, repository: str) -> object: ...
    def get_ref(self, repository: str, branch: str) -> object: ...
    def get_commit(self, repository: str, commit_sha: str) -> object: ...
    def get_tree(self, repository: str, tree_sha: str, *, recursive: bool) -> object: ...
    def get_blob(self, repository: str, blob_sha: str) -> object: ...
    def create_blob(self, repository: str, content: bytes) -> object: ...
    def create_tree(
        self,
        repository: str,
        *,
        base_tree_sha: str,
        path: str,
        blob_sha: str,
        mode: str,
    ) -> object: ...
    def create_commit(
        self,
        repository: str,
        *,
        message: str,
        tree_sha: str,
        parent_sha: str,
    ) -> object: ...
    def update_ref(
        self,
        repository: str,
        *,
        branch: str,
        new_sha: str,
        force: bool,
    ) -> object: ...


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SUPPORTED_ACTIONS = {"create", "update", "append", "update_or_create"}
_WRITABLE_FILE_MODES = {"100644", "100755"}
_WILDCARD_CHARS = frozenset("*?[]")


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bytes_digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dict(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GitHubBranchWriteError(f"{context} must be a JSON object")
    return value


def _required_str(value: dict[str, Any], key: str, context: str) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw:
        raise GitHubBranchWriteError(f"{context}.{key} must be a non-empty string")
    return raw


def _git_sha(value: object, context: str) -> str:
    if not isinstance(value, str) or _GIT_SHA.fullmatch(value) is None:
        raise GitHubBranchWriteError(f"{context} must be a lowercase 40-character git SHA")
    return value


def _validate_repository(repository: str) -> None:
    if _REPOSITORY.fullmatch(repository) is None:
        raise GitHubBranchWriteError("repository must use a safe owner/name form")


def _validate_pull_request(pull_request: int) -> None:
    if not isinstance(pull_request, int) or isinstance(pull_request, bool) or pull_request < 1:
        raise GitHubBranchWriteError("pull_request must be a positive integer")


def _target_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GitHubBranchWriteError("authorized target must be a non-empty string")
    target = value.strip()
    if "\\" in target:
        raise GitHubBranchWriteError("authorized target must use POSIX separators")
    if target.startswith("/") or any(character in target for character in _WILDCARD_CHARS):
        raise GitHubBranchWriteError("authorized target must be a concrete relative path")
    parts = target.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise GitHubBranchWriteError("authorized target contains an unsafe path segment")
    if ".git" in parts:
        raise GitHubBranchWriteError("writes to .git control paths are forbidden")
    return target


def _expected_digest(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise GitHubBranchWriteError(
                "existing target requires expected_before_sha256 pre-state binding"
            )
        return None
    if _SHA256.fullmatch(value) is None:
        raise GitHubBranchWriteError(
            "expected_before_sha256 must be a lowercase SHA-256 digest"
        )
    return value


def _decode_blob(payload: object, *, context: str) -> bytes:
    item = _dict(payload, context)
    encoding = item.get("encoding")
    content = item.get("content")
    if encoding != "base64" or not isinstance(content, str):
        raise GitHubBranchWriteError(f"{context} is not base64 Git blob content")
    try:
        return base64.b64decode(content, validate=False)
    except (ValueError, TypeError) as exc:
        raise GitHubBranchWriteError(f"{context} contains invalid base64") from exc


def _pull_binding(
    writer: GitHubBranchWriter,
    *,
    repository: str,
    pull_request: int,
    branch: str,
    expected_head_sha: str,
) -> None:
    pull = _dict(writer.get_pull(repository, pull_request), "GitHub pull request")
    if str(pull.get("state") or "") != "open" or bool(pull.get("merged", False)):
        raise GitHubBranchWriteError("grant-bound pull request must be open and unmerged")

    head = _dict(pull.get("head"), "GitHub pull request head")
    head_repo = _dict(head.get("repo"), "GitHub pull request head.repo")
    head_repo_name = _required_str(head_repo, "full_name", "GitHub pull request head.repo")
    head_ref = _required_str(head, "ref", "GitHub pull request head")
    head_sha = _git_sha(head.get("sha"), "GitHub pull request head.sha")

    if head_repo_name != repository:
        raise GitHubBranchWriteError("fork pull request heads are not writable in R13.2")
    if head_ref != branch:
        raise GitHubBranchWriteError(
            "requested branch does not match the grant-bound PR head branch"
        )
    if head_sha != expected_head_sha:
        raise GitHubBranchWriteError("grant-bound PR head moved after authorization")


def _ref_sha(payload: object) -> str:
    ref = _dict(payload, "GitHub branch ref")
    obj = _dict(ref.get("object"), "GitHub branch ref.object")
    return _git_sha(obj.get("sha"), "GitHub branch ref.object.sha")


def _assert_branch_writable(
    writer: GitHubBranchWriter,
    *,
    repository: str,
    branch: str,
) -> None:
    repository_info = _dict(writer.get_repository(repository), "GitHub repository")
    default_branch = _required_str(repository_info, "default_branch", "GitHub repository")
    if branch == "main" or branch == default_branch:
        raise GitHubBranchWriteError(
            "direct writes to main or the repository default branch are forbidden"
        )


def _root_tree_sha(writer: GitHubBranchWriter, repository: str, commit_sha: str) -> str:
    commit = _dict(writer.get_commit(repository, commit_sha), "GitHub commit")
    tree = _dict(commit.get("tree"), "GitHub commit.tree")
    return _git_sha(tree.get("sha"), "GitHub commit.tree.sha")


def _tree_entry(
    writer: GitHubBranchWriter,
    *,
    repository: str,
    root_tree_sha: str,
    target: str,
) -> dict[str, Any] | None:
    payload = _dict(
        writer.get_tree(repository, root_tree_sha, recursive=True),
        "GitHub recursive tree",
    )
    if payload.get("truncated") is True:
        raise GitHubBranchWriteError("GitHub recursive tree response is truncated")

    raw_entries = payload.get("tree")
    if not isinstance(raw_entries, list):
        raise GitHubBranchWriteError("GitHub recursive tree.tree must be a JSON array")

    relevant = {target}
    parts = target.split("/")
    relevant.update("/".join(parts[:index]) for index in range(1, len(parts)))

    entries: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise GitHubBranchWriteError(
                f"GitHub recursive tree.tree[{index}] must be a JSON object"
            )
        path = raw.get("path")
        if isinstance(path, str) and path in relevant:
            if path in entries:
                raise GitHubBranchWriteError(f"GitHub recursive tree duplicates path {path}")
            entries[path] = raw

    for ancestor in sorted(relevant - {target}, key=lambda value: value.count("/")):
        entry = entries.get(ancestor)
        if entry is not None and entry.get("type") != "tree":
            raise GitHubBranchWriteError(
                "authorized target has a non-directory Git tree ancestor"
            )

    return entries.get(target)


def _target_snapshot(
    writer: GitHubBranchWriter,
    *,
    repository: str,
    root_tree_sha: str,
    target: str,
) -> tuple[bool, bytes, str | None]:
    entry = _tree_entry(
        writer,
        repository=repository,
        root_tree_sha=root_tree_sha,
        target=target,
    )
    if entry is None:
        return False, b"", None

    if entry.get("type") != "blob":
        raise GitHubBranchWriteError("authorized target exists but is not a regular Git blob")
    mode = _required_str(entry, "mode", "GitHub target tree entry")
    if mode not in _WRITABLE_FILE_MODES:
        raise GitHubBranchWriteError(
            f"authorized target has unsupported Git file mode {mode}"
        )
    blob_sha = _git_sha(entry.get("sha"), "GitHub target tree entry.sha")
    content = _decode_blob(
        writer.get_blob(repository, blob_sha),
        context="GitHub target blob",
    )
    return True, content, mode


def _verify_post_state(
    writer: GitHubBranchWriter,
    *,
    repository: str,
    branch: str,
    new_head_sha: str,
    target: str,
    expected_content: bytes,
    expected_mode: str,
) -> bytes:
    if _ref_sha(writer.get_ref(repository, branch)) != new_head_sha:
        raise GitHubBranchWriteAmbiguousError(
            f"post-write branch verification failed after ref update; expected {new_head_sha}"
        )

    root_tree_sha = _root_tree_sha(writer, repository, new_head_sha)
    existed, written, mode = _target_snapshot(
        writer,
        repository=repository,
        root_tree_sha=root_tree_sha,
        target=target,
    )
    if not existed or written != expected_content or mode != expected_mode:
        raise GitHubBranchWriteAmbiguousError(
            f"post-write content verification failed after ref update at {new_head_sha}"
        )
    return written


def execute_github_branch_write(
    writer: GitHubBranchWriter,
    grant: dict[str, Any],
    gate_report: dict[str, Any],
    *,
    repository: str,
    pull_request: int,
    head_sha: str,
    change_digest: str,
    branch: str,
    operation: dict[str, Any],
    content: str,
    expected_before_sha256: str | None = None,
) -> dict[str, Any]:
    """Execute one grant-authorized UTF-8 text mutation on an existing PR head branch."""

    _validate_repository(repository)
    _validate_pull_request(pull_request)
    head_sha = _git_sha(head_sha, "head_sha")
    if not isinstance(branch, str) or not branch.strip():
        raise GitHubBranchWriteError("branch must be a non-empty string")
    branch = branch.strip()
    if branch.startswith("refs/"):
        raise GitHubBranchWriteError("branch must be a short ref name")

    try:
        authorize_operation(
            grant,
            gate_report,
            repository=repository,
            pull_request=pull_request,
            head_sha=head_sha,
            change_digest=change_digest,
            operation=operation,
        )
    except WriteGrantError as exc:
        raise GitHubBranchWriteError(f"write grant authorization failed: {exc}") from exc

    raw_operations = grant.get("operations")
    if not isinstance(raw_operations, list) or operation not in raw_operations:
        raise GitHubBranchWriteError(
            "executor requires the exact canonical operation shape from the write grant"
        )

    action = operation.get("action")
    if not isinstance(action, str) or action not in _SUPPORTED_ACTIONS:
        raise GitHubBranchWriteError(f"unsupported GitHub branch write action: {action}")
    target = _target_path(operation.get("target"))

    if not isinstance(content, str):
        raise GitHubBranchWriteError("GitHub branch writer content must be UTF-8 text")
    try:
        payload = content.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise GitHubBranchWriteError(
            "GitHub branch writer content is not valid UTF-8 text"
        ) from exc

    _assert_branch_writable(writer, repository=repository, branch=branch)
    _pull_binding(
        writer,
        repository=repository,
        pull_request=pull_request,
        branch=branch,
        expected_head_sha=head_sha,
    )
    if _ref_sha(writer.get_ref(repository, branch)) != head_sha:
        raise GitHubBranchWriteError(
            "branch ref does not match the grant-bound expected HEAD"
        )

    base_tree_sha = _root_tree_sha(writer, repository, head_sha)
    existed, before, before_mode = _target_snapshot(
        writer,
        repository=repository,
        root_tree_sha=base_tree_sha,
        target=target,
    )
    before_digest = _bytes_digest(before) if existed else None

    if action == "create":
        if existed:
            raise GitHubBranchWriteError("create action cannot overwrite an existing target")
        if expected_before_sha256 is not None:
            raise GitHubBranchWriteError(
                "create action must not declare a pre-state digest"
            )
    elif action in {"update", "append"}:
        if not existed:
            raise GitHubBranchWriteError(f"{action} action requires an existing target")
        expected = _expected_digest(expected_before_sha256, required=True)
        if expected != before_digest:
            raise GitHubBranchWriteError(
                "target pre-state digest does not match expected_before_sha256"
            )
    else:
        expected = _expected_digest(expected_before_sha256, required=existed)
        if existed and expected != before_digest:
            raise GitHubBranchWriteError(
                "target pre-state digest does not match expected_before_sha256"
            )
        if not existed and expected is not None:
            raise GitHubBranchWriteError(
                "update_or_create on a missing target must not declare a pre-state digest"
            )

    after = before + payload if action == "append" else payload
    if existed and after == before:
        raise GitHubBranchWriteError(
            "authorized GitHub branch write would not change target state"
        )
    try:
        after.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GitHubBranchWriteError(
            "authorized GitHub branch write would not produce UTF-8 text"
        ) from exc

    file_mode = before_mode if before_mode is not None else "100644"

    _assert_branch_writable(writer, repository=repository, branch=branch)
    _pull_binding(
        writer,
        repository=repository,
        pull_request=pull_request,
        branch=branch,
        expected_head_sha=head_sha,
    )
    if _ref_sha(writer.get_ref(repository, branch)) != head_sha:
        raise GitHubBranchWriteError("branch ref changed after pre-state verification")

    blob = _dict(writer.create_blob(repository, after), "GitHub blob")
    blob_sha = _git_sha(blob.get("sha"), "GitHub blob.sha")
    tree = _dict(
        writer.create_tree(
            repository,
            base_tree_sha=base_tree_sha,
            path=target,
            blob_sha=blob_sha,
            mode=file_mode,
        ),
        "GitHub tree",
    )
    tree_sha = _git_sha(tree.get("sha"), "GitHub tree.sha")
    operation_digest = _digest(operation)
    commit = _dict(
        writer.create_commit(
            repository,
            message=(
                f"goverdocs(writer): {action} {target}\n\n"
                f"Grant: {grant.get('grant_id')}\n"
                f"Operation: {operation_digest}"
            ),
            tree_sha=tree_sha,
            parent_sha=head_sha,
        ),
        "GitHub commit",
    )
    new_head_sha = _git_sha(commit.get("sha"), "GitHub commit.sha")
    if new_head_sha == head_sha:
        raise GitHubBranchWriteError("GitHub commit did not produce a new HEAD SHA")

    _assert_branch_writable(writer, repository=repository, branch=branch)
    _pull_binding(
        writer,
        repository=repository,
        pull_request=pull_request,
        branch=branch,
        expected_head_sha=head_sha,
    )
    if _ref_sha(writer.get_ref(repository, branch)) != head_sha:
        raise GitHubBranchWriteError(
            "branch ref changed before compare-and-swap update"
        )

    try:
        update = writer.update_ref(
            repository,
            branch=branch,
            new_sha=new_head_sha,
            force=False,
        )
    except Exception as exc:
        raise GitHubBranchWriteError(
            "GitHub branch compare-and-swap update failed; expected HEAD may be stale"
        ) from exc
    if _ref_sha(update) != new_head_sha:
        raise GitHubBranchWriteAmbiguousError(
            f"GitHub ref update returned an unexpected SHA after mutation attempt; expected {new_head_sha}"
        )

    written = _verify_post_state(
        writer,
        repository=repository,
        branch=branch,
        new_head_sha=new_head_sha,
        target=target,
        expected_content=after,
        expected_mode=file_mode,
    )

    subject = grant.get("subject")
    grant_id = grant.get("grant_id")
    if not isinstance(subject, dict) or not isinstance(grant_id, str) or not grant_id:
        raise GitHubBranchWriteAmbiguousError(
            f"branch updated to {new_head_sha} but canonical grant receipt fields are missing"
        )

    receipt_payload: dict[str, Any] = {
        "schema_version": 1,
        "executor": "github-pr-head-v1",
        "grant_id": grant_id,
        "subject": subject,
        "branch": branch,
        "old_head_sha": head_sha,
        "new_head_sha": new_head_sha,
        "operation_digest": operation_digest,
        "action": action,
        "target": target,
        "file_mode": file_mode,
        "pre_state": {"exists": existed, "sha256": before_digest},
        "payload_sha256": _bytes_digest(payload),
        "payload_bytes": len(payload),
        "post_state": {"exists": True, "sha256": _bytes_digest(written)},
        "result_bytes": len(written),
    }
    return {
        "receipt_id": f"github-branch-write-execution-v1:{_digest(receipt_payload)}",
        **receipt_payload,
    }


@dataclass(frozen=True)
class GitHubBranchRESTClient:
    token: str
    api_url: str = "https://api.github.com"
    api_version: str = "2022-11-28"
    timeout: float = 15.0

    @classmethod
    def from_env(
        cls,
        token_env: str = "GITHUB_TOKEN",
        *,
        api_url: str = "https://api.github.com",
        api_version: str = "2022-11-28",
        timeout: float = 15.0,
    ) -> GitHubBranchRESTClient:
        token = os.environ.get(token_env)
        if not token:
            raise GitHubBranchWriteError(
                f"{token_env} is required for GitHub branch writes"
            )
        return cls(
            token=token,
            api_url=api_url.rstrip("/"),
            api_version=api_version,
            timeout=timeout,
        )

    def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        *,
        allow_not_found: bool = False,
    ) -> object | None:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.api_url}{path}{query}"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": self.api_version,
            "User-Agent": "goverdocs-github-branch-writer/1",
        }
        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value: object = json.loads(response.read().decode("utf-8"))
                return value
        except HTTPError as exc:
            if allow_not_found and exc.code == 404:
                return None
            raise GitHubBranchWriteError(
                f"GitHub {method} {path} failed with HTTP {exc.code}"
            ) from exc
        except URLError as exc:
            raise GitHubBranchWriteError(
                f"GitHub {method} {path} failed: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise GitHubBranchWriteError(
                f"GitHub {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _required_response(value: object | None, context: str) -> object:
        if value is None:
            raise GitHubBranchWriteError(
                f"{context} unexpectedly returned no response"
            )
        return value

    def get_pull(self, repository: str, pull_request: int) -> object:
        return self._required_response(
            self._json("GET", f"/repos/{repository}/pulls/{pull_request}"),
            "GitHub pull request",
        )

    def get_repository(self, repository: str) -> object:
        return self._required_response(
            self._json("GET", f"/repos/{repository}"),
            "GitHub repository",
        )

    def get_ref(self, repository: str, branch: str) -> object:
        encoded = quote(f"heads/{branch}", safe="/")
        return self._required_response(
            self._json("GET", f"/repos/{repository}/git/ref/{encoded}"),
            "GitHub branch ref",
        )

    def get_commit(self, repository: str, commit_sha: str) -> object:
        return self._required_response(
            self._json("GET", f"/repos/{repository}/git/commits/{commit_sha}"),
            "GitHub commit",
        )

    def get_tree(self, repository: str, tree_sha: str, *, recursive: bool) -> object:
        params = {"recursive": "1"} if recursive else None
        return self._required_response(
            self._json(
                "GET",
                f"/repos/{repository}/git/trees/{tree_sha}",
                params=params,
            ),
            "GitHub tree",
        )

    def get_blob(self, repository: str, blob_sha: str) -> object:
        return self._required_response(
            self._json("GET", f"/repos/{repository}/git/blobs/{blob_sha}"),
            "GitHub blob",
        )

    def create_blob(self, repository: str, content: bytes) -> object:
        encoded = base64.b64encode(content).decode("ascii")
        return self._required_response(
            self._json(
                "POST",
                f"/repos/{repository}/git/blobs",
                {"content": encoded, "encoding": "base64"},
            ),
            "GitHub blob creation",
        )

    def create_tree(
        self,
        repository: str,
        *,
        base_tree_sha: str,
        path: str,
        blob_sha: str,
        mode: str,
    ) -> object:
        return self._required_response(
            self._json(
                "POST",
                f"/repos/{repository}/git/trees",
                {
                    "base_tree": base_tree_sha,
                    "tree": [
                        {
                            "path": path,
                            "mode": mode,
                            "type": "blob",
                            "sha": blob_sha,
                        }
                    ],
                },
            ),
            "GitHub tree creation",
        )

    def create_commit(
        self,
        repository: str,
        *,
        message: str,
        tree_sha: str,
        parent_sha: str,
    ) -> object:
        return self._required_response(
            self._json(
                "POST",
                f"/repos/{repository}/git/commits",
                {"message": message, "tree": tree_sha, "parents": [parent_sha]},
            ),
            "GitHub commit creation",
        )

    def update_ref(
        self,
        repository: str,
        *,
        branch: str,
        new_sha: str,
        force: bool,
    ) -> object:
        encoded = quote(f"heads/{branch}", safe="/")
        return self._required_response(
            self._json(
                "PATCH",
                f"/repos/{repository}/git/refs/{encoded}",
                {"sha": new_sha, "force": force},
            ),
            "GitHub branch ref update",
        )
