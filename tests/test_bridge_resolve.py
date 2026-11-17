"""Unit tests for alfred.bridge.resolve — contact -> vault-entity matching.

Uses tmp_path + the real alfred.core.vault_ops primitives (vault_create /
vault_search / vault_read) rather than mocks: this exercises the actual
grep-then-confirm resolution path against real markdown+frontmatter files,
matching this repo's existing vault_ops test convention (see
tests/test_vault_ops.py) and staying entirely off the real vault on disk.
"""
from __future__ import annotations

import pytest

from alfred.bridge.resolve import resolve_contact_to_entity
from alfred.core.vault_ops import vault_create


@pytest.fixture
def vault(tmp_path):
    v = tmp_path / "vault-main"
    v.mkdir()
    return v


def _make_person(vault, name, *, email=None, body=None, **extra):
    fields = {"email": email, "status": "active"}
    fields.update(extra)
    vault_create(vault, "person", name, set_fields=fields, body=body)


def test_exact_email_match(vault):
    _make_person(vault, "David Szabo-Stuban", email="dstuban@example.com")

    match = resolve_contact_to_entity(vault, email="dstuban@example.com")

    assert match is not None
    assert match.rel_path == "person/David Szabo-Stuban.md"
    assert match.name == "David Szabo-Stuban"
    assert match.confidence == 1.0
    assert match.reason == "email_exact"


def test_case_insensitive_email_match(vault):
    _make_person(vault, "David Szabo-Stuban", email="DStuban@Example.COM")

    match = resolve_contact_to_entity(vault, email="dstuban@example.com")

    assert match is not None
    assert match.rel_path == "person/David Szabo-Stuban.md"


def test_case_insensitive_match_from_uppercase_crm_input(vault):
    _make_person(vault, "David Szabo-Stuban", email="dstuban@example.com")

    match = resolve_contact_to_entity(vault, email="DSTUBAN@EXAMPLE.COM")

    assert match is not None
    assert match.rel_path == "person/David Szabo-Stuban.md"


def test_no_match_returns_none(vault):
    _make_person(vault, "David Szabo-Stuban", email="dstuban@example.com")

    match = resolve_contact_to_entity(vault, email="nobody@nowhere.com")

    assert match is None


def test_contact_with_no_email_returns_none(vault):
    """Even with a name that exactly matches a vault record, no email means
    no match — resolve() must never fall back to name-only matching."""
    _make_person(vault, "David Szabo-Stuban", email="dstuban@example.com")

    match = resolve_contact_to_entity(vault, email=None, name="David Szabo-Stuban")

    assert match is None


def test_common_first_name_alone_does_not_false_positive(vault):
    """Multiple 'John' person records must never be guessed between when the
    CRM contact's email doesn't match either of them."""
    _make_person(vault, "John Smith", email="john.smith@acme.com")
    _make_person(vault, "John Doe", email="john.doe@other.com")

    match = resolve_contact_to_entity(vault, email="john@crm-contact.example.com", name="John")

    assert match is None


def test_ambiguous_duplicate_email_returns_none(vault):
    """Two person records sharing the same email is a vault data-integrity
    problem, not something resolve() should arbitrate — bias toward no
    match rather than picking one candidate over the other."""
    _make_person(vault, "Person One", email="shared@example.com")
    _make_person(vault, "Person Two", email="shared@example.com")

    match = resolve_contact_to_entity(vault, email="shared@example.com")

    assert match is None


def test_email_mentioned_only_in_body_is_not_a_false_positive(vault):
    """The grep prefilter can hit an email mentioned in body prose, not the
    frontmatter `email:` key — the frontmatter-field confirmation step must
    reject that as a real match."""
    _make_person(
        vault,
        "Someone Else",
        email="someone@example.com",
        body="Mentioned meeting notes for dstuban@example.com in passing.\n",
    )

    match = resolve_contact_to_entity(vault, email="dstuban@example.com")

    assert match is None


def test_org_records_are_not_matched(vault):
    """resolve_contact_to_entity only ever scans person/*.md — org records
    (which don't carry an email field per the observed schema) must never
    surface as a match."""
    from alfred.core.vault_ops import vault_create as _vc

    _vc(vault, "org", "Acme Corp", set_fields={"org_type": "vendor", "status": "active"})

    match = resolve_contact_to_entity(vault, email="anything@acme.com")

    assert match is None
