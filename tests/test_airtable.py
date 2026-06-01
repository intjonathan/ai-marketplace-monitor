"""Tests for the Airtable notification backend.

These exercise the real ``pyairtable`` library — `MockAirtable` intercepts
at the HTTP layer so real ``Api`` / ``Base`` / ``Table`` types and method
signatures are used. ``Base.schema`` / ``Base.create_table`` aren't in
``MockAirtable.mocked``, so we patch them on the real class with
``autospec=True`` to catch signature drift.
"""

from typing import TYPE_CHECKING, Any, Iterator, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from pyairtable import Api  # type: ignore
from pyairtable.api.base import Base  # type: ignore
from pyairtable.testing import MockAirtable  # type: ignore

from ai_marketplace_monitor.ai import AIResponse
from ai_marketplace_monitor.airtable import (
    DEFAULT_TABLE_NAME,
    AirtableNotificationConfig,
)
from ai_marketplace_monitor.listing import Listing
from ai_marketplace_monitor.notification import NotificationStatus

if TYPE_CHECKING:
    from typing_extensions import Self


class _FakeTableSpec:
    """Minimal stand-in for one entry of ``BaseSchema.tables``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeBaseSchema:
    def __init__(self, table_names: List[str]) -> None:
        self.tables = [_FakeTableSpec(n) for n in table_names]


def _make_listing(listing_id: str = "111", marketplace: str = "facebook") -> Listing:
    return Listing(
        marketplace=marketplace,
        name="widget",
        id=listing_id,
        title="A Widget",
        image="https://example.com/img.jpg",
        price="$100",
        post_url="https://www.facebook.com/marketplace/item/1234567890?foo=bar",
        location="Houston, TX",
        seller="some seller",
        condition="New",
        description="a nice widget",
    )


class TestAirtableConfigValidation:
    def test_table_name_defaults(self: "Self") -> None:
        cfg = AirtableNotificationConfig(
            name="atb",
            airtable_token="patABCDEFG",
            airtable_base_id="appABCDEFG1234567",
        )
        assert cfg.airtable_table_name == DEFAULT_TABLE_NAME

    def test_table_name_custom(self: "Self") -> None:
        cfg = AirtableNotificationConfig(
            name="atb",
            airtable_token="patABCDEFG",
            airtable_base_id="appABCDEFG1234567",
            airtable_table_name="My Table",
        )
        assert cfg.airtable_table_name == "My Table"

    def test_base_id_must_start_with_app(self: "Self") -> None:
        with pytest.raises(ValueError, match="must start with 'app'"):
            AirtableNotificationConfig(
                name="atb",
                airtable_token="patABCDEFG",
                airtable_base_id="bseWRONG12345",
            )

    def test_empty_token_rejected(self: "Self") -> None:
        with pytest.raises(ValueError, match="airtable_token"):
            AirtableNotificationConfig(
                name="atb",
                airtable_token="   ",
                airtable_base_id="appABCDEFG1234567",
            )

    def test_empty_table_name_rejected(self: "Self") -> None:
        with pytest.raises(ValueError, match="airtable_table_name"):
            AirtableNotificationConfig(
                name="atb",
                airtable_token="patABCDEFG",
                airtable_base_id="appABCDEFG1234567",
                airtable_table_name="   ",
            )

    def test_missing_credentials_skips_notify(self: "Self") -> None:
        cfg = AirtableNotificationConfig(name="atb")
        result = cfg.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.NOT_NOTIFIED],
        )
        assert result is False


class TestParsePrice:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("$100", 100.0),
            ("$1,200.50", 1200.50),
            ("€500", 500.0),
            ("1500", 1500.0),
            ("Free", 0.0),
            ("**unspecified**", None),
            ("", None),
            (None, None),
            ("not a price", None),
        ],
    )
    def test_parse_variants(self: "Self", raw: str, expected: float | None) -> None:
        assert AirtableNotificationConfig._parse_price(raw) == expected


class TestNotifyFlow:
    @pytest.fixture
    def config(self: "Self") -> AirtableNotificationConfig:
        AirtableNotificationConfig._table_verified.clear()
        return AirtableNotificationConfig(
            name="atb",
            airtable_token="patTEST",
            airtable_base_id="appTEST1234567890",
            max_retries=1,
            retry_delay=0,
        )

    @pytest.fixture
    def airtable_env(
        self: "Self", config: AirtableNotificationConfig
    ) -> Iterator[Tuple[MockAirtable, Any, MagicMock, MagicMock]]:
        """Yield (mock_airtable, schema_patch_setter).

        Sets up MockAirtable (covers batch_upsert + all) and patches
        Base.schema / Base.create_table with autospec=True. The yielded
        helper lets each test pick which tables `schema()` reports.
        """
        existing_tables: List[str] = []
        with (
            MockAirtable() as m,
            patch.object(Base, "schema", autospec=True) as schema_mock,
            patch.object(Base, "create_table", autospec=True) as create_mock,
        ):
            schema_mock.side_effect = lambda self: _FakeBaseSchema(existing_tables)

            def set_tables(names: List[str]) -> None:
                existing_tables.clear()
                existing_tables.extend(names)

            yield m, set_tables, schema_mock, create_mock

    def test_notify_creates_table_when_missing(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, _schema_mock, create_mock = airtable_env
        set_tables([])  # no tables exist

        ok = config.notify(
            [_make_listing()],
            [AIResponse(score=5, comment="great")],
            [NotificationStatus.NOT_NOTIFIED],
        )
        assert ok is True
        # create_table called with autospec — wrong signature would have raised
        create_mock.assert_called_once()
        call_args = create_mock.call_args
        # autospec includes `self` as the first arg
        assert call_args.args[1] == DEFAULT_TABLE_NAME
        assert isinstance(call_args.args[2], list)

        # Verify the upsert hit the right table with the right keys
        records = next(iter(m.records.values()))
        assert len(records) == 1
        # The record should carry our composite key fields
        record = next(iter(records.values()))
        assert record["fields"]["Listing ID"] == "111"
        assert record["fields"]["Marketplace"] == "facebook"

    def test_notify_skips_creation_when_table_exists(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        _m, set_tables, _schema_mock, create_mock = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        config.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.NOT_NOTIFIED],
        )
        create_mock.assert_not_called()

    def test_new_listing_record_shape(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, _schema_mock, _create_mock = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        config.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok", name="openai")],
            [NotificationStatus.NOT_NOTIFIED],
        )
        records = list(next(iter(m.records.values())).values())
        assert len(records) == 1
        fields = records[0]["fields"]
        assert fields["Listing ID"] == "111"
        assert fields["Marketplace"] == "facebook"
        assert fields["Status"] == "New"
        assert fields["Status History"].startswith("New @ ")
        assert fields["Price (Numeric)"] == 100.0
        assert "?" not in fields["URL"]
        assert fields["First Seen"] == fields["Last Updated"]
        assert fields["AI Score"] == 4
        assert fields["AI Backend"] == "openai"

    def test_not_evaluated_ai_fields_blank(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, *_ = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        config.notify(
            [_make_listing()],
            [AIResponse(score=0, comment=AIResponse.NOT_EVALUATED)],
            [NotificationStatus.NOT_NOTIFIED],
        )
        fields = next(iter(next(iter(m.records.values())).values()))["fields"]
        assert "AI Score" not in fields
        assert fields["AI Comment"] == ""
        assert fields["AI Conclusion"] == ""

    def test_status_history_appended_on_update(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, *_ = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        # Pre-seed an existing row so the lookup formula returns history.
        table = Api(config.airtable_token).base(config.airtable_base_id).table(DEFAULT_TABLE_NAME)
        m.add_records(
            table,
            [
                {
                    "Listing ID": "111",
                    "Marketplace": "facebook",
                    "Status History": "New @ 2026-04-18 12:30 UTC",
                }
            ],
        )

        config.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.LISTING_DISCOUNTED],
        )

        # Find the upserted record (it'll be the most recent fields for the listing)
        all_records = list(next(iter(m.records.values())).values())
        # Either the seeded record was upserted in place, or a new one was added.
        matched = [
            r
            for r in all_records
            if r["fields"].get("Listing ID") == "111" and r["fields"].get("Status") == "Discounted"
        ]
        assert len(matched) == 1
        fields = matched[0]["fields"]
        assert fields["Status History"].startswith("New @ 2026-04-18 12:30 UTC\nDiscounted @ ")
        # First Seen NOT set on updates (preserved by replace=False)
        assert "First Seen" not in fields

    def test_notified_listings_skipped_without_force(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, *_ = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        ok = config.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.NOTIFIED],
        )
        assert ok is False
        assert not m.records  # nothing upserted

    def test_force_resends_notified(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, *_ = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        ok = config.notify(
            [_make_listing()],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.NOTIFIED],
            force=True,
        )
        assert ok is True
        records = list(next(iter(m.records.values())).values())
        assert len(records) == 1

    def test_image_url_omitted_when_empty(
        self: "Self",
        config: AirtableNotificationConfig,
        airtable_env: Any,
    ) -> None:
        m, set_tables, *_ = airtable_env
        set_tables([DEFAULT_TABLE_NAME])

        listing = _make_listing()
        listing.image = ""
        config.notify(
            [listing],
            [AIResponse(score=4, comment="ok")],
            [NotificationStatus.NOT_NOTIFIED],
        )
        fields = next(iter(next(iter(m.records.values())).values()))["fields"]
        assert "Image URL" not in fields


class TestImportFallback:
    """If pyairtable is genuinely missing, notify() should degrade gracefully."""

    def test_pyairtable_missing_returns_false(self: "Self") -> None:
        cfg = AirtableNotificationConfig(
            name="atb",
            airtable_token="patTEST",
            airtable_base_id="appTEST1234567890",
            max_retries=1,
            retry_delay=0,
        )

        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pyairtable":
                raise ImportError("pyairtable not installed")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            ok = cfg.notify(
                [_make_listing()],
                [AIResponse(score=4, comment="ok")],
                [NotificationStatus.NOT_NOTIFIED],
            )
        assert ok is False
