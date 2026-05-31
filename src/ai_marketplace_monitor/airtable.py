import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from logging import Logger
from typing import Any, ClassVar, Dict, List, Set, Tuple

from .ai import AIResponse  # type: ignore
from .listing import Listing
from .notification import NotificationConfig, NotificationStatus
from .utils import hilight

DEFAULT_TABLE_NAME = "Marketplace Listings"

STATUS_LABELS: Dict[NotificationStatus, str] = {
    NotificationStatus.NOT_NOTIFIED: "New",
    NotificationStatus.LISTING_CHANGED: "Changed",
    NotificationStatus.LISTING_DISCOUNTED: "Discounted",
    NotificationStatus.EXPIRED: "Reminder",
}

# Schema sent to base.create_table() when auto-creating the table.
_TABLE_FIELD_SPEC: List[Dict[str, Any]] = [
    {"name": "Listing ID", "type": "singleLineText"},
    {"name": "Marketplace", "type": "singleLineText"},
    {"name": "Title", "type": "singleLineText"},
    {"name": "Price", "type": "singleLineText"},
    {"name": "Price (Numeric)", "type": "number", "options": {"precision": 2}},
    {"name": "Location", "type": "singleLineText"},
    {"name": "URL", "type": "url"},
    {"name": "Image URL", "type": "url"},
    {"name": "Seller", "type": "singleLineText"},
    {"name": "Condition", "type": "singleLineText"},
    {"name": "Description", "type": "multilineText"},
    {"name": "Item Name", "type": "singleLineText"},
    {"name": "AI Score", "type": "number", "options": {"precision": 0}},
    {"name": "AI Conclusion", "type": "singleLineText"},
    {"name": "AI Comment", "type": "multilineText"},
    {"name": "AI Backend", "type": "singleLineText"},
    {
        "name": "Status",
        "type": "singleSelect",
        "options": {
            "choices": [
                {"name": "New"},
                {"name": "Changed"},
                {"name": "Discounted"},
                {"name": "Reminder"},
            ]
        },
    },
    {"name": "Status History", "type": "multilineText"},
    {"name": "Content Hash", "type": "singleLineText"},
    {
        "name": "First Seen",
        "type": "dateTime",
        "options": {
            "timeZone": "utc",
            "dateFormat": {"name": "iso"},
            "timeFormat": {"name": "24hour"},
        },
    },
    {
        "name": "Last Updated",
        "type": "dateTime",
        "options": {
            "timeZone": "utc",
            "dateFormat": {"name": "iso"},
            "timeFormat": {"name": "24hour"},
        },
    },
]


@dataclass
class AirtableNotificationConfig(NotificationConfig):
    notify_method = "airtable"
    required_fields: ClassVar[List[str]] = ["airtable_token", "airtable_base_id"]

    airtable_token: str | None = None
    airtable_base_id: str | None = None
    airtable_table_name: str | None = None

    # Cached set of (base_id, table_name) tuples already verified to exist —
    # avoids hitting base.schema() on every notification.
    _table_verified: ClassVar[Set[Tuple[str, str]]] = set()

    def handle_airtable_token(self: "AirtableNotificationConfig") -> None:
        if self.airtable_token is None:
            return
        if not isinstance(self.airtable_token, str) or not self.airtable_token.strip():
            raise ValueError(
                f"Item {hilight(self.name)} airtable_token must be a non-empty string."
            )
        self.airtable_token = self.airtable_token.strip()

    def handle_airtable_base_id(self: "AirtableNotificationConfig") -> None:
        if self.airtable_base_id is None:
            return
        if not isinstance(self.airtable_base_id, str) or not self.airtable_base_id.strip():
            raise ValueError(
                f"Item {hilight(self.name)} airtable_base_id must be a non-empty string."
            )
        self.airtable_base_id = self.airtable_base_id.strip()
        if not self.airtable_base_id.startswith("app"):
            raise ValueError(f"Item {hilight(self.name)} airtable_base_id must start with 'app'.")

    def handle_airtable_table_name(self: "AirtableNotificationConfig") -> None:
        if self.airtable_table_name is None:
            self.airtable_table_name = DEFAULT_TABLE_NAME
            return
        if not isinstance(self.airtable_table_name, str) or not self.airtable_table_name.strip():
            raise ValueError(
                f"Item {hilight(self.name)} airtable_table_name must be a non-empty string."
            )
        self.airtable_table_name = self.airtable_table_name.strip()

    @staticmethod
    def _parse_price(price_str: str | None) -> float | None:
        """Extract a numeric price from a string like '$1,200', '€500', 'Free'."""
        if not price_str or not isinstance(price_str, str):
            return None
        s = price_str.strip()
        if not s or s == "**unspecified**":
            return None
        if s.lower() == "free":
            return 0.0
        matched = re.match(r"(\D*)([\d.,\s]+)", s)
        if not matched:
            return None
        numeric = matched.group(2).replace(",", "").replace(" ", "")
        try:
            return float(numeric)
        except ValueError:
            return None

    def _ensure_table_exists(
        self: "AirtableNotificationConfig", api: Any, logger: Logger | None
    ) -> Any:
        """Return the pyairtable Table object, creating the table if needed."""
        assert self.airtable_base_id is not None
        assert self.airtable_table_name is not None
        base = api.base(self.airtable_base_id)
        cache_key = (self.airtable_base_id, self.airtable_table_name)

        if cache_key not in self._table_verified:
            schema = base.schema()
            existing = {t.name for t in schema.tables}
            if self.airtable_table_name not in existing:
                if logger:
                    logger.info(
                        f"""{hilight("[Airtable]")} Creating table {hilight(self.airtable_table_name)} in base {self.airtable_base_id}."""
                    )
                base.create_table(self.airtable_table_name, _TABLE_FIELD_SPEC)
            self._table_verified.add(cache_key)

        return base.table(self.airtable_table_name)

    @staticmethod
    def _fetch_existing_histories(
        table: Any, listings: List[Listing]
    ) -> Dict[Tuple[str, str], str]:
        """Return {(listing_id, marketplace): existing_status_history} for updates."""
        if not listings:
            return {}
        # Airtable formula: OR(AND({Listing ID}='id1', {Marketplace}='fb'), ...)
        terms = []
        for listing in listings:
            lid = str(listing.id).replace("'", r"\'")
            mp = str(listing.marketplace).replace("'", r"\'")
            terms.append(f"AND({{Listing ID}}='{lid}', {{Marketplace}}='{mp}')")
        formula = f"OR({', '.join(terms)})" if len(terms) > 1 else terms[0]
        rows = table.all(formula=formula, fields=["Listing ID", "Marketplace", "Status History"])
        result: Dict[Tuple[str, str], str] = {}
        for row in rows:
            f = row.get("fields", {})
            key = (f.get("Listing ID", ""), f.get("Marketplace", ""))
            result[key] = f.get("Status History", "") or ""
        return result

    def _build_record(
        self: "AirtableNotificationConfig",
        listing: Listing,
        rating: AIResponse,
        status: NotificationStatus,
        now: datetime,
        existing_history: str,
    ) -> Dict[str, Any]:
        now_iso = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        now_label = now.strftime("%Y-%m-%d %H:%M UTC")
        status_label = STATUS_LABELS.get(status, "New")

        if status == NotificationStatus.NOT_NOTIFIED or not existing_history:
            history = f"{status_label} @ {now_label}"
        else:
            history = f"{existing_history}\n{status_label} @ {now_label}"

        url = listing.post_url.split("?")[0] if listing.post_url else None
        image_url = listing.image if listing.image else None

        ai_evaluated = rating.comment != AIResponse.NOT_EVALUATED

        fields: Dict[str, Any] = {
            "Listing ID": str(listing.id),
            "Marketplace": listing.marketplace,
            "Title": listing.title,
            "Price": listing.price,
            "Location": listing.location,
            "Seller": listing.seller,
            "Condition": listing.condition,
            "Description": listing.description,
            "Item Name": listing.name,
            "AI Conclusion": rating.conclusion if ai_evaluated else "",
            "AI Comment": rating.comment if ai_evaluated else "",
            "AI Backend": rating.name if ai_evaluated else "",
            "Status": status_label,
            "Status History": history,
            "Content Hash": listing.hash,
            "Last Updated": now_iso,
        }

        price_num = self._parse_price(listing.price)
        if price_num is not None:
            fields["Price (Numeric)"] = price_num
        if url:
            fields["URL"] = url
        if image_url:
            fields["Image URL"] = image_url
        if ai_evaluated:
            fields["AI Score"] = rating.score

        # First Seen is only set on new records; for updates, Airtable's
        # batch_upsert with replace=False preserves the existing value.
        if status == NotificationStatus.NOT_NOTIFIED:
            fields["First Seen"] = now_iso

        return {"fields": fields}

    def notify(
        self: "AirtableNotificationConfig",
        listings: List[Listing],
        ratings: List[AIResponse],
        notification_status: List[NotificationStatus],
        force: bool = False,
        logger: Logger | None = None,
    ) -> bool:
        if not self._has_required_fields():
            if logger:
                logger.debug(
                    f"Missing required fields {', '.join(self.required_fields)}. No {self.notify_method} notification sent."
                )
            return False

        actionable = [
            (listing, rating, status)
            for listing, rating, status in zip(listings, ratings, notification_status)
            if force or status != NotificationStatus.NOTIFIED
        ]
        if not actionable:
            if logger:
                logger.debug("No actionable listings for Airtable notification.")
            return False

        try:
            from pyairtable import Api  # type: ignore
        except ImportError:
            if logger:
                logger.error(
                    "pyairtable is required for Airtable notifications. "
                    "Install with: pip install ai-marketplace-monitor[airtable]"
                )
            return False

        for attempt in range(self.max_retries):
            try:
                api = Api(self.airtable_token)
                table = self._ensure_table_exists(api, logger)

                update_listings = [
                    listing
                    for (listing, _rating, status) in actionable
                    if status != NotificationStatus.NOT_NOTIFIED
                ]
                histories = (
                    self._fetch_existing_histories(table, update_listings)
                    if update_listings
                    else {}
                )

                now = datetime.now(timezone.utc)
                records = [
                    self._build_record(
                        listing,
                        rating,
                        status,
                        now,
                        histories.get((str(listing.id), listing.marketplace), ""),
                    )
                    for (listing, rating, status) in actionable
                ]

                table.batch_upsert(
                    records,
                    key_fields=["Listing ID", "Marketplace"],
                    replace=False,
                )

                if logger:
                    logger.info(
                        f"""{hilight("[Notify]", "succ")} Upserted {len(records)} listing(s) to Airtable table {hilight(self.airtable_table_name or "")}."""
                    )
                return True
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if logger:
                    logger.debug(
                        f"""{hilight("[Notify]", "fail")} Attempt {attempt + 1} failed: {e}"""
                    )
                if attempt < self.max_retries - 1:
                    if logger:
                        logger.debug(
                            f"""{hilight("[Notify]", "fail")} Retrying in {self.retry_delay} seconds..."""
                        )
                    time.sleep(self.retry_delay)
                else:
                    if logger:
                        logger.error(
                            f"""{hilight("[Notify]", "fail")} Max retries reached. Failed to upsert to Airtable for {self.name}."""
                        )
                    return False
        return False
