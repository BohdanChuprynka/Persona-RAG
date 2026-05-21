"""Whitelist the admin (you) in the bot's SQLite users table.

The bot's auth graph treats every user — including the admin's own Telegram
account — as unknown until they appear in the SQLite ``users`` table with
``state=whitelisted``. Run this once after ingest so your first DM doesn't
trigger the admin-approval flow against yourself.

Idempotent. Safe to re-run after wiping the DB.
"""

from __future__ import annotations

from persona_rag.bot.auth import approve_user, ensure_user
from persona_rag.config import get_settings


def main() -> None:
    admin_id = get_settings().ADMIN_TELEGRAM_ID
    ensure_user(admin_id, "admin", "Admin")
    approve_user(admin_id, admin_id=admin_id)
    print(f"whitelisted admin {admin_id}")


if __name__ == "__main__":
    main()
