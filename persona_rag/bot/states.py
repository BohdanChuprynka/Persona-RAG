from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class AuthApproval(StatesGroup):
    waiting_for_decision = State()
    viewing_more = State()
