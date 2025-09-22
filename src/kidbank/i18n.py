"""Internationalisation helpers for KidBank."""

from __future__ import annotations

from typing import Dict, Mapping, Optional


class Translator:
    """Store translations for short interface strings."""

    def __init__(self, default_locale: str = "en", *, translations: Optional[Mapping[str, Mapping[str, str]]] = None) -> None:
        self.default_locale = default_locale
        self._translations: Dict[str, Dict[str, str]] = {
            "en": {
                "dashboard.title": "KidBank Dashboard",
                "dashboard.pending": "Pending chores",
                "dashboard.next_allowance": "Next allowance",
            },
            "es": {
                "dashboard.title": "Panel KidBank",
                "dashboard.pending": "Tareas pendientes",
                "dashboard.next_allowance": "Próxima mesada",
            },
        }
        if translations:
            for locale, mapping in translations.items():
                self._translations.setdefault(locale, {}).update(mapping)

    def set_translation(self, locale: str, key: str, value: str) -> None:
        self._translations.setdefault(locale, {})[key] = value

    def translate(self, key: str, *, locale: Optional[str] = None) -> str:
        target_locale = locale or self.default_locale
        language = self._translations.get(target_locale) or self._translations[self.default_locale]
        return language.get(key, key)

    def available_locales(self) -> tuple[str, ...]:
        return tuple(sorted(self._translations))


__all__ = ["Translator"]
