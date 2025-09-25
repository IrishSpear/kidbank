"""Configuration constants for the KidBank web frontend."""
from __future__ import annotations

import json
import os
import textwrap
from datetime import timedelta
from typing import Dict, Tuple

try:  # pragma: no cover - optional dependency guard
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv() -> None:  # type: ignore[override]
        """Fallback when python-dotenv is not installed."""

        return None
else:  # pragma: no cover - simple passthrough
    load_dotenv()

MOM_PIN = os.environ.get("MOM_PIN", "1022")
DAD_PIN = os.environ.get("DAD_PIN", "2097")
SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-this-session-secret")
SQLITE_FILE_NAME = os.environ.get("KIDBANK_SQLITE", "kidbank.db")
DEFAULT_PARENT_ROLES: Tuple[str, ...] = ("mom", "dad")
DEFAULT_PARENT_LABELS: Dict[str, str] = {"mom": "Mom", "dad": "Dad"}
EXTRA_PARENT_ADMINS_KEY = "parent_admins"
GLOBAL_CHORE_TYPES: Tuple[str, ...] = ("daily", "weekly", "monthly")
DEFAULT_GLOBAL_CHORE_TYPE = "daily"
PWA_CACHE_NAME = "kidbank-shell-v1"
PWA_SHELL_PATHS: Tuple[str, ...] = (
    "/",
    "/kid",
    "/admin",
    "/admin/login",
    "/manifest.webmanifest",
    "/pwa-icon.svg",
    "/offline",
)
PWA_ICON_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 128 128'>
  <defs>
    <linearGradient id='grad' x1='0' x2='1' y1='0' y2='1'>
      <stop offset='0%' stop-color='#2563eb'/>
      <stop offset='100%' stop-color='#1d4ed8'/>
    </linearGradient>
  </defs>
  <rect width='128' height='128' rx='28' ry='28' fill='url(#grad)'/>
  <text x='64' y='82' font-size='64' font-family='Roboto, Arial, sans-serif' font-weight='700' fill='#f8fafc' text-anchor='middle'>K</text>
</svg>
"""
SERVICE_WORKER_JS = (
    textwrap.dedent(
        """
        const CACHE_NAME = '__CACHE_NAME__';
        const OFFLINE_URLS = __OFFLINE_URLS__;

        self.addEventListener('install', event => {
          event.waitUntil(
            caches.open(CACHE_NAME).then(cache => cache.addAll(OFFLINE_URLS)).then(() => self.skipWaiting())
          );
        });

        self.addEventListener('activate', event => {
          event.waitUntil(
            caches.keys().then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
          );
          self.clients.claim();
        });

        async function fetchAndCache(request) {
          try {
            const response = await fetch(request);
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(request, clone));
            return response;
          } catch (error) {
            throw error;
          }
        }

        self.addEventListener('fetch', event => {
          if (event.request.method !== 'GET') {
            return;
          }
          const request = event.request;
          const url = new URL(request.url);
          if (request.mode === 'navigate') {
            event.respondWith(
              fetchAndCache(request).catch(() => caches.match(request).then(resp => resp || caches.match('/offline')))
            );
            return;
          }
          if (OFFLINE_URLS.includes(url.pathname)) {
            event.respondWith(
              caches.match(request).then(resp => resp || fetchAndCache(request))
            );
            return;
          }
          event.respondWith(
            fetchAndCache(request).catch(() => caches.match(request))
          );
        });
        """
    )
    .strip()
    .replace("__CACHE_NAME__", PWA_CACHE_NAME)
    .replace("__OFFLINE_URLS__", json.dumps(list(PWA_SHELL_PATHS)))
)

REMEMBER_COOKIE_NAME = "kid_remember"
REMEMBER_NAME_COOKIE = "kid_remember_name"
REMEMBER_COOKIE_LIFETIME = timedelta(days=30)
REMEMBER_COOKIE_MAX_AGE = int(REMEMBER_COOKIE_LIFETIME.total_seconds())

__all__ = [
    "MOM_PIN",
    "DAD_PIN",
    "SESSION_SECRET",
    "SQLITE_FILE_NAME",
    "DEFAULT_PARENT_ROLES",
    "DEFAULT_PARENT_LABELS",
    "EXTRA_PARENT_ADMINS_KEY",
    "GLOBAL_CHORE_TYPES",
    "DEFAULT_GLOBAL_CHORE_TYPE",
    "PWA_CACHE_NAME",
    "PWA_SHELL_PATHS",
    "PWA_ICON_SVG",
    "SERVICE_WORKER_JS",
    "REMEMBER_COOKIE_NAME",
    "REMEMBER_NAME_COOKIE",
    "REMEMBER_COOKIE_LIFETIME",
    "REMEMBER_COOKIE_MAX_AGE",
]

