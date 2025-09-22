"""Simple SMTP email helper for KidBank notifications."""

from __future__ import annotations

from email.message import EmailMessage
from typing import List, Optional, Sequence


class EmailClient:
    """Very small wrapper around :mod:`smtplib` with test-friendly fallbacks."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self._outbox: List[EmailMessage] = []

    def build_message(self, subject: str, body: str, *, sender: str, recipients: Sequence[str]) -> EmailMessage:
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = sender
        message["To"] = ", ".join(recipients)
        message.set_content(body)
        return message

    def send(self, message: EmailMessage) -> None:
        try:
            import smtplib

            if self.use_tls:
                with smtplib.SMTP(self.host, self.port, timeout=5) as smtp:
                    smtp.starttls()
                    if self.username and self.password:
                        smtp.login(self.username, self.password)
                    smtp.send_message(message)
            else:
                with smtplib.SMTP(self.host, self.port, timeout=5) as smtp:
                    if self.username and self.password:
                        smtp.login(self.username, self.password)
                    smtp.send_message(message)
        except Exception:  # pragma: no cover - network operations optional
            # Store messages locally for inspection when SMTP is unavailable.
            self._outbox.append(message)

    def deliveries(self) -> Sequence[EmailMessage]:
        return tuple(self._outbox)


__all__ = ["EmailClient"]
