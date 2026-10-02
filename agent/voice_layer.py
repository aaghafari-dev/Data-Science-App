"""Module duty: Voice layer.

This module provides the implementation used by Data Science Studio Pro for its named component and preserves evidence-bound, testable application behaviour.
"""

from __future__ import annotations

import re
from typing import Iterable


class VoiceCommandLayer:
    """Optional hands-free command layer modeled on the attached Master Agent Pro reference."""
    INTENTS = {
        "approve": ["approve", "yes", "okay", "continue", "approved"],
        "reject": ["reject", "no", "redo", "try again", "repeat"],
        "pause": ["pause", "stop", "save and quit"],
        "abort": ["abort", "cancel", "quit"],
        "ml": ["ml", "machine learning"],
        "dl": ["dl", "deep learning"],
        "both": ["both", "compete", "ml and dl", "both ml and dl"],
        "reread": ["read again", "say that again", "what", "pardon", "repeat that"],
        "help": ["help", "options"],
    }

    @staticmethod
    def normalize(text: str) -> str:
        """Perform the normalize operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower().replace("-", " ")).strip()

    @classmethod
    def resolve(cls, text: str, allowed: Iterable[str] | None = None) -> str | None:
        """Perform the resolve operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        text = cls.normalize(text)
        allowed = set(allowed or cls.INTENTS)
        hits = []
        for intent, phrases in cls.INTENTS.items():
            if intent not in allowed:
                continue
            if any(p in text for p in phrases):
                hits.append(intent)
        if len(set(hits)) != 1:
            return None
        return hits[0]

    def listen_once(self, prompt: str = "") -> str:
        """Perform the listen once operation for this component.

The function keeps inputs explicit, avoids hidden global mutation where practical, and returns evidence or application state required by its caller.
"""
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            if prompt:
                try:
                    import pyttsx3
                    engine = pyttsx3.init(); engine.say(prompt); engine.runAndWait()
                except Exception:
                    pass
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=.5)
                audio = r.listen(source, timeout=8, phrase_time_limit=10)
            return r.recognize_google(audio)
        except Exception:
            return ""
