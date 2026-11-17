"""Tests for the Settings window's wiring (no window is opened: the build machines have no display)."""
import re
import unittest
from pathlib import Path

from program import app_window, settings_window

ROOT = Path(__file__).resolve().parent.parent


class Settings(unittest.TestCase):
    def test_every_part_of_the_app_has_one_tab(self):
        keys = [key for key, _, _ in settings_window.TABS]
        self.assertEqual(keys, ["general", "account", "sites", "vision", "addons", "training"])
        self.assertEqual(len({title for _, title, _ in settings_window.TABS}), len(keys))

    def test_buttons_around_the_app_open_tabs_that_exist(self):
        source = (ROOT / "program" / "app_window.py").read_text(encoding="utf-8")
        asked = set(re.findall(r'open_settings\("(\w+)"\)', source))
        self.assertEqual(asked, {"account", "sites", "vision", "addons"})
        self.assertLessEqual(asked, {key for key, _, _ in settings_window.TABS})

    def test_the_separate_settings_windows_are_gone(self):
        for name in ("SiteLoginDialog", "AccountDialog", "ApiKeyDialog"):
            self.assertFalse(hasattr(app_window, name), name)
        for path in list((ROOT / "program").rglob("*.py")) + [ROOT / "main.py"]:
            text = path.read_text(encoding="utf-8")
            for old in ("File > Model setup", "File > ML settings", "Images, Site logins"):
                self.assertNotIn(old, text, f"{path.name} still points to {old}")


if __name__ == "__main__":
    unittest.main()
