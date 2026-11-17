"""File > Settings: every setting of the app in one window, one tab per part of the app.

    General          theme, log, keep awake, the data folder, updates
    GBIF account     the account new GBIF downloads are requested with
    Image sites      how images are downloaded, sites that refuse downloads, saved sign-ins
    Vision model     the provider's API key, server, model and options used for measuring
    Model add-ons    the optional PyTorch add-on, SAM 2, label reader and starting weights
    Training         how your own models are trained

Each tab is a panel (a ttk.Frame) built the first time it is shown. A panel may have refresh() (called whenever
its tab is shown) and can_close() (asked before the window closes, so a running install can object).
"""
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

import config
from config import (APP_NAME, APP_VERSION, GBIF_SIGNUP_URL, THINKING_CHOICES, ai_api_key, gbif_credentials,
                    password_store_name, remove_ai_api_key, remove_gbif_account, save_ai_api_key, save_gbif_account,
                    saved_gbif_account, saved_gbif_password)
from . import browser_sign_in, image_download, updater
from .gbif_download import check_gbif_login
from .gui_widgets import (THEME_CHOICES, HelpTip, ScrollFrame, help_label, placeholder, set_title_bar, tip,
                          titled_frame)
from .keep_awake import KeepAwake
from .measure.ml_settings import TrainingPanel
from .measure.model_setup import ModelSetupPanel
from .measure.providers import PROVIDERS, describe_key, list_models, settings_from_app
from .measure.records import check_connection

PAD = 16


class Panel(ttk.Frame):
    """A settings tab. `say` shows one line of feedback at the bottom of the panel."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=PAD)
        self.app = app
        self.columnconfigure(0, weight=1)
        self.status = ttk.Label(self, style="Muted.TLabel", wraplength=640, justify="left")

    def say(self, text, kind="muted"):
        if self.winfo_exists():
            self.status.configure(text=text, foreground=self.app.palette[kind])

    def background(self, work, done):
        """Run work() off the window thread, then done(result) on it (unless the window was closed)."""
        def run():
            try:
                result = work()
            except Exception as exc:
                result = exc
            try:
                self.after(0, lambda: self.winfo_exists() and done(result))
            except (tk.TclError, RuntimeError):
                pass

        threading.Thread(target=run, daemon=True).start()


# ------------------------------------------------------------------------------------------------ General

class GeneralPanel(Panel):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        look = titled_frame(self, "Appearance")
        look.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        ttk.Label(look, text="Theme").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        theme = ttk.Combobox(look, textvariable=app.theme_var, values=list(THEME_CHOICES), state="readonly", width=12)
        theme.grid(row=0, column=1, sticky="w", pady=3)
        theme.bind("<<ComboboxSelected>>", self._theme)
        tip(ttk.Checkbutton(look, text="Show the log", variable=app.show_log_var, command=app._toggle_log),
            "The log at the bottom of the window lists what each task did. The log file keeps everything either "
            "way.").grid(row=1, column=0, columnspan=2, sticky="w", pady=3)

        work = titled_frame(self, "Long tasks")
        work.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        awake = tip(ttk.Checkbutton(work, text="Keep the computer awake while the app is open",
                                    variable=app.keep_awake_var, command=app._toggle_keep_awake),
                    "Stops the computer from going to sleep, so long downloads and measuring runs can finish. The "
                    "screen may still turn off.")
        awake.grid(row=0, column=0, sticky="w", pady=3)
        if not KeepAwake.supported():
            awake.configure(state="disabled")

        data = titled_frame(self, "Data folder",
                            "Each GBIF download is kept in its own folder inside the data folder, with its CSV "
                            "files, images and measurements. Choose another folder to keep them elsewhere, for "
                            "example on an external drive; the downloads found there are listed under Active "
                            "download.")
        data.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        data.columnconfigure(0, weight=1)
        self.data_label = ttk.Label(data, wraplength=620, justify="left")
        self.data_label.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        buttons = ttk.Frame(data)
        buttons.grid(row=1, column=0, sticky="w")
        ttk.Button(buttons, text="Choose…", command=self._choose).pack(side="left")
        self.default_button = ttk.Button(buttons, text="Use the default", command=self._default)
        self.default_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Open", command=lambda: app._open(config.DATA_DIR)).pack(side="left")

        updates = titled_frame(self, "Updates")
        updates.grid(row=3, column=0, sticky="ew")
        ttk.Checkbutton(updates, text="Check for updates when the app starts", variable=app.check_updates_var,
                        command=app._save_settings).grid(row=0, column=0, columnspan=2, sticky="w", pady=3)
        tip(ttk.Checkbutton(updates, text="Install updates without asking", variable=app.auto_update_var,
                            command=app._save_settings),
            "A new version found at launch is downloaded and installed straight away. The app checks that the new "
            "version starts before it replaces this one.").grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        row = ttk.Frame(updates)
        row.grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(row, text="Check now", command=lambda: app.check_for_updates(manual=True)).pack(side="left")
        source = "" if updater.install_location() else " (running from source: update it with git)"
        ttk.Label(row, text=f"{APP_NAME} {APP_VERSION}{source}", style="Muted.TLabel").pack(side="left", padx=10)
        self.refresh()

    def refresh(self):
        self.data_label.configure(text=str(config.DATA_DIR))
        self.default_button.configure(state="disabled" if config.DATA_DIR == config.DEFAULT_DATA_DIR else "normal")

    def _theme(self, _event=None):
        self.app._change_theme()
        self.winfo_toplevel().restyle()

    def _choose(self):
        self.app._choose_data_dir(parent=self.winfo_toplevel())
        self.refresh()

    def _default(self):
        self.app._reset_data_dir()
        self.refresh()


# ------------------------------------------------------------------------------------------------ GBIF account

class AccountPanel(Panel):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        account = saved_gbif_account()
        self.had_password = account["has_password"]
        form = titled_frame(self, "GBIF account",
                            "New GBIF downloads are requested with your GBIF account. Downloading by DOI does not "
                            "need one.\n\n" + self._where())
        form.grid(row=0, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        self.user_var = tk.StringVar(value=account["user"])
        self.email_var = tk.StringVar(value=account["email"])
        self.password_var = tk.StringVar()
        for row, (label, variable) in enumerate((("Username", self.user_var), ("Email", self.email_var),
                                                 ("Password", self.password_var))):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            entry = ttk.Entry(form, textvariable=variable, width=40, show="•" if label == "Password" else "")
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            entry.bind("<Return>", lambda _: self._save())
            if label == "Password":
                self.password_entry = entry
        self._password_prompt()
        self.show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(form, text="Show password", variable=self.show_var,
                        command=lambda: self.password_entry.configure(show="" if self.show_var.get() else "•")
                        ).grid(row=3, column=1, sticky="w")
        link = ttk.Label(form, text="Create a free GBIF account", style="Link.TLabel", cursor="hand2")
        link.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        link.bind("<Button-1>", lambda _: webbrowser.open(GBIF_SIGNUP_URL))

        buttons = ttk.Frame(self)
        buttons.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="Save", style="Accent.TButton", command=self._save).pack(side="left")
        self.test_button = ttk.Button(buttons, text="Test sign-in", command=self._test)
        self.test_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Remove account", command=self._remove).pack(side="left")
        self.status.grid(row=2, column=0, sticky="w", pady=(10, 0))
        if gbif_credentials()[3] == "environment variables":
            self.say("GBIF_ environment variables are set and are used instead of the account saved here.", "busy")

    @staticmethod
    def _where():
        store = password_store_name()
        return (f"The password is kept in {store}, not in a plain file." if store else
                "No system keychain was found, so the password is kept in a file readable only by your user.")

    def _password_prompt(self):
        if self.had_password:
            placeholder(self.password_entry, "Saved. Leave empty to keep it", self.password_var)

    def _password(self):
        return self.password_var.get() or saved_gbif_password(self.user_var.get().strip())

    def _test(self):
        user, password = self.user_var.get().strip(), self._password()
        if not user or not password:
            self.say("Enter a username and password first.", "error")
            return
        self.test_button.configure(state="disabled")
        self.say("Checking with GBIF…", "busy")

        def done(result):
            self.test_button.configure(state="normal")
            if isinstance(result, Exception):
                self.say(f"Could not reach GBIF: {result}", "error")
            else:
                self.say(result[1], "ok" if result[0] else "error")

        self.background(lambda: check_gbif_login(user, password), done)

    def _save(self):
        user, email = self.user_var.get().strip(), self.email_var.get().strip()
        if not user or not email:
            return self.say("Enter your GBIF username and email.", "error")
        if "@" not in email:
            return self.say("The email address does not look right.", "error")
        if not self.password_var.get() and not self.had_password:
            return self.say("Enter your GBIF password.", "error")
        try:
            where = save_gbif_account(user, email, self.password_var.get() or None)
        except OSError as exc:
            return self.say(f"Could not save the account: {exc}", "error")
        self.app._log(f"GBIF account saved for {user}" + (f"; the password is kept in {where}." if where else "."))
        self.had_password = True
        self.password_var.set("")
        self._password_prompt()
        self.app._refresh_account()
        self.say(f"Saved. Signed in as {user}.", "ok")

    def _remove(self):
        if not saved_gbif_account()["user"]:
            return self.say("No account is saved in this app.")
        if messagebox.askyesno("Remove account", "Remove the saved GBIF account and password?",
                               parent=self.winfo_toplevel()):
            remove_gbif_account()
            self.app._log("Saved GBIF account removed.")
            self.had_password = False
            for variable in (self.user_var, self.email_var, self.password_var):
                variable.set("")
            self.app._refresh_account()
            self.say("The saved account was removed.")


# ------------------------------------------------------------------------------------------------ Image sites

class SitesPanel(Panel):
    """How images are downloaded, the sites that refused downloads, and the sign-ins saved for them."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.done = None
        self.signing_in = False
        self.rowconfigure(1, weight=1)

        prefs = titled_frame(self, "Downloading",
                             "Websites at once: images come from many websites; this many are downloaded from at "
                             "the same time, one image at a time from each, so a slow site does not hold up the "
                             "rest. Sites that need the browser take turns in one window.\n\nWhen a site refuses: "
                             "what the app tries when a plain download is refused.\n\nWait: how long a browser "
                             "window stays open for you to pass a CAPTCHA or bot check before the app moves on."
                             "\n\nDelay: the pause between downloads made through the browser, so the site is not "
                             "flooded.")
        prefs.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        prefs.columnconfigure(1, weight=1)
        current = image_download.PREFS
        labels = {v: k for k, v in image_download.STRATEGIES.items()}
        self.strategy_var = tk.StringVar(value=labels.get(current["strategy"], next(iter(image_download.STRATEGIES))))
        self.wait_var = tk.StringVar(value=str(int(current["challenge_wait"] // 60)))
        self.delay_var = tk.StringVar(value=str(current["browser_delay"]))
        self.sites_var = tk.StringVar(value=str(current["parallel_sites"]))
        ttk.Label(prefs, text="Websites at once").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        sites = ttk.Spinbox(prefs, textvariable=self.sites_var, from_=1, to=8, width=5, command=self._save_prefs)
        sites.grid(row=0, column=1, sticky="w", pady=3)
        ttk.Label(prefs, text="When a site refuses").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        box = ttk.Combobox(prefs, textvariable=self.strategy_var, values=list(image_download.STRATEGIES),
                           state="readonly")
        box.grid(row=1, column=1, columnspan=3, sticky="ew", pady=3)
        box.bind("<<ComboboxSelected>>", lambda _: self._save_prefs())
        ttk.Label(prefs, text="Wait for a check (min)").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
        wait = ttk.Spinbox(prefs, textvariable=self.wait_var, from_=1, to=30, width=5, command=self._save_prefs)
        wait.grid(row=2, column=1, sticky="w", pady=3)
        ttk.Label(prefs, text="Delay between browser downloads (s)").grid(row=2, column=2, sticky="e", padx=(12, 10))
        delay = ttk.Spinbox(prefs, textvariable=self.delay_var, from_=0, to=60, increment=0.5, width=5,
                            command=self._save_prefs)
        delay.grid(row=2, column=3, sticky="w")
        for widget in (wait, delay, sites):
            widget.bind("<FocusOut>", lambda _: self._save_prefs())

        self.scroll = ScrollFrame(self, app.palette)            # the list of sites can be long
        self.scroll.grid(row=1, column=0, sticky="nsew")
        self.list_frame = ttk.Frame(self.scroll.inner)
        self.list_frame.grid(row=0, column=0, sticky="nsew")
        self.list_frame.columnconfigure(0, weight=1)

        other = ttk.Frame(self)
        other.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        other.columnconfigure(0, weight=1)
        self.url_entry = placeholder(ttk.Entry(other), "Sign in to another site: its web address, for example "
                                                       "herbarium.example.org")
        self.url_entry.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.url_entry.bind("<Return>", lambda _: self._start(self.url_entry.get()))
        ttk.Button(other, text="Open browser", command=lambda: self._start(self.url_entry.get())).grid(row=0, column=1)
        self.done_button = ttk.Button(other, text="Done, save my sign-in", style="Accent.TButton",
                                      command=self._finish, state="disabled")
        self.done_button.grid(row=0, column=2, padx=(6, 0))
        HelpTip(other, "Some image sites only allow downloads from a signed-in or verified browser.\n\n1. Press "
                       "Sign in… beside a site (or enter another site and press Open browser).\n2. Sign in, or "
                       "pass the check, in the browser window that opens.\n3. Come back here and press Done.\n\n"
                       "The app keeps that site's cookies and uses them for its downloads. It never solves "
                       "CAPTCHAs itself.").grid(row=0, column=3, padx=(8, 0))
        self.status.grid(row=3, column=0, sticky="w", pady=(10, 0))
        if not browser_sign_in.available():
            self.say("Browser sign-in needs Playwright. " + browser_sign_in.INSTALL_HINT, "error")
        self.refresh()

    def _save_prefs(self):
        try:
            wait = max(int(float(self.wait_var.get())), 1) * 60
            delay = max(float(self.delay_var.get()), 0.0)
            sites = min(max(int(float(self.sites_var.get())), 1), 8)
        except ValueError:
            return
        self.app._set_site_prefs({"strategy": image_download.STRATEGIES[self.strategy_var.get()],
                                  "challenge_wait": wait, "browser_delay": delay, "parallel_sites": sites})

    def refresh(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        self.mode_vars = []
        saved = image_download.saved_hosts()
        refused = image_download.known_sites()
        reasons = image_download.site_reasons()
        row = 0
        if refused:
            help_label(self.list_frame, "Sites that need attention", "Sites that refused access, failed "
                       "repeatedly, or that you skipped. Sign in, then retry their images.",
                       style="Heading.TLabel").grid(row=row, column=0, columnspan=3, sticky="w")
            row += 1
            for host in sorted(refused):
                ids = self.app._refused_ids(host)
                signed_in = any(image_download._domain_matches(host, h) or image_download._domain_matches(h, host)
                                for h in saved)
                notes = [f"{len(ids)} images waiting"] + ([reasons[host]] if reasons.get(host) else []) \
                    + (["signed in"] if signed_in else [])
                ttk.Label(self.list_frame, text=host).grid(row=row, column=0, sticky="w")
                ttk.Label(self.list_frame, text=", ".join(notes), style="Muted.TLabel").grid(row=row, column=1,
                                                                                            sticky="w", padx=10)
                ttk.Button(self.list_frame, text="Sign in…", command=lambda h=host: self._start(h)).grid(
                    row=row, column=2, padx=6, pady=2)
                ttk.Button(self.list_frame, text=f"Retry {len(ids)}", command=lambda i=list(ids): self._retry(i),
                           state="normal" if ids else "disabled").grid(row=row, column=3, pady=2)
                row += 1
        ttk.Label(self.list_frame, text="Saved sign-ins", style="Heading.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(10 if row else 0, 0))
        row += 1
        if not saved:
            ttk.Label(self.list_frame, text="None yet", style="Muted.TLabel").grid(row=row, column=0, sticky="w")
        for host, when in sorted(saved.items()):
            ttk.Label(self.list_frame, text=host).grid(row=row, column=0, sticky="w")
            mode = tk.BooleanVar(value=image_download.site_mode(host) == image_download.MODE_BROWSER)
            self.mode_vars.append(mode)
            tip(ttk.Checkbutton(self.list_frame, text="Use the browser", variable=mode,
                                command=lambda h=host, v=mode: image_download.set_site_mode(
                                    h, image_download.MODE_BROWSER if v.get() else image_download.MODE_REQUESTS)),
                f"Saved {when.replace('T', ' ')}.\n\nDownload this site's images through the signed-in browser "
                "instead of plain requests. Slower, but it gets past sites that check for a real browser."
                ).grid(row=row, column=1, sticky="w", padx=10)
            ttk.Button(self.list_frame, text="Sign in again", command=lambda h=host: self._start(h)).grid(
                row=row, column=2, padx=6, pady=2)
            ttk.Button(self.list_frame, text="Forget", command=lambda h=host: self._forget(h)).grid(
                row=row, column=3, pady=2)
            row += 1

    def can_close(self):
        if self.done:                      # a sign-in window is open: closing saves whatever it has
            self.done.set()
        return True

    def _forget(self, host):
        image_download.forget_login(host)
        self.refresh()

    def _retry(self, ids):
        dialog = self.winfo_toplevel()
        dialog.retry_ids = set(ids)
        dialog.close()

    def _start(self, url):
        url = url.strip()
        if not url:
            self.bell()
            return
        if self.signing_in:
            return
        if not browser_sign_in.available():
            return self.say("Browser sign-in needs Playwright. " + browser_sign_in.INSTALL_HINT, "error")
        self.signing_in = True
        self.done = threading.Event()
        self.done_button.configure(state="normal")
        self.say("A browser window is opening. Sign in or pass the check, then press Done here.", "busy")
        done = self.done

        def finished(result):
            self.signing_in = False
            self.done = None
            self.done_button.configure(state="disabled")
            if isinstance(result, browser_sign_in.BrowserMissing):
                return self._offer_browser(url, str(result))
            if isinstance(result, Exception):
                self.say(str(result), "error")
            else:
                self.say(f"Saved the sign-in for {result}. You can retry its images now.", "ok")
            self.refresh()

        self.background(lambda: browser_sign_in.sign_in(url, done, self.app._log), finished)

    def _offer_browser(self, url, message):
        if not messagebox.askyesno("Browser needed", message + "\n\nDownload Chromium now?",
                                   parent=self.winfo_toplevel()):
            return self.say(message, "error")
        self.say("Downloading Chromium…", "busy")
        self.signing_in = True

        def installed(result):
            self.signing_in = False
            if isinstance(result, Exception):
                self.say(str(result), "error")
            else:
                self._start(url)

        self.background(lambda: browser_sign_in.install_chromium(self.app._log), installed)

    def _finish(self):
        if self.done:
            self.done.set()
            self.say("Saving…", "busy")


# ------------------------------------------------------------------------------------------------ Vision model

class VisionPanel(Panel):
    """The model that looks at the sheets: provider, API key, server, model and options. The provider and model
    are also on the Measure tab (the same values); the rest lives only here."""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        model = titled_frame(self, "Vision model",
                             "What looks at the sheets. LM Studio runs a vision model on this computer. The others "
                             "send the images to that provider, and usage is billed to the API key's account.")
        model.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        model.columnconfigure(1, weight=1)
        ttk.Label(model, text="Provider").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=3)
        provider = ttk.Combobox(model, textvariable=app.provider_var, state="readonly", width=34,
                                values=[info["label"] for info in PROVIDERS.values()])
        provider.grid(row=0, column=1, sticky="w", pady=3)
        provider.bind("<<ComboboxSelected>>", self._provider)
        help_label(model, "Server", "The address the model is reached at. Change it only for LM Studio on another "
                   "port or computer, or for another OpenAI-compatible server (it ends in /v1).").grid(
            row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        ttk.Entry(model, textvariable=app.server_var).grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)
        help_label(model, "Model", lambda: app.model_help).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
        self.model_box = ttk.Combobox(model, textvariable=app.model_var, values=app.model_choices.get(app.provider, []))
        self.model_box.grid(row=2, column=1, sticky="ew", pady=3)
        buttons = ttk.Frame(model)
        buttons.grid(row=2, column=2, sticky="w", padx=(6, 0))
        self.refresh_button = tip(ttk.Button(buttons, text="Refresh", command=self._models),
                                  "Ask the provider which models are available and list them in the drop-down")
        self.refresh_button.pack(side="left")
        self.test_button = tip(ttk.Button(buttons, text="Test connection", command=self._test),
                               "Check that the provider answers with these settings")
        self.test_button.pack(side="left", padx=(6, 0))

        self.key = titled_frame(self, "API key")
        self.key.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        self.key.columnconfigure(0, weight=1)
        self.key_var = tk.StringVar()
        self.key_entry = ttk.Entry(self.key, textvariable=self.key_var, show="•")
        self.key_entry.grid(row=0, column=0, sticky="ew", pady=3)
        self.key_entry.bind("<Return>", lambda _: self._save_key())
        self.show_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.key, text="Show", variable=self.show_var,
                        command=lambda: self.key_entry.configure(show="" if self.show_var.get() else "•")
                        ).grid(row=0, column=1, padx=(8, 0))
        self.save_button = ttk.Button(self.key, text="Save and test", command=self._save_key)
        self.save_button.grid(row=0, column=2, padx=(8, 0))
        self.remove_button = ttk.Button(self.key, text="Remove", command=self._remove_key)
        self.remove_button.grid(row=0, column=3, padx=(6, 0))
        self.key_state = ttk.Label(self.key, style="Muted.TLabel", wraplength=600, justify="left")
        self.key_state.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.key_link = ttk.Label(self.key, text="Get an API key", style="Link.TLabel", cursor="hand2")
        self.key_link.grid(row=1, column=3, sticky="e", pady=(4, 0))
        self.key_link.bind("<Button-1>", lambda _: webbrowser.open(PROVIDERS[self.app.provider]["key_url"]))

        options = titled_frame(self, "Options")
        options.grid(row=2, column=0, sticky="ew")
        rows = (("Token limit", "The most the model may write per image. 0 means no limit.",
                 lambda p: ttk.Spinbox(p, textvariable=app.token_var, from_=0, to=65536, increment=1000, width=8)),
                ("Thinking", "How much the model reasons before it answers, where the model supports it.",
                 lambda p: ttk.Combobox(p, textvariable=app.thinking_var, values=list(THINKING_CHOICES),
                                        state="readonly", width=14)),
                ("Image size (px)", "The longest side of the image that is sent. 0 uses the provider's default; "
                 "smaller is faster and cheaper but finds less.",
                 lambda p: ttk.Spinbox(p, textvariable=app.image_side_var, from_=0, to=4096, increment=256, width=8)))
        for row, (label, hint, make) in enumerate(rows):
            help_label(options, label, hint).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            make(options).grid(row=row, column=1, sticky="w", pady=3)
        self.status.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.refresh()

    def refresh(self):
        app, info = self.app, PROVIDERS[self.app.provider]
        local = app.provider == "lmstudio"
        for widget in (self.key_entry, self.save_button, self.remove_button):
            widget.configure(state="disabled" if local else "normal")
        store = password_store_name()
        self.key_state.configure(text="LM Studio runs on this computer and needs no key." if local else
                                 describe_key(app.provider) + (f" Saved keys are kept in {store}." if store else ""))
        if info.get("key_url") and not local:
            self.key_link.grid()
        else:
            self.key_link.grid_remove()
        self.model_box.configure(values=app.model_choices.get(app.provider, []))

    def _provider(self, _event=None):
        self.app._provider_changed()
        self.key_var.set("")
        self.say("")
        self.refresh()

    def _settings(self):
        return settings_from_app(self.app._model_settings_dict(), self.app.provider)

    def _busy(self, busy):
        for button in (self.refresh_button, self.test_button, self.save_button):
            button.configure(state="disabled" if busy else "normal")
        if not busy:
            self.refresh()

    def _models(self, after_key=False):
        provider, settings = self.app.provider, self._settings()
        self._busy(True)
        self.say("Checking the key…" if after_key else "Listing models…", "busy")

        def done(result):
            self._busy(False)
            if isinstance(result, Exception):
                return self.say(str(result), "error")
            self.app._set_model_choices(provider, result)
            self.model_box.configure(values=result)
            self.say(("The key works. " if after_key else "") + f"{len(result)} models are available.", "ok")

        self.background(lambda: list_models(settings), done)

    def _test(self):
        settings = self._settings()
        self._busy(True)
        self.say("Connecting…", "busy")

        def done(result):
            self._busy(False)
            message = str(result)
            self.say(message, "ok" if message.startswith("Connected") else "error")

        self.background(lambda: check_connection(settings), done)

    def _save_key(self):
        provider, key = self.app.provider, self.key_var.get().strip()
        if key:
            try:
                where = save_ai_api_key(provider, key)
            except OSError as exc:
                return self.say(f"Could not save the key: {exc}", "error")
            self.app._log(f"{PROVIDERS[provider]['label']} API key saved in {where}.")
            self.key_var.set("")
        elif not ai_api_key(provider, PROVIDERS[provider]["env"])[0]:
            return self.say("Paste an API key first.", "error")
        self.app._update_provider_widgets()
        self._models(after_key=True)

    def _remove_key(self):
        provider = self.app.provider
        if messagebox.askyesno("Remove key", f"Remove the saved {PROVIDERS[provider]['label']} API key?",
                               parent=self.winfo_toplevel()):
            remove_ai_api_key(provider)
            self.app._log(f"Saved {PROVIDERS[provider]['label']} API key removed.")
            self.app._update_provider_widgets()
            self.refresh()
            self.say("The saved key was removed.")


# ------------------------------------------------------------------------------------------------ the window

TABS = (("general", "General", GeneralPanel), ("account", "GBIF account", AccountPanel),
        ("sites", "Image sites", SitesPanel), ("vision", "Vision model", VisionPanel),
        ("addons", "Model add-ons", ModelSetupPanel), ("training", "Training", TrainingPanel))


class SettingsDialog(tk.Toplevel):
    """File > Settings. `tab` is the key of the tab to open on (see TABS); the last one used otherwise."""

    def __init__(self, app, tab=None):
        super().__init__(app.root)
        self.app = app
        self.retry_ids = None              # set by the Image sites tab: image downloads to run after closing
        self.title("Settings")
        self.geometry("760x660")
        self.minsize(680, 520)
        self.transient(app.root)
        self.panels, self.holders = {}, {}
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        for key, title, _ in TABS:
            self.holders[key] = ttk.Frame(self.notebook)
            self.holders[key].columnconfigure(0, weight=1)
            self.holders[key].rowconfigure(0, weight=1)
            self.notebook.add(self.holders[key], text=title)
        footer = ttk.Frame(self, padding=(12, 10, 12, 12))
        footer.pack(fill="x")
        ttk.Button(footer, text="Close", command=self.close).pack(side="right")
        self.restyle()
        self.notebook.bind("<<NotebookTabChanged>>", self._shown)
        self.show(tab if tab in self.holders else app.settings.get("settings_tab"))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda _: self.close())
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass

    def restyle(self):
        self.configure(background=self.app.palette["bg"])
        set_title_bar(self, self.app.theme_name == "dark")

    def show(self, key):
        key = key if key in self.holders else TABS[0][0]
        self.notebook.select(self.holders[key])
        self._shown()

    def _key(self):
        current = self.notebook.select()
        return next(key for key, holder in self.holders.items() if str(holder) == current)

    def _shown(self, _event=None):
        """Build the tab's panel the first time it is shown; bring it up to date every other time."""
        key = self._key()
        if key not in self.panels:
            make = next(panel for name, _, panel in TABS if name == key)
            self.panels[key] = make(self.holders[key], self.app)
            self.panels[key].grid(row=0, column=0, sticky="nsew")
        elif hasattr(self.panels[key], "refresh"):
            self.panels[key].refresh()

    def close(self):
        if not all(panel.can_close() for panel in self.panels.values() if hasattr(panel, "can_close")):
            return
        app = self.app
        app.settings["settings_tab"] = self._key()
        app._save_settings()
        app._refresh_account()
        app._update_provider_widgets()
        app.model_setup_changed()
        self.destroy()
