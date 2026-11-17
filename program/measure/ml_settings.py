"""Settings > Training: how training runs (device, epochs, image size, batch), with a button that picks values
suited to this computer and the labelled sheets. Saved in settings.json under "training"; training.train() reads
them, and anything passed to it (the Train dialog, the command line) wins over them."""
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from config import save_settings
from . import training
from ..gui_widgets import help_label, titled_frame


class TrainingPanel(ttk.Frame):
    """Settings > Training: device, epochs, image size and batch for every training run."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=16)
        self.app = app
        saved = training.saved_settings()
        self.device = tk.StringVar(value=training.DEVICE_CHOICES[saved["device"]])
        self.epochs = tk.StringVar(value=str(saved["epochs"]))
        self.imgsz = tk.StringVar(value=str(saved["imgsz"]))
        self.batch = tk.StringVar(value=str(saved["batch"] or ""))
        memory = training.system_memory_gb()
        self.hardware = tk.StringVar(value=f"This computer: {memory:.0f} GB memory." if memory else "")

        self.columnconfigure(0, weight=1)
        form = titled_frame(self, "Training your own models",
                            "These apply to every training run (Review tab > Train models; the Train window still "
                            "lets you change the epochs and image size for one run).\n\nSet optimal chooses "
                            "values for this computer and your reviewed sheets. If a run still runs out of memory, "
                            "training lowers the batch size and then the image size by itself and carries on.")
        form.grid(row=0, column=0, sticky="ew")
        rows = (("Run on", self.device, "Which processor trains the model. Automatic uses an Apple or NVIDIA GPU "
                 "when there is one. A GPU that is not there falls back to the CPU."),
                ("Epochs", self.epochs, "How many passes over the sheets. More takes longer and fits closer."),
                ("Image size (px)", self.imgsz, "The size the sheets are trained at, at least 320. Larger sees "
                 "small seeds better but needs much more memory."),
                ("Batch size", self.batch, "Images trained at once. Larger is faster but needs more memory. "
                 "Leave it empty to let training pick one for the device."))
        for row, (text, var, hint) in enumerate(rows):
            help_label(form, text, hint).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 14))
            if var is self.device:
                ttk.Combobox(form, textvariable=var, values=list(training.DEVICE_CHOICES.values()),
                             state="readonly", width=14).grid(row=row, column=1, sticky="w", pady=3)
            else:
                ttk.Entry(form, textvariable=var, width=10).grid(row=row, column=1, sticky="w", pady=3)
        buttons = ttk.Frame(self)
        buttons.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(buttons, text="Save", style="Accent.TButton", command=self._save).pack(side="left")
        self.optimal_button = ttk.Button(buttons, text="Set optimal", command=self._optimal)
        self.optimal_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Defaults", command=self._defaults).pack(side="left")
        ttk.Label(buttons, textvariable=self.hardware, style="Muted.TLabel").pack(side="left", padx=12)
        self.note = ttk.Label(self, text="", wraplength=640, justify="left", style="Muted.TLabel")
        self.note.grid(row=2, column=0, sticky="w", pady=(10, 0))

    def _device_key(self):
        return next((key for key, label in training.DEVICE_CHOICES.items() if label == self.device.get()), "auto")

    def _fill(self, settings):
        self.device.set(training.DEVICE_CHOICES[settings.get("device", "auto")])
        self.epochs.set(str(settings["epochs"]))
        self.imgsz.set(str(settings["imgsz"]))
        self.batch.set(str(settings.get("batch") or ""))

    def _defaults(self):
        self._fill({**training.DEFAULTS, "batch": None, "device": "auto"})
        self.note.configure(text="The built-in defaults. Training picks the batch size for the device. Press Save "
                                 "to keep them.")

    def _optimal(self):
        """Look at the computer and the labelled sheets (off the window thread: PyTorch takes a moment to load)."""
        self.optimal_button.state(["disabled"])
        self.hardware.set("Checking this computer…")

        def work():
            try:
                found = training.hardware()
                sheets = len(training.gather_labels())
                error = None
            except Exception as exc:
                found, sheets, error = None, 0, exc
            try:
                self.after(0, lambda: self._optimal_done(found, sheets, error))
            except tk.TclError:
                pass                      # the window was closed meanwhile

        threading.Thread(target=work, daemon=True).start()

    def _optimal_done(self, found, sheets, error):
        if not self.winfo_exists():          # the Settings window was closed meanwhile
            return
        self.optimal_button.state(["!disabled"])
        if error or not found:
            self.hardware.set("")
            messagebox.showwarning("Training", f"Could not check this computer: {error}",
                                   parent=self.winfo_toplevel())
            return
        memory = found["memory_gb"]
        self.hardware.set(f"This computer: {found['name']}" + (f", {memory:.0f} GB memory." if memory else "."))
        train_images = round(sheets * dict(training.SPLITS)["train"]) if sheets else None
        settings = training.recommend(found["device"], memory, train_images)
        self._fill(settings)
        self.note.configure(text=(
            f"Suggested for {found['name']}: image size {settings['imgsz']}, batch {settings['batch']}, "
            f"{settings['epochs']} epochs" + (f" for about {train_images} training sheets." if train_images else
                                              " (no reviewed sheets yet, so the default epochs).")
            + " Press Save to keep them."))

    def _save(self):
        try:
            epochs, imgsz = int(self.epochs.get()), int(self.imgsz.get())
            batch = int(self.batch.get()) if self.batch.get().strip() else None
        except ValueError:
            messagebox.showwarning("Training", "Epochs, image size and batch size must be whole numbers.",
                                   parent=self.winfo_toplevel())
            return
        if epochs < 1 or imgsz < 320 or (batch is not None and batch < 1):
            messagebox.showwarning("Training", "Use at least 1 epoch, 320 px and a batch of 1.",
                                   parent=self.winfo_toplevel())
            return
        self.app.settings["training"] = {"device": self._device_key(), "epochs": epochs, "imgsz": imgsz,
                                         "batch": batch}
        save_settings(self.app.settings)
        self.app._log(f"Training settings saved: {training.DEVICE_CHOICES[self._device_key()]}, {epochs} epochs, "
                      f"image size {imgsz}, batch {batch or 'automatic'}.")
        self.note.configure(text="Saved.")
