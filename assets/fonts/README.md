# Figure fonts

Publication figures prefer **Helvetica**. This machine/repo may not ship the
proprietary Helvetica files, so we bundle **GNU FreeSans** (FreeFont), a
Helvetica-compatible open substitute, and fall back to Arial / DejaVu Sans
when needed.

`src/visualization/style.py` registers these TTFs and selects the best
available family at runtime.
