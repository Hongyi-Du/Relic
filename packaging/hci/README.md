# Relic P2/P3 Windows package

Build the P2 and P3 desktop installer from the HCI worktree:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\hci\build_windows.ps1
```

The build reads the active, gitignored `config/llm.local.yaml` at build time.
The key is present inside the resulting installer and must only be distributed
to trusted testers. Build intermediates and installers are intentionally kept
under `packaging/hci/build` and `packaging/hci/release`.

Version `0.1.13` installs two explicit shortcuts. `Relic P2 Transparent`
opens the transparent human-seat interface with the seat picker, complete
organization surface, liaison, object drill-down, and direct confirmation
controls. `Relic P3 Secretary` opens the Victor-fixed integrated
secretary interface. Both interfaces also show a persistent in-app
`P2 Transparent` / `P3 Secretary` selector, including on the P2 seat picker,
and switch within the current world. Each launch starts a fresh
`traffic_watch_v1` world in an
isolated Microsoft Edge app profile and stops its local service when that
window closes. The two modes use separate profiles and can be evaluated
independently. The package does not require Python, Node, Git, or the source
repository. It bundles the pack's declared Flask, NumPy, OpenCV, and pytest
runtime so Walker agents can run the real public suite from the installed
application.
Only the pack's manifest, starter repository, public issues, and public tests
are shipped. Reference code, hidden tests, and held-out issues stay outside the
tester package.
