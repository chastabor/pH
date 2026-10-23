# Running `phern daemon` under something that restarts it

phern never restarts itself. It reports facts and resumes work when asked, and
leaves the process, its environment and its backoff to whatever already owns
those on the host. What it offers an orchestrator is small and the same
everywhere:

| it needs | phern gives |
|---|---|
| to know when the daemon is up | `READY=1` on `$NOTIFY_SOCKET`; `GET /ready` on `--probe-bind`; `phern agents probe ready` |
| to know when it is stuck | `WATCHDOG=1` while the event loop moves; `GET /live` |
| to resume what the last daemon owed | `phern daemon --resume-owed`, `phern agents resume --all`, `phern service start` |
| to see what it owes | `GET /status`; `phern agents owed`; `phern service status` |
| an exit it can branch on | 0 clean, 3 when a session log's tail could not be written |

Three shapes are here. Each is a template to edit, not a package to install.

## systemd, as a user service — `systemd/user/phern-daemon.service`

For a workstation or a single-user host. `phern` is on `~/.local/bin` where
`uv tool install phern` put it, `$PH_HOME` is `~/.ph`, the socket is under
`$XDG_RUNTIME_DIR`.

```bash
mkdir -p ~/.config/systemd/user ~/.config/phern
cp contrib/systemd/user/phern-daemon.service ~/.config/systemd/user/
cp contrib/systemd/phern-daemon.env.example ~/.config/phern/daemon.env   # edit it
systemctl --user daemon-reload
systemctl --user enable --now phern-daemon
loginctl enable-linger "$USER"
```

The last line is the one `phern doctor` tells you about: without it logind ends
your user services, and removes the directory the socket lives in, when your last
session ends — so a daemon enabled at 09:00 is gone by the time a schedule fires at
02:00.

## systemd, as a system service — `systemd/phern-daemon@.service`

One instance per user, because one daemon serves one `$PH_HOME` and holds every
session lease under it. The instance name is the unix user it runs as.

```bash
sudo cp contrib/systemd/phern-daemon@.service /etc/systemd/system/
sudo mkdir -p /etc/phern
sudo cp contrib/systemd/phern-daemon.env.example /etc/phern/alice.env       # edit it
sudo systemctl daemon-reload
sudo systemctl enable --now phern-daemon@alice
```

`$PH_RUNTIME` is `/run/phern-alice`, made by `RuntimeDirectory=` and owned by the
instance's user; `$PH_HOME` is that user's `~/.ph` unless the env file moves it.

## Kubernetes — `kubernetes/phern-daemon.yaml`

A `Deployment` of one replica with `Recreate` (never two daemons on one home), a
volume for `$PH_HOME`, an `emptyDir` for `$PH_RUNTIME`, and the three probes over
`--probe-bind`. The resume pass is the daemon's own (`--resume-owed`), since there
is no `ExecStartPost` to run it from; readiness is no until it ends.

## No init system — `phern service`

A cron slot, a container entrypoint, a shell:

```bash
phern service start --profile work      # spawn, wait for ready (its own resume pass included), report
phern service status                    # exit 0 ready, 1 down, 2 degraded
phern service watch --profile work      # start when down, resume when owed, report when degraded
phern service watch --once --profile work   # one pass, for cron
phern service stop
```

Everything after `start`, `restart` or `watch` goes to `phern daemon` as typed,
and a daemon `start` spawns gets `--resume-owed` too. The daemon composes its own
configuration from `$PH_HOME`, `$PH_RUNTIME` and those options; `phern service`
passes no environment and reads no files. `watch` waits between passes on the
scheduler's retry ladder, and a signal ends the watcher, not the daemon.

## What every shape has in common

- **Credentials reach the daemon as environment and are never written down.** A
  root whose route names a key the daemon does not have is held, by name, until
  `credentials/store` or a restart with the key supplies it.
- **`SIGTERM` is `phern agents shutdown`.** Sub-agents are suspended in their own
  logs, child processes stopped, logs written, leases released. The daemon's own
  budget for that is fifteen seconds; give it twenty before a `SIGKILL`.
- **Exit 3 is a lost tail.** A stop that could not write a session log says so in
  its exit status (and `ready` said no while it ran). `on-failure` restarts it, and
  the next daemon's resume repairs what this one could not write.
- **Nothing fires while the daemon is down.** A schedule is kept while
  `phern daemon` runs and caught up on start, one run per missed window. Work that
  must happen whether or not the daemon is up belongs to cron or a timer.
