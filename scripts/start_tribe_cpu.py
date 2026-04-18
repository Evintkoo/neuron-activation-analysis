#!/usr/bin/env python3
"""Start the prebuilt tribe-server in CPU mode."""
import os, subprocess, sys
env = dict(os.environ)
env["TRIBE_DEVICE"] = "cpu"
proc = subprocess.Popen(
    ["./target/release/tribe-server"],
    cwd="/Users/evintleovonzko/Documents/projects/evint/neuron-activation-analysis/tribe-playground",
    stdout=open("/tmp/srv.log", "w"),
    stderr=subprocess.STDOUT,
    env=env,
    start_new_session=True,
)
print(f"PID {proc.pid}")
