"""Exercise one prepared Linux profile via a PTY. Requires pexpect and seven stopped services."""
import argparse
import json
import os
import socket
import subprocess
import urllib.request
from pathlib import Path

import pexpect

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", required=True, type=Path)
parser.add_argument("--config", required=True, type=Path, help="SA3 base profile")
args = parser.parse_args()
binary = str(args.binary.resolve())
env = {**os.environ, "GARY4LOCAL_HEADLESS_CONFIG": str(args.config.resolve())}
routes = {"gary": "/api/models", "melodyflow": "/variations", "stable-audio": "/ready",
          "foundation": "/ready", "carey": "/props", "yuey": "/health", "sa3": "/health"}


def status(service):
    output = subprocess.check_output([binary, "status", service, "--json"], env=env)
    return json.loads(output)["data"]


for service in routes:
    assert not status(service)["operation_active"], f"{service} is already in use"
terminal = pexpect.spawn(binary, env=env, encoding="utf-8", timeout=240)
try:
    terminal.expect_exact("gary> ")
    ports = {}
    for service, route in routes.items():
        terminal.sendline("/start " + service)
        label = "SA3" if service == "sa3" else service
        terminal.expect_exact(label + " ready at")
        terminal.expect_exact("gary> ")
        state = status(service)
        assert state["running"] and state["health"], service
        with urllib.request.urlopen(state["public_url"] + route, timeout=15) as response:
            body = json.load(response)
        if route == "/ready":
            assert body["ready"], service
        ports[service] = state["state"]["public_port"]
        print(f"{service}: {state['public_url']}{route} OK", flush=True)

    terminal.sendline("/status")
    for service in ["sa3", "gary", "melodyflow", "stable-audio", "carey", "foundation", "yuey"]:
        terminal.expect(service + r" +running +http://127.0.0.1:")
    terminal.expect_exact("gary> ")
    terminal.sendline("/stop")
    terminal.expect_exact("Choose a service")
    terminal.send("\x1b[B\r")  # SA3 -> Gary
    terminal.expect_exact("gary stopped.")
    terminal.expect_exact("gary> ")
    assert not status("gary")["running"]
    assert all(status(service)["running"] for service in routes if service != "gary")
    terminal.sendline("/start gary")
    terminal.expect_exact("gary ready at")
    terminal.expect_exact("gary> ")
    # A second console must leave all seven external owners alone on exit.
    observer = pexpect.spawn(binary, env=env, encoding="utf-8", timeout=30)
    try:
        observer.expect_exact("Running in another session.")
        observer.expect_exact("gary> ")
        observer.sendline("/quit")
        observer.expect(pexpect.EOF)
    finally:
        observer.close(force=True)
    assert all(status(service)["running"] for service in routes)
    terminal.sendline("/quit")
    terminal.expect(pexpect.EOF)
    terminal.close()
    assert terminal.exitstatus == 0
    for service, port in ports.items():
        assert not status(service)["operation_active"]
        with socket.socket() as probe:
            assert probe.connect_ex(("127.0.0.1", port)) != 0, service
    print("Selective stop, restart and owned-session cleanup passed.", flush=True)
finally:
    if terminal.isalive():
        terminal.sendintr()
        terminal.expect(pexpect.EOF)
        terminal.close()
