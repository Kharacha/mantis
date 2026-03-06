import os, subprocess, sys, time, shutil

# (name, module, seed_env, seed, port_env, port)
AGENTS = [
    ("weapon-detector",   "src.agents.weapon_agent",        "WEAPON_AGENT_SEED",     "weapon-seed-1",     "WEAPON_AGENT_PORT",     "8101"),
    ("robbery-detector",  "src.agents.robbery_agent",       "ROBBERY_AGENT_SEED",    "robbery-seed-1",    "ROBBERY_AGENT_PORT",    "8102"),
    ("face-detector",     "src.agents.face_agent",          "FACE_AGENT_SEED",       "face-seed-1",       "FACE_AGENT_PORT",       "8103"),
    ("transcript-maker",  "src.agents.transcript_agent",    "TRANSCRIPT_AGENT_SEED", "transcript-seed-1", "TRANSCRIPT_AGENT_PORT", "8104"),
    ("voice-agent",       "src.agents.voice_agent",         "VOICE_AGENT_SEED",      "voice-seed-1",      "VOICE_AGENT_PORT",      "8105"),
    ("video-process",     "src.agents.video_process_agent", "VIDEO_AGENT_SEED",      "video-seed-1",      "VIDEO_AGENT_PORT",      "8009"),
    ("timestamp-finder",  "src.agents.timestamp_agent",     "TS_AGENT_SEED",         "ts-seed-1",         "TS_AGENT_PORT",         "8107"),
    ("search-query",      "src.agents.search_agent",        "SEARCH_AGENT_SEED",     "search-seed-1",     "SEARCH_AGENT_PORT",     "8201"),
    ("notification",      "src.agents.notification_agent",  "NOTIFY_AGENT_SEED",     "notify-seed-1",     "NOTIFY_AGENT_PORT",     "8202"),
    ("timeline-ingest",   "src.agents.timeline_ingest",     "TIMELINE_AGENT_SEED",   "timeline-seed-1",   "TIMELINE_AGENT_PORT",   "8203"),
]

def port_in_use(port: str) -> bool:
    # Cross-platform-ish check using 'netstat' or 'ss'
    netstat = shutil.which("netstat") or shutil.which("ss")
    if not netstat:
        return False
    try:
        out = subprocess.check_output([netstat, "-ano"], stderr=subprocess.STDOUT, text=True)
        return f":{port} " in out or f":{port}\n" in out
    except Exception:
        return False

def spawn(name, module, seed_env, seed, port_env, port):
    env = os.environ.copy()
    env[seed_env] = seed
    env[port_env] = port
    print(f"-> starting {name:18s} on {port}  ({module})")
    return subprocess.Popen([sys.executable, "-m", module], env=env)

if __name__ == "__main__":
    ports = [p for *_, p in AGENTS]
    if len(ports) != len(set(ports)):
        print("Duplicate ports detected in AGENTS list. Make them unique.")
        sys.exit(1)
    for p in ports:
        if port_in_use(p):
            print(f"Port {p} appears to be in use. Stop the process using it or choose another port.")
            sys.exit(1)

    procs = []
    try:
        for a in AGENTS:
            procs.append(spawn(*a))
            time.sleep(0.3)  # small stagger
        print("\nAll agents launched. Press Ctrl+C to stop.")
        # Wait on children
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nStopping agents...")
        for p in procs:
            p.terminate()
