# scripts/run-agents.ps1
# Launch all agents in separate Windows Terminal tabs (fallback: new PowerShell windows)
# Works from anywhere: sets working dir to repo root.

# --- go to repo root (this file lives in scripts/) ---
Set-Location ($PSScriptRoot | Join-Path -ChildPath "..")

# --- prefer venv python if present ---
$VenvPython = Join-Path ".\.venv\Scripts" "python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

# --- agents: unique seed + port + module path ---
$AGENTS = @(
    @{ name="weapon-detector";   module="src.agents.weapon_agent";        seedVar="WEAPON_AGENT_SEED";     seed="weapon-seed-1";     portVar="WEAPON_AGENT_PORT";     port=8101 },
    @{ name="robbery-detector";  module="src.agents.robbery_agent";       seedVar="ROBBERY_AGENT_SEED";    seed="robbery-seed-1";    portVar="ROBBERY_AGENT_PORT";    port=8102 },
    @{ name="face-detector";     module="src.agents.face_agent";          seedVar="FACE_AGENT_SEED";       seed="face-seed-1";       portVar="FACE_AGENT_PORT";       port=8103 },
    @{ name="transcript-maker";  module="src.agents.transcript_agent";    seedVar="TRANSCRIPT_AGENT_SEED"; seed="transcript-seed-1"; portVar="TRANSCRIPT_AGENT_PORT"; port=8104 },
    @{ name="voice-agent";       module="src.agents.voice_agent";         seedVar="VOICE_AGENT_SEED";      seed="voice-seed-1";      portVar="VOICE_AGENT_PORT";      port=8105 },
    @{ name="video-process";     module="src.agents.video_process_agent"; seedVar="VIDEO_AGENT_SEED";      seed="video-seed-1";      portVar="VIDEO_AGENT_PORT";      port=8009 },
    @{ name="timestamp-finder";  module="src.agents.timestamp_agent";     seedVar="TS_AGENT_SEED";         seed="ts-seed-1";         portVar="TS_AGENT_PORT";         port=8107 },
    @{ name="search-query";      module="src.agents.search_agent";        seedVar="SEARCH_AGENT_SEED";     seed="search-seed-1";     portVar="SEARCH_AGENT_PORT";     port=8201 },
    @{ name="notification";      module="src.agents.notification_agent";  seedVar="NOTIFY_AGENT_SEED";     seed="notify-seed-1";     portVar="NOTIFY_AGENT_PORT";     port=8202 },
    @{ name="timeline-ingest";   module="src.agents.timeline_ingest";     seedVar="TIMELINE_AGENT_SEED";   seed="timeline-seed-1";   portVar="TIMELINE_AGENT_PORT";   port=8203 }
)

function Test-PortFree {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return -not $conn
}

function Launch-Agent {
    param($a)

    # Per-process env
    $envCmd = "$($a.seedVar)='$($a.seed)'; $($a.portVar)='$($a.port)';"
    # Ensure imports work even if run from IDE
    $envCmd += " `$env:PYTHONPATH='$PWD';"

    $cmd = "$envCmd & `"$Python`" -m $($a.module)"

    if (Get-Command wt -ErrorAction SilentlyContinue) {
        wt -w 0 nt --title $($a.name) powershell -NoExit -Command $cmd | Out-Null
    } else {
        Start-Process powershell -ArgumentList "-NoExit","-Command",$cmd | Out-Null
    }
}

# --- sanity: package structure ---
if (-not (Test-Path "src\__init__.py"))     { New-Item -ItemType File -Path "src\__init__.py"     -Force | Out-Null }
if (-not (Test-Path "src\common\__init__.py")) { New-Item -ItemType File -Path "src\common\__init__.py" -Force | Out-Null }
# src\agents\__init__.py already exists per your tree

# --- safety checks: unique ports and availability ---
$ports = $AGENTS.port
if ($ports.Count -ne ($ports | Select-Object -Unique).Count) {
    Write-Error "Duplicate ports in AGENTS list. Assign unique ports to each agent."
    exit 1
}
foreach ($p in $ports) {
    if (-not (Test-PortFree -Port $p)) {
        Write-Error "Port $p is already in use. Stop the process using it or choose a different port."
        exit 1
    }
}

Write-Host "Starting agents with $Python from: $PWD"
foreach ($a in $AGENTS) {
    Write-Host (" - {0} on {1}" -f $a.name, $a.port)
    Launch-Agent $a
}

Write-Host "`nAll agents launched in separate tabs/windows."
Write-Host "Each agent will print its own Inspector link (e.g., https://agentverse.ai/inspect/?uri=http://127.0.0.1:<PORT>&address=<DID>)"
Write-Host "Open the link and click 'Create mailbox' once per agent DID."
