#!/bin/bash
# sync_and_build.sh
# Periodically fetches simulation data from the remote server, regenerates
# all figures, and rebuilds the showyourwork paper.
#
# Usage:
#   chmod +x sync_and_build.sh
#   ./sync_and_build.sh            # run indefinitely (Ctrl-C to stop)
#   ./sync_and_build.sh --once     # run a single cycle and exit

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
INTERVAL_SECONDS=600   # 10 minutes

REMOTE_USER="picogna"
REMOTE_HOST="ercol1"
REMOTE_BASE="/e/ocean1/users/picogna/Photoevaporation-OuterRadius-Project/SPOCK_runs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/src/data"
SCRIPTS_DIR="${SCRIPT_DIR}/src/scripts"

LOG_FILE="${SCRIPT_DIR}/sync_and_build.log"

# ── Helpers ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "${LOG_FILE}"
}

run_step() {
    local label="$1"; shift
    log "START  $label"
    if "$@" >> "${LOG_FILE}" 2>&1; then
        log "OK     $label"
    else
        log "FAILED $label (exit $?) — continuing"
    fi
}

# ── Single cycle ──────────────────────────────────────────────────────────────
run_cycle() {
    log "════════════════════════════════════════"
    log "Beginning sync-and-build cycle"

    # 1. Sync data from remote server
    local_dirs=(
        "pop_spreading_reduced"
        "pop_spreading"
        "pop_spreading_reduced_extern"
        "pop_spreading_extern"
    )
    remote_paths=(
        "${REMOTE_BASE}/pop_XEUV_reduced/*.dat"
        "${REMOTE_BASE}/pop_XEUV/*.dat"
        "${REMOTE_BASE}/pop_XEUV_FUV_factor10/*.dat"
        "${REMOTE_BASE}/pop_XEUV_FUV_standard/*.dat"
    )

    for i in "${!local_dirs[@]}"; do
        local_dir="${local_dirs[$i]}"
        remote_path="${remote_paths[$i]}"
        dest="${DATA_DIR}/${local_dir}/"
        mkdir -p "$dest"
        run_step "scp → ${local_dir}" \
            scp "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}" "$dest"
    done

    # 2. Regenerate figures
    for fig in fig1 fig2 fig3 fig4 fig5_mamajek fig6 fig7; do
        run_step "python ${fig}.py" \
            python "${SCRIPTS_DIR}/${fig}.py"
    done

    # 3. Rebuild paper
    run_step "showyourwork build" \
        showyourwork build

    log "Cycle complete"
    log "════════════════════════════════════════"
}

# ── Main loop ─────────────────────────────────────────────────────────────────
ONCE=false
[[ "${1:-}" == "--once" ]] && ONCE=true

log "sync_and_build.sh started (interval=${INTERVAL_SECONDS}s, once=${ONCE})"
log "Log file: ${LOG_FILE}"

while true; do
    run_cycle

    if $ONCE; then
        log "Single-cycle mode — exiting"
        exit 0
    fi

    log "Sleeping ${INTERVAL_SECONDS}s until next cycle…"
    sleep "${INTERVAL_SECONDS}"
done
