#!/usr/bin/env bash
# Configure a GNOME custom keyboard shortcut to toggle microphone mute.
#
# Tested ONLY on Ubuntu 20.04 (Focal) + GNOME 3.36 on X11.
# The gsettings schema paths used here (media-keys custom-keybindings) have
# stayed stable on Ubuntu 22.04/24.04 but this script hard-checks 20.04 so you
# can review if you ever run it on a newer release.
#
# Usage:
#   ./setup_mic_toggle.sh                   # default binding: <Ctrl><Alt>m
#   ./setup_mic_toggle.sh '<Super>m'        # custom binding
#   ./setup_mic_toggle.sh --remove          # remove the binding
#
# GNOME binding syntax: <Ctrl>, <Alt>, <Shift>, <Super>, then a key name
# (letter, Fn, space, grave, etc.). Examples:
#   '<Ctrl><Alt>m'   Ctrl+Alt+M
#   '<Super>grave'   Super+` (backtick)
#   'Pause'          bare Pause key

set -euo pipefail

BINDING_NAME="Toggle microphone mute"
COMMAND="pactl set-source-mute @DEFAULT_SOURCE@ toggle"
DEFAULT_BINDING='<Ctrl><Alt>m'

SCHEMA="org.gnome.settings-daemon.plugins.media-keys"
LIST_KEY="custom-keybindings"
BASE="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings"

die() { echo "Error: $*" >&2; exit 1; }

# --- OS check (explicit, per request) ---
[[ -f /etc/os-release ]] || die "/etc/os-release not found; cannot verify OS."
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "20.04" ]]; then
    die "This script targets Ubuntu 20.04. Detected: ${PRETTY_NAME:-$ID $VERSION_ID}
The gsettings schema path may have changed on newer GNOME versions — review before running."
fi

# --- Desktop / session checks ---
if [[ "${XDG_CURRENT_DESKTOP:-}" != *GNOME* ]]; then
    die "XDG_CURRENT_DESKTOP='${XDG_CURRENT_DESKTOP:-unset}' — GNOME Shell required for this schema."
fi

# --- Dependency checks ---
for cmd in gsettings pactl; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command '$cmd' not found."
done

# --- Verify pactl can talk to the sound server ---
pactl info >/dev/null 2>&1 || die "pactl cannot reach the sound server. Is PulseAudio running?"

# --- Parse args ---
MODE="set"
BINDING="$DEFAULT_BINDING"
case "${1:-}" in
    --remove|-r) MODE="remove" ;;
    -h|--help)
        sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    "") ;;
    *) BINDING="$1" ;;
esac

# --- Find existing slot for our command (match by command, not by binding) ---
existing=$(gsettings get "$SCHEMA" "$LIST_KEY")

# Extract paths into a bash array. Output looks like:
#   @as []
#   ['/org/.../custom0/', '/org/.../custom1/']
paths=()
if [[ "$existing" != "@as []" && "$existing" != "[]" ]]; then
    while IFS= read -r p; do
        [[ -n "$p" ]] && paths+=("$p")
    done < <(echo "$existing" | grep -oE "'/[^']+/'" | tr -d "'")
fi

our_path=""
for p in "${paths[@]}"; do
    sub="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:${p}"
    cur_cmd=$(gsettings get "$sub" command 2>/dev/null || echo "''")
    if [[ "$cur_cmd" == "'$COMMAND'" ]]; then
        our_path="$p"
        break
    fi
done

# --- Remove mode ---
if [[ "$MODE" == "remove" ]]; then
    if [[ -z "$our_path" ]]; then
        echo "No existing mic-toggle binding found. Nothing to remove."
        exit 0
    fi
    # Rebuild list without our_path
    new_paths=()
    for p in "${paths[@]}"; do
        [[ "$p" != "$our_path" ]] && new_paths+=("'$p'")
    done
    if [[ ${#new_paths[@]} -eq 0 ]]; then
        gsettings set "$SCHEMA" "$LIST_KEY" "@as []"
    else
        IFS=, ; new_list="[${new_paths[*]}]" ; unset IFS
        gsettings set "$SCHEMA" "$LIST_KEY" "$new_list"
    fi
    # Reset the per-slot keys so they don't linger
    sub="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:${our_path}"
    gsettings reset "$sub" name 2>/dev/null || true
    gsettings reset "$sub" command 2>/dev/null || true
    gsettings reset "$sub" binding 2>/dev/null || true
    echo "Removed binding at $our_path"
    exit 0
fi

# --- Set mode: pick slot (reuse existing, else next free customN) ---
if [[ -z "$our_path" ]]; then
    i=0
    while :; do
        candidate="$BASE/custom$i/"
        clash=0
        for p in "${paths[@]}"; do
            [[ "$p" == "$candidate" ]] && { clash=1; break; }
        done
        [[ $clash -eq 0 ]] && { our_path="$candidate"; break; }
        i=$((i + 1))
    done
    # Append to the list
    new_paths=()
    for p in "${paths[@]}"; do new_paths+=("'$p'"); done
    new_paths+=("'$our_path'")
    IFS=, ; new_list="[${new_paths[*]}]" ; unset IFS
    gsettings set "$SCHEMA" "$LIST_KEY" "$new_list"
    echo "Allocated slot: $our_path"
else
    echo "Reusing existing slot: $our_path"
fi

# Warn (don't block) if the binding is already bound to a different command in another slot.
for p in "${paths[@]}"; do
    [[ "$p" == "$our_path" ]] && continue
    sub="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:${p}"
    cur_bind=$(gsettings get "$sub" binding 2>/dev/null || echo "''")
    if [[ "$cur_bind" == "'$BINDING'" ]]; then
        cur_cmd=$(gsettings get "$sub" command 2>/dev/null || echo "''")
        echo "Warning: $BINDING is already bound in $p to $cur_cmd — GNOME may ignore one." >&2
    fi
done

sub="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:${our_path}"
gsettings set "$sub" name "$BINDING_NAME"
gsettings set "$sub" command "$COMMAND"
gsettings set "$sub" binding "$BINDING"

cat <<EOF

Done.
  Binding: $BINDING
  Command: $COMMAND
  Slot:    $our_path

Test it:
  1. Press $BINDING (a few times)
  2. Check state (works on PulseAudio 13+):
     pactl list sources | awk -v s="\$(pactl info | awk -F': ' '/Default Source/{print \$2}')" '\$1=="Name:"{n=\$2} n==s && \$1=="Mute:"{print "mic "\$2; exit}'

Remove it later with:
  $0 --remove
EOF
