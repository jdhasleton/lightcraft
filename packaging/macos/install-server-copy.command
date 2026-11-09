#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$HERE/LightCraftServer.app"
DESTINATION="$HOME/Applications/LightCraftServer.app"

if [[ ! -d "$SOURCE" ]]; then
    echo "Keep this installer beside LightCraftServer.app and run it again."
    exit 1
fi

echo "Quit LightCraft and LightCraft Server before continuing."
if /usr/bin/pgrep -x LightCraft >/dev/null 2>&1; then
    echo "LightCraft is still running. Quit it, then run this installer again."
    exit 1
fi

mkdir -p "$HOME/Applications"
if [[ -e "$DESTINATION" ]]; then
    mv "$DESTINATION" "$DESTINATION.backup-$(date +%Y%m%d-%H%M%S)"
fi
/usr/bin/ditto "$SOURCE" "$DESTINATION"

# Copy only UI preferences from the stock app; the local library stays in its existing folder.
# Keep this fork's queue and settings if the installer is being run a second time.
/usr/bin/python3 <<'PY'
import json
import os
import shutil
import tempfile
from pathlib import Path

home = Path.home()
original = home / 'Library/Application Support/LightCraft/ui.json'
folder = home / 'Library/Application Support/LightCraft Server'
folder.mkdir(parents=True, exist_ok=True)
target = folder / 'ui.json'
source = target if target.exists() else original
ui = json.loads(source.read_text()) if source.exists() else {}
if not isinstance(ui, dict) or not isinstance(ui.get('settings', {}), dict):
    raise SystemExit('The existing settings are invalid; no settings have been replaced.')
if target.exists():
    shutil.copy2(target, target.with_name('ui.json.before-server-setup'))
settings = ui.setdefault('settings', {})
settings.update(dropCopyEnabled=True, dropCopyDestination='/Volumes/Photography', dropCopySmartPreviews=True)
fd, temporary = tempfile.mkstemp(prefix='.ui-', dir=folder)
with os.fdopen(fd, 'w') as handle:
    json.dump(ui, handle, ensure_ascii=False, indent=2)
    handle.flush()
    os.fsync(handle.fileno())
os.chmod(temporary, 0o600)
os.replace(temporary, target)
print('Automatic server copies enabled: /Volumes/Photography')
PY

echo "Installed LightCraft Server. Source photos will be kept; server copies and local Smart Previews are automatic."
/usr/bin/open "$DESTINATION"
