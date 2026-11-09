# Automatic server copies

Settings > Import > Automatic server copy can save one destination for photos and folders dropped
on the native app. Enable it, choose a mounted server folder, and optionally enable local Smart
Previews. This takes precedence over the ordinary drop-import review setting.

Drops enter a queue saved with the application preferences. While the app is open, it copies
the sources into capture-date folders using the existing verified Copy importer, references
the server originals, and optionally builds Smart Previews in the library's preview location.
Keep the library and previews on the computer for offline use.

The destination root must already exist. On macOS a destination under `/Volumes` must be on
a currently mounted SMB share; the app never creates a replacement root on the startup disk.
An unavailable destination is retried once a minute. Connect shares through Finder or a separate
automount service. Pending batches remain tied to the library for which they were queued and
are never imported into a different library. Sources must stay at their original paths until
the transfer finishes. Photos queued while offline appear in the library after their transfer.

Settings lists pending batches and their last error, with Retry now and Clear pending copies.
Clearing the queue does not delete source photos or server copies. Cancelling an active import
retains its batch for retry. A catalog duplicate referenced outside the destination is retained
as an error rather than incorrectly reported as backed up.

Source files are never deleted. Copying an original to its primary storage is not a versioned
backup of that storage or of the local catalog. Full-resolution exports need server originals;
Smart Previews support offline edits and smaller exports.

Private macOS builds compiled with `LIGHTCRAFT_SERVER_BUILD=1` use the separate application
settings directory `~/Library/Application Support/LightCraft Server` and disable startup checks
for stock releases, which would replace this modification. `LIGHTCRAFT_CONFIG_DIR` still
overrides that directory for isolated tests. Existing libraries keep their normal locations.
