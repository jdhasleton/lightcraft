//! Automatic, non-destructive server copies for photos dropped on the native app.
//! The saved queue retains sources until an import succeeds; filesystem checks run off the UI thread.

use crate::{LightcraftApp, import};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pending {
    pub paths: Vec<String>,
    pub destination: String,
    pub library: String,
    pub smart_previews: bool,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Default)]
pub(crate) struct Runtime {
    pub next_retry: f64,
    checking: Option<(Pending, std::sync::mpsc::Receiver<Result<(), String>>, lightcraft_engine::activity::TaskGuard)>,
}

fn library(app: &LightcraftApp) -> Option<String> {
    app.session.library.as_ref().map(|l| l.dir.to_string_lossy().to_string())
}

pub(crate) fn enqueue(app: &mut LightcraftApp, paths: Vec<String>) -> Result<(), String> {
    let destination = app.ui.settings.drop_copy_destination.trim().to_string();
    if destination.is_empty() || !std::path::Path::new(&destination).is_absolute() {
        return Err("Choose an absolute server folder in Settings > Import > Automatic server copy first".into());
    }
    let library = library(app).ok_or("Open a library on disk before queuing server copies")?;
    if app.ui.drop_transfers.len() >= 10_000 || paths.len() > 100_000 {
        return Err("Too many pending transfers; finish existing transfers before adding more photos".into());
    }
    let pending = Pending { paths, destination, library, smart_previews: app.ui.settings.drop_copy_smart_previews, error: None };
    if !app.ui.drop_transfers.iter().any(|p| same_job(p, &pending)) {
        app.ui.drop_transfers.push(pending);
    }
    app.drop_transfer.next_retry = 0.0;
    Ok(())
}

fn same_job(a: &Pending, b: &Pending) -> bool {
    a.paths == b.paths && a.destination == b.destination && a.library == b.library
}

/// Never create the chosen root: an offline share must not become a local folder by accident.
/// On macOS, a path below /Volumes additionally has to be on a currently mounted SMB volume.
pub(crate) fn check_destination(destination: &str) -> Result<(), String> {
    let path = std::path::Path::new(destination);
    if !path.is_absolute() || path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return Err("The transfer destination must be an absolute path without '..'".into());
    }
    #[cfg(target_os = "macos")]
    if let Ok(relative) = path.strip_prefix("/Volumes") {
        let first = relative.components().next().ok_or("Choose a mounted share, not /Volumes itself")?;
        let root = std::path::Path::new("/Volumes").join(first.as_os_str());
        let output = std::process::Command::new("/sbin/mount").output().map_err(|e| format!("Cannot check mounted shares: {e}"))?;
        let marker = format!(" on {} (smbfs", root.display());
        if !output.status.success() || !String::from_utf8_lossy(&output.stdout).lines().any(|line| line.contains(&marker)) {
            return Err("Server share is not mounted; waiting for it to reconnect".into());
        }
    }
    if !path.is_dir() {
        return Err("Server folder is unavailable; waiting for it to reconnect".into());
    }
    Ok(())
}

fn failed(app: &mut LightcraftApp, ctx: &egui::Context, pending: &Pending, error: String) {
    if let Some(p) = app.ui.drop_transfers.iter_mut().find(|p| same_job(p, pending)) {
        let changed = p.error.as_ref() != Some(&error);
        p.error = Some(error.clone());
        if changed {
            app.toast_for(ctx, format!("Server copy pending: {error}. Source files have been kept."), 8.0);
        }
    }
    app.drop_transfer.next_retry = app.tick_now() + 60.0;
}

pub(crate) fn tick(app: &mut LightcraftApp, ctx: &egui::Context) {
    if app.ui.drop_transfers.is_empty() {
        app.drop_transfer.checking = None;
        return;
    }
    ctx.request_repaint_after(std::time::Duration::from_secs(1));
    if app.ui.dialog.is_some()
        || crate::panels::library_problem::holds_drops(app)
        || app.lightroom.is_some()
        || app.sync_run.is_some()
        || import::running(app)
        || app.session.preview_build.as_ref().is_some_and(|p| p.running())
    {
        return;
    }
    if let Some((pending, rx, guard)) = app.drop_transfer.checking.take() {
        match rx.try_recv() {
            Ok(Ok(())) => {
                // Opening another library or cancelling the queue while the check ran must not
                // import these files into a different catalog.
                if library(app).as_deref() == Some(pending.library.as_str())
                    && app.ui.drop_transfers.iter().any(|p| same_job(p, &pending))
                    && let Err(e) = import::start_server_copy(app, pending.clone())
                {
                    failed(app, ctx, &pending, e);
                }
            }
            Ok(Err(e)) => failed(app, ctx, &pending, e),
            Err(std::sync::mpsc::TryRecvError::Empty) => app.drop_transfer.checking = Some((pending, rx, guard)),
            Err(std::sync::mpsc::TryRecvError::Disconnected) => failed(app, ctx, &pending, "Destination check stopped unexpectedly".into()),
        }
        return;
    }
    if app.tick_now() < app.drop_transfer.next_retry {
        return;
    }
    let current = library(app);
    let Some(pending) = app.ui.drop_transfers.iter().find(|p| Some(&p.library) == current.as_ref()).cloned() else { return };
    let (tx, rx) = std::sync::mpsc::channel();
    let destination = pending.destination.clone();
    let repaint = ctx.clone();
    match std::thread::Builder::new().name("lc-server-check".into()).spawn(move || {
        let _ = tx.send(check_destination(&destination));
        repaint.request_repaint();
    }) {
        Ok(_) => {
            let guard = app.session.activity.start("import", "Checking server folder", lightcraft_engine::activity::Cancel::No);
            app.drop_transfer.checking = Some((pending, rx, guard));
        }
        Err(e) => failed(app, ctx, &pending, format!("Could not start destination check: {e}")),
    }
}

pub(crate) fn complete(app: &mut LightcraftApp, ctx: &egui::Context, pending: &Pending, ids: &[u64], error: Option<String>) {
    if let Some(error) = error {
        failed(app, ctx, pending, error);
        return;
    }
    // A same-content photo already catalogued on the laptop is not a server backup.
    let destination = std::path::Path::new(&pending.destination);
    let all_on_server = !ids.is_empty()
        && ids.iter().all(|id| {
            app.session.catalog.photo(lightcraft_catalog::PhotoId(*id)).is_some_and(
                |p| matches!(&p.source, lightcraft_catalog::Source::File { path } if std::path::Path::new(path).starts_with(destination)),
            )
        });
    if !all_on_server {
        failed(
            app,
            ctx,
            pending,
            "No verified server originals were catalogued, or a duplicate is still referenced outside the server folder".into(),
        );
        return;
    }
    // Use the app's normal asynchronous journal writer. Waiting for fsync here could freeze
    // the UI on a slow or failing catalog drive; sources are retained regardless.
    app.session.persist_if_dirty();
    app.session.refresh_unsaved();
    if let Some(error) = app.session.unsaved().map(|(_, error)| error.to_string()) {
        failed(app, ctx, pending, format!("Catalog could not be saved: {error}"));
        return;
    }
    app.ui.drop_transfers.retain(|p| !same_job(p, pending));
    app.drop_transfer.next_retry = 0.0;
    if pending.smart_previews
        && let Err(e) = app.run("library.smartPreviews", json!({"ids": ids, "background": true}))
    {
        app.toast_for(ctx, format!("Server originals copied; smart previews could not start: {e}"), 8.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offline_root_is_never_created() {
        let root = std::env::temp_dir().join(format!("lc-offline-copy-{}", std::process::id()));
        assert!(!root.exists());
        assert!(check_destination(&root.to_string_lossy()).is_err());
        assert!(!root.exists());
        assert!(check_destination("relative/folder").is_err());
    }

    #[test]
    fn pending_jobs_round_trip_with_their_library_and_destination() {
        let mut ui = crate::UiState::default();
        ui.drop_transfers.push(Pending {
            paths: vec!["/source/photo.cr3".into()],
            destination: "/server/photos".into(),
            library: "/local/catalog".into(),
            smart_previews: true,
            error: Some("offline".into()),
        });
        let saved = serde_json::to_vec(&ui).unwrap();
        let restored: crate::UiState = serde_json::from_slice(&saved).unwrap();
        assert_eq!(restored.drop_transfers, ui.drop_transfers);
    }
}
