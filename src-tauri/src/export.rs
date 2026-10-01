// Export pipeline: composite the multi-track timeline into one FFmpeg
// filter_complex invocation. Each clip is trimmed, speed-adjusted, scaled by
// its transform, time-shifted to its start and overlaid (z-ordered by track)
// onto a black canvas; audio is delayed and mixed. Progress streams back to the
// UI. FFmpeg/ffprobe ship as bundled sidecars (see scripts/fetch-ffmpeg.ps1).
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::process::CommandEvent;
use tauri_plugin_shell::ShellExt;

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportClip {
	/// "video" (composited) or "audio" (mixed only).
	kind: String,
	path: String,
	in_point: f64,
	out_point: f64,
	speed: f64,
	volume: f64,
	muted: bool,
	fade_in: f64,
	fade_out: f64,
	/// Absolute start on the master timeline (seconds).
	start: f64,
	/// Z-order lane (higher = composited on top).
	track: i64,
	/// Viewport transform: center offset (normalized) + scale (relative to fit).
	x: f64,
	y: f64,
	scale: f64,
	/// Source aspect ratio (width / height).
	aspect_ratio: f64,
	/// Text styling; present only on `kind: "text"` clips.
	#[serde(default)]
	text: Option<ExportText>,
}

/// Text-overlay styling carried on a text clip (mirrors the frontend TextStyle).
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportText {
	content: String,
	/// Bundled TTF filename (resolved to a resource path by the exporter).
	font_file: String,
	/// Font size as a percentage of the output frame height.
	size_pct: f64,
	/// Fill colour, hex #RRGGBB.
	color: String,
	/// "left" | "center" | "right".
	align: String,
	/// Outline width as a percentage of font size (0 = none).
	outline: f64,
	outline_color: String,
}

/// Resolved per-clip assets for a text overlay: a temp file holding the raw text
/// (sidesteps drawtext escaping) and the resolved bundled font path.
struct TextAsset {
	textfile: String,
	fontfile: String,
}

/// Output choices from the export dialog.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportSettings {
	/// Container + codec: "mp4-h264" | "mp4-h265" | "webm-vp9" | "mov-h264" | "gif".
	format: String,
	/// Target height: "source" | "2160" | "1440" | "1080" | "720" | "480".
	resolution: String,
	/// "high" | "medium" | "low".
	quality: String,
	/// H.264 backend: "cpu" (libx264) or "nvenc" (NVIDIA NVENC).
	#[serde(default = "default_encoder")]
	encoder: String,
}

fn default_encoder() -> String {
	"cpu".into()
}

/// Internal categories never contain driver messages, paths or stderr. The IPC
/// boundary exposes only these fixed messages, including cleanup diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExportError {
	InvalidRequest,
	Sidecar,
	Spawn,
	Event,
	MissingTermination,
	Process,
	OutputPreparation,
	OutputValidation,
	Publication,
}

impl ExportError {
	fn message(self) -> &'static str {
		match self {
			Self::InvalidRequest => "Invalid export request.",
			Self::Sidecar => "Required bundled media tool is unavailable.",
			Self::Spawn => "Could not start the media process.",
			Self::Event => "Media process communication failed.",
			Self::MissingTermination => "Media process did not report successful termination.",
			Self::Process => "Media process failed. Export was not published.",
			Self::OutputPreparation => "Could not prepare an isolated export output.",
			Self::OutputValidation => "Export output failed media validation.",
			Self::Publication => "Could not publish export. Existing destination was preserved.",
		}
	}
}

#[derive(Debug, PartialEq, Eq)]
enum NvencPreflight {
	Ready,
	NotReady,
	Indeterminate(ExportError),
}

/// Nonzero exit is a completed probe, distinct from an infrastructure failure.
#[derive(Debug, PartialEq, Eq)]
struct ProcessOutput {
	code: i32,
	stdout: Vec<u8>,
}

fn classify_preflight(
	nvenc: Result<ProcessOutput, ExportError>,
	control: impl FnOnce() -> Result<ProcessOutput, ExportError>,
) -> NvencPreflight {
	match nvenc {
		Ok(out) if out.code == 0 => NvencPreflight::Ready,
		Ok(_) => match control() {
			Ok(out) if out.code == 0 => NvencPreflight::NotReady,
			Ok(_) => NvencPreflight::Indeterminate(ExportError::Process),
			Err(e) => NvencPreflight::Indeterminate(e),
		},
		Err(e) => NvencPreflight::Indeterminate(e),
	}
}

fn synthetic_probe_args(encoder: &str) -> Vec<String> {
	let mut args: Vec<String> = [
		"-nostdin",
		"-f",
		"lavfi",
		"-i",
		"color=c=black:s=64x64:r=30",
		"-frames:v",
		"3",
		"-an",
	]
	.iter()
	.map(|s| s.to_string())
	.collect();
	// Fixed internal choices; identical source, pixel format and null sink.
	args.extend(h264_video_args(encoder, "medium").expect("fixed preflight encoder"));
	args.extend(["-f", "null", "-"].iter().map(|s| s.to_string()));
	args
}

/// Foundation only: deliberately not called by export selection/routing.
/// Three frames bound the workload, not wall time. A hung driver can still hang
/// this helper; timeout/kill/reap needs native qualification before M2 routing.
#[allow(dead_code)]
async fn nvenc_preflight(app: &AppHandle) -> NvencPreflight {
	let nvenc = run_media(app, "ffmpeg", synthetic_probe_args("nvenc"), None).await;
	if matches!(&nvenc, Ok(out) if out.code != 0) {
		let control = run_media(app, "ffmpeg", synthetic_probe_args("cpu"), None).await;
		classify_preflight(nvenc, || control)
	} else {
		classify_preflight(nvenc, || {
			unreachable!("control only follows nonzero NVENC exit")
		})
	}
}

fn output_muxer(format: &str) -> &'static str {
	match format {
		"gif" => "gif",
		"webm-vp9" => "webm",
		"mov-h264" => "mov",
		_ => "mp4", // public settings validated before preparing output
	}
}

fn output_extension(format: &str) -> &'static str {
	output_muxer(format)
}

static INVOCATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn invocation_token() -> String {
	let stamp = SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_nanos())
		.unwrap_or(0);
	format!(
		"{}-{stamp}-{}",
		std::process::id(),
		INVOCATION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
	)
}

/// Canonical paths cover relative paths and symlinks. Unix additionally covers
/// hard links by inode/device. No claim of universal alias/race detection.
fn reject_source_destination(destination: &Path, sources: &[&Path]) -> Result<(), ExportError> {
	fn location(path: &Path) -> std::io::Result<PathBuf> {
		std::fs::canonicalize(path).or_else(|_| {
			let parent = path
				.parent()
				.filter(|p| !p.as_os_str().is_empty())
				.unwrap_or(Path::new("."));
			std::fs::canonicalize(parent).map(|p| p.join(path.file_name().unwrap_or_default()))
		})
	}
	let dest = location(destination).map_err(|_| ExportError::OutputPreparation)?;
	for source in sources {
		if location(source).ok().as_ref() == Some(&dest) {
			return Err(ExportError::InvalidRequest);
		}
		#[cfg(unix)]
		if let (Ok(d), Ok(s)) = (std::fs::metadata(destination), std::fs::metadata(source)) {
			use std::os::unix::fs::MetadataExt;
			if d.dev() == s.dev() && d.ino() == s.ino() {
				return Err(ExportError::InvalidRequest);
			}
		}
	}
	Ok(())
}

/// Only paths successfully reserved with create_new belong to this invocation.
/// Never truncate a collision or delete a path merely because its name matches.
struct ExportFiles {
	destination: PathBuf,
	stage: PathBuf,
	stage_owned: bool,
	text: Vec<PathBuf>,
}

impl ExportFiles {
	fn reserve(destination: &Path, format: &str, token: &str) -> Result<Self, ExportError> {
		let parent = destination
			.parent()
			.filter(|p| !p.as_os_str().is_empty())
			.unwrap_or(Path::new("."));
		let parent = std::fs::canonicalize(parent).map_err(|_| ExportError::OutputPreparation)?;
		let name = destination.file_name().ok_or(ExportError::InvalidRequest)?;
		let destination = parent.join(name);
		if std::fs::symlink_metadata(&destination).is_ok_and(|m| !m.file_type().is_file()) {
			return Err(ExportError::OutputPreparation);
		}
		let stage = parent.join(format!(
			".katana-stage-{token}.{}",
			output_extension(format)
		));
		std::fs::OpenOptions::new()
			.write(true)
			.create_new(true)
			.open(&stage)
			.map_err(|_| ExportError::OutputPreparation)?;
		Ok(Self {
			destination,
			stage,
			stage_owned: true,
			text: Vec::new(),
		})
	}

	fn reserve_text(
		&mut self,
		content: &str,
		token: &str,
		index: usize,
	) -> Result<String, ExportError> {
		let path = std::env::temp_dir().join(format!("katana-text-{token}-{index}.txt"));
		let mut file = std::fs::OpenOptions::new()
			.write(true)
			.create_new(true)
			.open(&path)
			.map_err(|_| ExportError::OutputPreparation)?;
		self.text.push(path.clone()); // own even if write fails
		file.write_all(content.as_bytes())
			.map_err(|_| ExportError::OutputPreparation)?;
		Ok(path.to_string_lossy().into_owned())
	}

	fn validate_file(&self) -> Result<(), ExportError> {
		let metadata =
			std::fs::symlink_metadata(&self.stage).map_err(|_| ExportError::OutputValidation)?;
		if metadata.file_type().is_file() && metadata.len() > 0 {
			Ok(())
		} else {
			Err(ExportError::OutputValidation)
		}
	}

	fn cleanup(&mut self) -> bool {
		fn remove(path: &Path) -> bool {
			match std::fs::remove_file(path) {
				Ok(()) => true,
				Err(e) => e.kind() == std::io::ErrorKind::NotFound,
			}
		}
		if self.stage_owned && remove(&self.stage) {
			self.stage_owned = false;
		}
		self.text.retain(|path| !remove(path));
		self.stage_owned || !self.text.is_empty()
	}

	/// One rename boundary; never delete destination or use copy-overwrite.
	/// Native Windows/locking/network filesystem behavior must be qualified.
	fn finish(
		&self,
		process: Result<(), ExportError>,
		validate: impl FnOnce() -> Result<(), ExportError>,
		publish: impl FnOnce(&Path, &Path) -> std::io::Result<()>,
		complete: impl FnOnce(),
	) -> Result<(), ExportError> {
		process?;
		self.validate_file()?;
		validate()?;
		publish(&self.stage, &self.destination).map_err(|_| ExportError::Publication)?;
		complete();
		Ok(())
	}
}

impl Drop for ExportFiles {
	fn drop(&mut self) {
		self.cleanup();
	}
}

fn finish_cleanup(files: &mut ExportFiles, result: Result<(), ExportError>) -> Result<(), String> {
	let cleanup_failed = files.cleanup();
	match result {
		Err(e) if cleanup_failed => Err(format!(
			"{} Temporary file cleanup also failed.",
			e.message()
		)),
		Err(e) => Err(e.message().into()),
		Ok(()) => {
			if cleanup_failed {
				log::warn!("Export temporary file cleanup failed after publication.");
			}
			Ok(())
		}
	}
}

#[derive(Debug)]
struct MediaExpectation {
	format: String,
	dimensions: Option<(u32, u32)>,
	video: String,
	audio: Option<String>,
}

impl MediaExpectation {
	fn composite(settings: &ExportSettings, aspect: &str, base: Option<(u32, u32)>) -> Self {
		let (w, h) = canvas_dims(aspect, base);
		let (w, h) = apply_resolution(w, h, &settings.resolution);
		let (video, audio) = match settings.format.as_str() {
			"gif" => ("gif", None),
			"webm-vp9" => ("vp9", Some("opus")),
			"mp4-h265" => ("hevc", Some("aac")),
			_ => ("h264", Some("aac")),
		};
		Self {
			format: settings.format.clone(),
			dimensions: Some((w as u32, h as u32)),
			video: video.into(),
			audio: audio.map(String::from),
		}
	}

	fn copied(
		format: &str,
		dimensions: Option<(u32, u32)>,
		video: &str,
		audio: Option<&str>,
	) -> Self {
		Self {
			format: format.into(),
			dimensions,
			video: video.into(),
			audio: audio.map(String::from),
		}
	}

	fn validate(&self, bytes: &[u8]) -> Result<(), ExportError> {
		let value: serde_json::Value =
			serde_json::from_slice(bytes).map_err(|_| ExportError::OutputValidation)?;
		let streams = value
			.get("streams")
			.and_then(|s| s.as_array())
			.ok_or(ExportError::OutputValidation)?;
		let container = value
			.pointer("/format/format_name")
			.and_then(|v| v.as_str())
			.ok_or(ExportError::OutputValidation)?;
		let expected = output_muxer(&self.format);
		// ffprobe reports MOV/MP4 and Matroska/WebM as shared demuxer families.
		if !container.split(',').any(|s| s == expected) {
			return Err(ExportError::OutputValidation);
		}
		let video = streams
			.iter()
			.find(|v| v.get("codec_type").and_then(|v| v.as_str()) == Some("video"))
			.ok_or(ExportError::OutputValidation)?;
		let w = video.get("width").and_then(|v| v.as_u64()).unwrap_or(0);
		let h = video.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
		if w == 0
			|| h == 0 || video.get("codec_name").and_then(|v| v.as_str())
			!= Some(self.video.as_str())
		{
			return Err(ExportError::OutputValidation);
		}
		if self
			.dimensions
			.is_some_and(|(ew, eh)| w != ew as u64 || h != eh as u64)
		{
			return Err(ExportError::OutputValidation);
		}
		if let Some(codec) = &self.audio {
			if !streams.iter().any(|s| {
				s.get("codec_type").and_then(|v| v.as_str()) == Some("audio")
					&& s.get("codec_name").and_then(|v| v.as_str()) == Some(codec.as_str())
			}) {
				return Err(ExportError::OutputValidation);
			}
		}
		Ok(())
	}
}

/// Frames per second for GIF output.
const GIF_FPS: u32 = 15;

impl ExportClip {
	fn source_span(&self) -> f64 {
		(self.out_point - self.in_point).max(0.01)
	}
	fn timeline_dur(&self) -> f64 {
		self.source_span() / self.speed.max(0.01)
	}
	fn timeline_end(&self) -> f64 {
		self.start + self.timeline_dur()
	}
	fn is_video(&self) -> bool {
		self.kind == "video"
	}
	fn is_text(&self) -> bool {
		self.kind == "text"
	}
}

fn format_dims(format: &str) -> Option<(u32, u32)> {
	match format {
		"16:9" => Some((1920, 1080)),
		"9:16" => Some((1080, 1920)),
		"1:1" => Some((1080, 1080)),
		_ => None,
	}
}

/// Round down to the nearest even number (libx264/yuv420p needs even dims).
fn even(v: i64) -> i64 {
	(v - (v % 2)).max(2)
}

/// Target output height for a resolution preset (None = keep source/canvas).
fn target_height(res: &str) -> Option<i64> {
	match res {
		"2160" => Some(2160),
		"1440" => Some(1440),
		"1080" => Some(1080),
		"720" => Some(720),
		"480" => Some(480),
		_ => None,
	}
}

/// Scale canvas dims to a resolution preset, preserving aspect ratio (even dims).
fn apply_resolution(cw: i64, ch: i64, res: &str) -> (i64, i64) {
	match target_height(res) {
		Some(th) => {
			let s = th as f64 / ch as f64;
			(even((cw as f64 * s).round() as i64), even(th))
		}
		None => (even(cw), even(ch)),
	}
}

/// CRF for a codec family at a quality level (lower = better quality).
fn crf(codec: &str, quality: &str) -> &'static str {
	match (codec, quality) {
		("x265", "high") => "20",
		("x265", "medium") => "25",
		("x265", "low") => "30",
		("vp9", "high") => "24",
		("vp9", "medium") => "31",
		("vp9", "low") => "37",
		(_, "medium") => "23",
		(_, "low") => "28",
		_ => "18", // x264 high / default
	}
}

/// Constant-quality target for NVENC VBR (lower = better quality).
fn nvenc_cq(quality: &str) -> &'static str {
	match quality {
		"medium" => "23",
		"low" => "28",
		_ => "19",
	}
}

/// Validate the deliberately narrow M1 encoder contract. Other formats retain
/// their existing software encoders until a later milestone explicitly expands it.
fn validate_encoder(format: &str, encoder: &str) -> Result<(), String> {
	match (format, encoder) {
		(_, "cpu") | ("mp4-h264", "nvenc") => Ok(()),
		(_, "nvenc") => Err("NVIDIA NVENC is only available for MP4 H.264 export.".into()),
		(_, _) => Err("Unsupported H.264 encoder.".into()),
	}
}

fn validate_export_settings(settings: &ExportSettings) -> Result<(), String> {
	match settings.format.as_str() {
		"mp4-h264" | "mp4-h265" | "webm-vp9" | "mov-h264" | "gif" => {}
		_ => return Err("Unsupported export format.".into()),
	}
	validate_encoder(&settings.format, &settings.encoder)
}

/// Backend-specific H.264 flags. Keep x264-only CRF/preset flags out of NVENC.
fn h264_video_args(encoder: &str, quality: &str) -> Result<Vec<String>, String> {
	match encoder {
		"cpu" => Ok(vec![
			"-c:v".into(),
			"libx264".into(),
			"-preset".into(),
			"veryfast".into(),
			"-crf".into(),
			crf("x264", quality).into(),
			"-pix_fmt".into(),
			"yuv420p".into(),
		]),
		"nvenc" => Ok(vec![
			"-c:v".into(),
			"h264_nvenc".into(),
			"-preset".into(),
			"p5".into(),
			"-tune".into(),
			"hq".into(),
			"-rc".into(),
			"vbr".into(),
			"-cq".into(),
			nvenc_cq(quality).into(),
			"-b:v".into(),
			"0".into(),
			"-pix_fmt".into(),
			"yuv420p".into(),
		]),
		_ => Err("Unsupported H.264 encoder.".into()),
	}
}

/// atempo only accepts 0.5..2.0 per stage, so chain stages for extreme speeds.
fn atempo_chain(speed: f64) -> String {
	let mut parts: Vec<String> = Vec::new();
	let mut s = speed;
	while s > 2.0 {
		parts.push("atempo=2.0".into());
		s /= 2.0;
	}
	while s < 0.5 {
		parts.push("atempo=0.5".into());
		s *= 2.0;
	}
	parts.push(format!("atempo={s:.4}"));
	parts.join(",")
}

/// Does this media file carry at least one audio stream? (via ffprobe sidecar)
async fn probe_has_audio(app: &AppHandle, path: &str) -> bool {
	let Ok(cmd) = app.shell().sidecar("ffprobe") else {
		return false;
	};
	let args = [
		"-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0",
		path,
	];
	match cmd.args(args).output().await {
		Ok(out) => out.status.success() && !String::from_utf8_lossy(&out.stdout).trim().is_empty(),
		Err(_) => false,
	}
}

/// Probe a clip's pixel dimensions (via ffprobe sidecar) for "original" canvas.
async fn probe_dims(app: &AppHandle, path: &str) -> Option<(u32, u32)> {
	let cmd = app.shell().sidecar("ffprobe").ok()?;
	let args = [
		"-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
		"csv=s=,:p=0", path,
	];
	let out = cmd.args(args).output().await.ok()?;
	if !out.status.success() {
		return None;
	}
	let text = String::from_utf8_lossy(&out.stdout);
	let mut it = text.trim().split(',');
	let w: u32 = it.next()?.trim().parse().ok()?;
	let h: u32 = it.next()?.trim().parse().ok()?;
	if w == 0 || h == 0 {
		None
	} else {
		Some((w, h))
	}
}

/// Parse an ffprobe rational ("30000/1001", "30/1") to fps; None if degenerate.
fn parse_rational(s: &str) -> Option<f64> {
	let mut it = s.trim().split('/');
	let n: f64 = it.next()?.trim().parse().ok()?;
	let d: f64 = it.next().map(|x| x.trim().parse().unwrap_or(1.0)).unwrap_or(1.0);
	if n <= 0.0 || d <= 0.0 {
		return None;
	}
	Some(n / d)
}

/// Probe a video's frame rate (fps) via the ffprobe sidecar. None if unknown.
#[tauri::command]
pub async fn probe_fps(app: AppHandle, path: String) -> Option<f64> {
	let cmd = app.shell().sidecar("ffprobe").ok()?;
	let args = [
		"-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of",
		"csv=p=0", &path,
	];
	let out = cmd.args(args).output().await.ok()?;
	if !out.status.success() {
		return None;
	}
	parse_rational(&String::from_utf8_lossy(&out.stdout))
}

/// Output canvas size: the chosen aspect format, or the probed base-clip size.
fn canvas_dims(aspect: &str, base_dims: Option<(u32, u32)>) -> (i64, i64) {
	if let Some((w, h)) = format_dims(aspect) {
		return (w as i64, h as i64);
	}
	let (w, h) = base_dims.unwrap_or((1920, 1080));
	(even(w as i64), even(h as i64))
}

/// Compositing placement of a clip on the canvas: scaled size (even) + overlay
/// offset (may be negative when the clip is panned/zoomed past an edge).
fn placement(cw: i64, ch: i64, c: &ExportClip) -> (i64, i64, i64, i64) {
	let (cwf, chf) = (cw as f64, ch as f64);
	let canvas_ar = cwf / chf;
	let ar = if c.aspect_ratio > 0.0 { c.aspect_ratio } else { canvas_ar };
	// Contain-fit baseline, then the clip's relative scale.
	let (fit_w, fit_h) = if ar > canvas_ar {
		(cwf, cwf / ar)
	} else {
		(chf * ar, chf)
	};
	let dw = even((fit_w * c.scale).round() as i64);
	let dh = even((fit_h * c.scale).round() as i64);
	let cx = cwf / 2.0 + c.x * cwf;
	let cy = chf / 2.0 + c.y * chf;
	let ox = (cx - dw as f64 / 2.0).round() as i64;
	let oy = (cy - dh as f64 / 2.0).round() as i64;
	(dw, dh, ox, oy)
}

/// Single-quote a path for the filtergraph (forward slashes; protects spaces).
/// The drive-letter ':' must be backslash-escaped even inside single quotes, or
/// drawtext's option parser splits on it (Windows `C:/…` → "No option name").
/// Non-Windows paths have no ':' so this is a no-op there. Temp/resource paths
/// won't contain single quotes.
fn ff_path(p: &str) -> String {
	format!("'{}'", p.replace('\\', "/").replace(':', "\\:"))
}

/// Hex #RRGGBB -> ffmpeg 0xRRGGBB (falls back to white if malformed).
fn ff_color(hex: &str) -> String {
	let h = hex.trim_start_matches('#');
	if h.len() == 6 && h.bytes().all(|b| b.is_ascii_hexdigit()) {
		format!("0x{h}")
	} else {
		"white".into()
	}
}

/// A `drawtext` filter for one text overlay: positioned + sized like the preview
/// (size = % of frame height × transform scale; align + center offset), with an
/// optional outline and a time-driven alpha ramp for fades. Returns `null` (a
/// pass-through) if the clip carries no text or its font/text files are missing.
fn drawtext_filter(c: &ExportClip, cw: i64, ch: i64, asset: Option<&TextAsset>) -> String {
	let (t, asset) = match (&c.text, asset) {
		(Some(t), Some(a)) => (t, a),
		_ => return "null".into(),
	};
	let fs = ((t.size_pct / 100.0) * ch as f64 * c.scale).round().max(1.0) as i64;
	let ox = (c.x * cw as f64).round() as i64;
	let oy = (c.y * ch as f64).round() as i64;
	let xexpr = match t.align.as_str() {
		"left" => format!("({ox})"),
		"right" => format!("w-text_w+({ox})"),
		_ => format!("(w-text_w)/2+({ox})"),
	};
	let yexpr = format!("(h-text_h)/2+({oy})");

	let mut parts = vec![
		format!("drawtext=fontfile={}", ff_path(&asset.fontfile)),
		format!("textfile={}", ff_path(&asset.textfile)),
		format!("fontsize={fs}"),
		format!("fontcolor={}", ff_color(&t.color)),
		format!("x={xexpr}"),
		format!("y={yexpr}"),
		"expansion=none".into(),
	];

	let bw = ((t.outline / 100.0) * fs as f64).round() as i64;
	if bw > 0 {
		parts.push(format!("borderw={bw}"));
		parts.push(format!("bordercolor={}", ff_color(&t.outline_color)));
	}

	// Fade in/out via a piecewise alpha ramp (gated to the clip window by enable).
	let (s, e, fin, fout) = (c.start, c.timeline_end(), c.fade_in, c.fade_out);
	let alpha = if fin > 0.0 && fout > 0.0 {
		Some(format!(
			"if(lt(t,{a:.6}),(t-{s:.6})/{fin:.4},if(gt(t,{b:.6}),({e:.6}-t)/{fout:.4},1))",
			a = s + fin,
			b = e - fout
		))
	} else if fin > 0.0 {
		Some(format!("if(lt(t,{a:.6}),(t-{s:.6})/{fin:.4},1)", a = s + fin))
	} else if fout > 0.0 {
		Some(format!("if(gt(t,{b:.6}),({e:.6}-t)/{fout:.4},1)", b = e - fout))
	} else {
		None
	};
	if let Some(a) = alpha {
		parts.push(format!("alpha='{a}'"));
	}

	parts.push(format!("enable='between(t,{s:.6},{e:.6})'"));
	parts.join(":")
}

/// Build the ffmpeg argument vector. Pure: all probing is done by the caller and
/// passed in via `order`, `audio_flags`, `base_dims` and `text_assets`.
fn build_args(
	clips: &[ExportClip],
	order: &[usize],
	aspect: &str,
	settings: &ExportSettings,
	audio_flags: &[bool],
	base_dims: Option<(u32, u32)>,
	text_assets: &[Option<TextAsset>],
	fps: f64,
	output: &str,
) -> Result<(Vec<String>, f64), String> {
	let total: f64 = clips.iter().map(|c| c.timeline_end()).fold(0.0, f64::max);
	// Project frame rate (fastest clip); slower clips are frame-held to it.
	let fps = if (1.0..=240.0).contains(&fps) { fps } else { 30.0 };

	let (cw, ch) = canvas_dims(aspect, base_dims);
	let (cw, ch) = apply_resolution(cw, ch, &settings.resolution);
	let is_gif = settings.format == "gif";

	let mut args: Vec<String> = vec!["-y".into()];
	// Trimmed inputs, one per media clip. Text clips carry no input stream, so
	// input indices are decoupled from clip indices via this map.
	let mut input_index: Vec<Option<usize>> = vec![None; clips.len()];
	let mut ii = 0usize;
	for (i, c) in clips.iter().enumerate() {
		if c.is_text() {
			continue;
		}
		args.push("-ss".into());
		args.push(format!("{:.6}", c.in_point));
		args.push("-t".into());
		args.push(format!("{:.6}", c.source_span()));
		args.push("-i".into());
		args.push(c.path.clone());
		input_index[i] = Some(ii);
		ii += 1;
	}

	// Filtergraph assembled as discrete chains, joined by ';'.
	let mut chains: Vec<String> = Vec::new();

	// Text overlays draw on top of the composited video, in z-order (track asc).
	let mut text_order: Vec<usize> = (0..clips.len()).filter(|&i| clips[i].is_text()).collect();
	text_order.sort_by(|&a, &b| {
		clips[a].track.cmp(&clips[b].track).then(
			clips[a]
				.start
				.partial_cmp(&clips[b].start)
				.unwrap_or(std::cmp::Ordering::Equal),
		)
	});
	let has_text = !text_order.is_empty();

	// Black base canvas spanning the whole timeline. With no video and no text the
	// base itself is the output (audio-only export = black frame + audio).
	let has_video = !order.is_empty();
	let base = if has_video || has_text { "bg" } else { "outv" };
	chains.push(format!("color=c=black:s={cw}x{ch}:r={fps:.5}:d={total:.6}[{base}]"));

	// Per-clip video: speed, time-shift to start, scale to placement size, and
	// optional visual fade in/out (alpha so it blends with the layers below).
	let places: Vec<(i64, i64, i64, i64)> = clips.iter().map(|c| placement(cw, ch, c)).collect();
	for (i, c) in clips.iter().enumerate() {
		if !c.is_video() {
			continue;
		}
		let speed = c.speed.max(0.01);
		let (dw, dh, _, _) = places[i];
		let src = input_index[i].unwrap_or(0);
		let mut chain = format!(
			"[{src}:v]setpts=(PTS-STARTPTS)/{speed:.6}+{start:.6}/TB,scale={dw}:{dh},setsar=1",
			start = c.start
		);
		if c.fade_in > 0.0 || c.fade_out > 0.0 {
			chain.push_str(",format=yuva420p");
			if c.fade_in > 0.0 {
				chain.push_str(&format!(
					",fade=t=in:st={:.6}:d={:.4}:alpha=1",
					c.start, c.fade_in
				));
			}
			if c.fade_out > 0.0 {
				let st = (c.timeline_end() - c.fade_out).max(0.0);
				chain.push_str(&format!(",fade=t=out:st={st:.6}:d={:.4}:alpha=1", c.fade_out));
			}
		}
		chain.push_str(&format!("[v{i}]"));
		chains.push(chain);
	}

	// Overlay chain in z-order (video clips only); each composites in its window.
	// The composited video lands on `outv`, unless text follows (then `vcomp`).
	let video_out = if has_video {
		let mut last = "bg".to_string();
		for (k, &i) in order.iter().enumerate() {
			let c = &clips[i];
			let (_, _, ox, oy) = places[i];
			let out_label = if k == order.len() - 1 {
				if has_text { "vcomp".to_string() } else { "outv".to_string() }
			} else {
				format!("ov{i}")
			};
			chains.push(format!(
				"[{last}][v{i}]overlay=x={ox}:y={oy}:eof_action=pass:enable='between(t,{s:.6},{e:.6})'[{out_label}]",
				s = c.start,
				e = c.timeline_end()
			));
			last = out_label;
		}
		last
	} else {
		// No video: text (if any) draws straight onto the black base canvas.
		"bg".to_string()
	};

	// Text overlays: a drawtext per clip, chained in z-order, producing `outv`.
	if has_text {
		let mut last = video_out;
		for (j, &i) in text_order.iter().enumerate() {
			let c = &clips[i];
			let out_label = if j == text_order.len() - 1 {
				"outv".to_string()
			} else {
				format!("tx{i}")
			};
			let dt = drawtext_filter(c, cw, ch, text_assets[i].as_ref());
			chains.push(format!("[{last}]{dt}[{out_label}]"));
			last = out_label;
		}
	}

	// Audio (skipped for GIF). Per-clip: speed, volume, fade, delay to start;
	// muted/silent clips contribute nothing and are dropped from the mix.
	if !is_gif {
		let mut alabels: Vec<String> = Vec::new();
		for (i, c) in clips.iter().enumerate() {
			if c.muted || !audio_flags[i] {
				continue;
			}
			let speed = c.speed.max(0.01);
			let src = input_index[i].unwrap_or(0);
			let mut chain = format!(
				"[{src}:a]asetpts=PTS-STARTPTS,{},volume={:.4}",
				atempo_chain(speed),
				c.volume
			);
			if c.fade_in > 0.0 {
				chain.push_str(&format!(",afade=t=in:st=0:d={:.4}", c.fade_in));
			}
			if c.fade_out > 0.0 {
				let st = (c.timeline_dur() - c.fade_out).max(0.0);
				chain.push_str(&format!(",afade=t=out:st={st:.4}:d={:.4}", c.fade_out));
			}
			let ms = (c.start * 1000.0).round() as i64;
			if ms > 0 {
				chain.push_str(&format!(",adelay={ms}:all=1"));
			}
			chain.push_str(&format!("[a{i}]"));
			chains.push(chain);
			alabels.push(format!("[a{i}]"));
		}
		if alabels.is_empty() {
			chains.push(format!("anullsrc=r=44100:cl=stereo,atrim=0:{total:.6}[outa]"));
		} else if alabels.len() == 1 {
			chains.push(format!("{}anull[outa]", alabels[0]));
		} else {
			chains.push(format!(
				"{}amix=inputs={}:normalize=0:dropout_transition=0[outa]",
				alabels.join(""),
				alabels.len()
			));
		}
	} else {
		// GIF: build an optimized palette from the composited video.
		chains.push(format!("[outv]fps={GIF_FPS},split[gv][gp]"));
		chains.push("[gp]palettegen=stats_mode=diff[pal]".into());
		chains.push("[gv][pal]paletteuse=dither=bayer[gifout]".into());
	}

	args.push("-filter_complex".into());
	args.push(chains.join(";"));

	// Encoder + muxer for the chosen format.
	let q = settings.quality.as_str();
	let tail: Vec<&str> = match settings.format.as_str() {
		"gif" => vec!["-map", "[gifout]", "-loop", "0"],
		"mp4-h265" => vec![
			"-map", "[outv]", "-map", "[outa]", "-c:v", "libx265", "-preset", "veryfast", "-crf",
			crf("x265", q), "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "192k",
			"-movflags", "+faststart",
		],
		"webm-vp9" => vec![
			"-map", "[outv]", "-map", "[outa]", "-c:v", "libvpx-vp9", "-crf", crf("vp9", q), "-b:v",
			"0", "-pix_fmt", "yuv420p", "-c:a", "libopus", "-b:a", "128k",
		],
		"mov-h264" => vec![
			"-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-preset", "veryfast", "-crf",
			crf("x264", q), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags",
			"+faststart",
		],
		"mp4-h264" => Vec::new(),
		_ => return Err("Unsupported export format.".into()),
	};
	if settings.format == "mp4-h264" {
		args.extend(["-map", "[outv]", "-map", "[outa]"].iter().map(|s| s.to_string()));
		args.extend(h264_video_args(&settings.encoder, q)?);
		args.extend(
			["-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart"]
				.iter()
				.map(|s| s.to_string()),
		);
	} else {
		args.extend(tail.iter().map(|s| s.to_string()));
	}
	// Machine-readable progress on stdout.
	args.extend(["-progress", "pipe:1", "-nostats"].iter().map(|s| s.to_string()));
	args.extend(["-f".into(), output_muxer(&settings.format).into()]);
	args.push(output.to_string());
	Ok((args, total))
}

/// Probe a single stream's codec name (e.g. "h264", "aac"); None if absent.
async fn probe_codec(app: &AppHandle, path: &str, stream: &str) -> Option<String> {
	let cmd = app.shell().sidecar("ffprobe").ok()?;
	let args = [
		"-v", "error", "-select_streams", stream, "-show_entries", "stream=codec_name", "-of",
		"csv=p=0", path,
	];
	let out = cmd.args(args).output().await.ok()?;
	if !out.status.success() {
		return None;
	}
	let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
	if s.is_empty() {
		None
	} else {
		Some(s)
	}
}

/// Structural eligibility for a lossless stream copy: a single, untouched,
/// timeline-origin video clip exported at source size/format. Returns the clip.
fn copy_candidate<'a>(
	clips: &'a [ExportClip],
	aspect: &str,
	settings: &ExportSettings,
) -> Option<&'a ExportClip> {
	if settings.format == "gif" || settings.resolution != "source" || aspect != "original" {
		return None;
	}
	if clips.len() != 1 {
		return None;
	}
	let c = &clips[0];
	let eps = 1e-6;
	if !c.is_video()
		|| c.muted
		|| c.start.abs() > eps
		|| (c.speed - 1.0).abs() > eps
		|| (c.volume - 1.0).abs() > eps
		|| c.fade_in > eps
		|| c.fade_out > eps
		|| c.x.abs() > eps
		|| c.y.abs() > eps
		|| (c.scale - 1.0).abs() > eps
	{
		return None;
	}
	Some(c)
}

/// Can the source streams be copied (not re-encoded) into the chosen format?
fn copy_compatible(format: &str, vcodec: &str, acodec: Option<&str>) -> bool {
	let audio_ok = |allowed: &[&str]| acodec.map_or(true, |a| allowed.contains(&a));
	match format {
		"mp4-h264" | "mov-h264" => vcodec == "h264" && audio_ok(&["aac", "mp3"]),
		"mp4-h265" => vcodec == "hevc" && audio_ok(&["aac", "mp3"]),
		"webm-vp9" => (vcodec == "vp9" || vcodec == "vp8") && audio_ok(&["opus", "vorbis"]),
		_ => false,
	}
}

/// Lossless stream-copy args: trim with `-c copy`, no re-encode.
fn build_copy_args(c: &ExportClip, format: &str, output: &str) -> (Vec<String>, f64) {
	let span = c.source_span();
	let mut args: Vec<String> = vec![
		"-y".into(),
		"-ss".into(),
		format!("{:.6}", c.in_point),
		"-i".into(),
		c.path.clone(),
		"-t".into(),
		format!("{span:.6}"),
		"-map".into(),
		"0:v:0".into(),
		"-map".into(),
		"0:a:0?".into(),
		"-c".into(),
		"copy".into(),
		"-avoid_negative_ts".into(),
		"make_zero".into(),
	];
	if matches!(format, "mp4-h264" | "mp4-h265" | "mov-h264") {
		args.push("-movflags".into());
		args.push("+faststart".into());
	}
	args.extend(["-progress", "pipe:1", "-nostats"].iter().map(|s| s.to_string()));
	args.extend(["-f".into(), output_muxer(format).into()]);
	args.push(output.to_string());
	(args, span)
}

/// Render the timeline to a single output file, emitting `export:progress`
/// (0..1) as it runs. Uses a lossless stream copy when the timeline is a single
/// untouched trim; otherwise the full compositing re-encode. Sidecars from bundle.
#[tauri::command]
pub async fn export_video(
	app: AppHandle,
	clips: Vec<ExportClip>,
	aspect: String,
	settings: ExportSettings,
	fps: f64,
	output: String,
) -> Result<(), String> {
	if clips.is_empty() {
		return Err("Nothing to export: the timeline is empty.".into());
	}
	validate_export_settings(&settings)
		.map_err(|_| ExportError::InvalidRequest.message().to_string())?;
	let destination = Path::new(&output);
	let sources: Vec<&Path> = clips
		.iter()
		.filter(|c| !c.is_text())
		.map(|c| Path::new(&c.path))
		.collect();
	reject_source_destination(destination, &sources).map_err(|e| e.message().to_string())?;
	let token = invocation_token();
	let mut files = ExportFiles::reserve(destination, &settings.format, &token)
		.map_err(|e| e.message().to_string())?;
	let stage = match files.stage.to_str().map(str::to_owned) {
		Some(stage) => stage,
		None => return finish_cleanup(&mut files, Err(ExportError::OutputPreparation)),
	};

	// Lossless stream-copy fastpath (single trimmed clip, compatible codecs).
	if let Some(c) = copy_candidate(&clips, &aspect, &settings) {
		let vcodec = probe_codec(&app, &c.path, "v:0").await;
		let acodec = probe_codec(&app, &c.path, "a:0").await;
		if let Some(vc) = vcodec.as_deref() {
			if copy_compatible(&settings.format, vc, acodec.as_deref()) {
				let dims = probe_dims(&app, &c.path).await;
				let expected =
					MediaExpectation::copied(&settings.format, dims, vc, acodec.as_deref());
				let (args, total) = build_copy_args(c, &settings.format, &stage);
				let result = run_export(&app, &files, args, total, &expected).await;
				return finish_cleanup(&mut files, result);
			}
		}
	}

	// Composite bottom-to-top: video clips only, lower track first, ties by start.
	let mut order: Vec<usize> = (0..clips.len()).filter(|&i| clips[i].is_video()).collect();
	order.sort_by(|&a, &b| {
		clips[a].track.cmp(&clips[b].track).then(
			clips[a]
				.start
				.partial_cmp(&clips[b].start)
				.unwrap_or(std::cmp::Ordering::Equal),
		)
	});

	// Probe up front (async) so build_args stays pure. Text clips have no media.
	let mut audio_flags: Vec<bool> = Vec::with_capacity(clips.len());
	for c in &clips {
		audio_flags.push(if c.is_text() {
			false
		} else {
			probe_has_audio(&app, &c.path).await
		});
	}
	let base_dims = if format_dims(&aspect).is_none() && !order.is_empty() {
		probe_dims(&app, &clips[order[0]].path).await
	} else {
		None
	};

	// Resolve text-overlay assets: the bundled font path + a temp file holding the
	// raw text (so drawtext needs no escaping). Cleaned up after the run.
	let mut text_assets: Vec<Option<TextAsset>> = Vec::with_capacity(clips.len());
	for (i, c) in clips.iter().enumerate() {
		let asset = match &c.text {
			Some(t) => {
				let fontfile = app
					.path()
					.resolve(
						format!("fonts/{}", t.font_file),
						tauri::path::BaseDirectory::Resource,
					)
					.ok()
					.map(|p| p.to_string_lossy().to_string());
				match fontfile {
					Some(fontfile) => match files.reserve_text(&t.content, &token, i) {
						Ok(textfile) => Some(TextAsset { textfile, fontfile }),
						Err(e) => return finish_cleanup(&mut files, Err(e)),
					},
					None => None, // retain existing missing-font behavior
				}
			}
			None => None,
		};
		text_assets.push(asset);
	}

	let built = build_args(
		&clips,
		&order,
		&aspect,
		&settings,
		&audio_flags,
		base_dims,
		&text_assets,
		fps,
		&stage,
	);
	let (args, total) = match built {
		Ok(plan) => plan,
		Err(_) => return finish_cleanup(&mut files, Err(ExportError::InvalidRequest)),
	};
	let expected = MediaExpectation::composite(&settings, &aspect, base_dims);
	let result = run_export(&app, &files, args, total, &expected).await;
	finish_cleanup(&mut files, result)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn process(code: i32) -> Result<ProcessOutput, ExportError> {
		Ok(ProcessOutput {
			code,
			stdout: Vec::new(),
		})
	}

	struct Fixture(PathBuf);
	impl Fixture {
		fn new() -> Self {
			let path = std::env::temp_dir().join(format!("katana-test-{}", invocation_token()));
			std::fs::create_dir(&path).unwrap();
			Self(path)
		}
		fn destination(&self) -> PathBuf {
			self.0.join("destination.mp4")
		}
		fn files(&self) -> ExportFiles {
			std::fs::write(self.destination(), b"sentinel").unwrap();
			let files =
				ExportFiles::reserve(&self.destination(), "mp4-h264", &invocation_token()).unwrap();
			std::fs::write(&files.stage, b"candidate").unwrap();
			files
		}
		fn sentinel(&self) {
			assert_eq!(std::fs::read(self.destination()).unwrap(), b"sentinel");
		}
	}
	impl Drop for Fixture {
		fn drop(&mut self) {
			std::fs::remove_dir_all(&self.0).unwrap();
		}
	}

	fn settings(format: &str) -> ExportSettings {
		ExportSettings {
			format: format.into(),
			resolution: "source".into(),
			quality: "high".into(),
			encoder: "cpu".into(),
		}
	}

	fn clip() -> ExportClip {
		ExportClip {
			kind: "video".into(),
			path: "source.mp4".into(),
			in_point: 0.0,
			out_point: 1.0,
			speed: 1.0,
			volume: 1.0,
			muted: false,
			fade_in: 0.0,
			fade_out: 0.0,
			start: 0.0,
			track: 0,
			x: 0.0,
			y: 0.0,
			scale: 1.0,
			aspect_ratio: 16.0 / 9.0,
			text: None,
		}
	}

	fn media(video: &str, audio: Option<&str>, container: &str, w: u32, h: u32) -> Vec<u8> {
		let mut streams = vec![
			serde_json::json!({"codec_type":"video", "codec_name":video, "width":w, "height":h}),
		];
		if let Some(audio) = audio {
			streams.push(serde_json::json!({"codec_type":"audio", "codec_name":audio}));
		}
		serde_json::to_vec(
			&serde_json::json!({"streams":streams, "format":{"format_name":container}}),
		)
		.unwrap()
	}

	#[test]
	fn nvenc_zero_is_ready_without_control() {
		assert_eq!(
			classify_preflight(process(0), || panic!("unexpected control")),
			NvencPreflight::Ready
		);
	}

	#[test]
	fn nvenc_nonzero_with_successful_control_is_not_ready() {
		assert_eq!(
			classify_preflight(process(7), || process(0)),
			NvencPreflight::NotReady
		);
	}

	#[test]
	fn probe_infrastructure_and_control_failures_are_indeterminate() {
		for error in [
			ExportError::Sidecar,
			ExportError::Spawn,
			ExportError::Event,
			ExportError::MissingTermination,
		] {
			assert_eq!(
				classify_preflight(Err(error), || panic!("control after infrastructure error")),
				NvencPreflight::Indeterminate(error)
			);
			assert_eq!(
				classify_preflight(process(1), || Err(error)),
				NvencPreflight::Indeterminate(error)
			);
		}
		assert_eq!(
			classify_preflight(process(1), || process(1)),
			NvencPreflight::Indeterminate(ExportError::Process)
		);
	}

	fn event_outcome(stderr: &str, code: i32) -> Result<ProcessOutput, ExportError> {
		let mut events = ProcessEvents::default();
		events.accept(&CommandEvent::Stderr(stderr.as_bytes().to_vec()), false);
		events.accept(
			&CommandEvent::Terminated(tauri_plugin_shell::process::TerminatedPayload {
				code: Some(code),
				signal: None,
			}),
			false,
		);
		events.finish()
	}

	#[test]
	fn stderr_content_cannot_change_preflight_classification() {
		for code in [0, 1, 9] {
			let a = classify_preflight(event_outcome("NVENC driver failed", code), || process(0));
			let b = classify_preflight(
				event_outcome("input/filter/audio/credential/path", code),
				|| process(0),
			);
			assert_eq!(a, b);
		}
	}

	#[test]
	fn event_error_and_missing_termination_never_succeed() {
		assert_eq!(
			ProcessEvents::default().finish(),
			Err(ExportError::MissingTermination)
		);
		let mut events = ProcessEvents::default();
		events.accept(
			&CommandEvent::Error("raw sensitive exception".into()),
			false,
		);
		events.accept(
			&CommandEvent::Terminated(tauri_plugin_shell::process::TerminatedPayload {
				code: Some(0),
				signal: None,
			}),
			false,
		);
		assert_eq!(events.finish(), Err(ExportError::Event));
		let mut events = ProcessEvents::default();
		events.accept(
			&CommandEvent::Terminated(tauri_plugin_shell::process::TerminatedPayload {
				code: None,
				signal: Some(9),
			}),
			false,
		);
		assert_eq!(events.finish(), Err(ExportError::MissingTermination));
	}

	#[test]
	fn synthetic_probe_is_fixed_and_has_no_user_output_or_audio() {
		let nvenc = synthetic_probe_args("nvenc");
		let cpu = synthetic_probe_args("cpu");
		for args in [&nvenc, &cpu] {
			assert!(args
				.windows(2)
				.any(|p| p == ["-i", "color=c=black:s=64x64:r=30"]));
			assert!(args.windows(2).any(|p| p == ["-frames:v", "3"]));
			assert!(args.windows(2).any(|p| p == ["-pix_fmt", "yuv420p"]));
			assert!(args.contains(&"-an".into()));
			assert_eq!(&args[args.len() - 3..], ["-f", "null", "-"]);
			assert!(!args.contains(&"-filter_complex".into()));
		}
	}

	#[test]
	fn all_actual_process_failures_are_terminal_and_preserve_destination() {
		for stderr in [
			"generic",
			"invalid input",
			"filter error",
			"audio error",
			"NVENC unavailable",
		] {
			let fixture = Fixture::new();
			let files = fixture.files();
			let calls = std::cell::Cell::new(0);
			let process = tauri::async_runtime::block_on(encode_once(|| async {
				calls.set(calls.get() + 1);
				event_outcome(stderr, 1)
			}));
			assert_eq!(calls.get(), 1);
			let result = files.finish(
				process,
				|| panic!("validation after failed process"),
				|_, _| panic!("publication after failed process"),
				|| panic!("completion after failure"),
			);
			assert_eq!(result, Err(ExportError::Process));
			fixture.sentinel();
		}
	}

	#[test]
	fn validation_failure_preserves_destination_and_never_completes() {
		let fixture = Fixture::new();
		let files = fixture.files();
		assert_eq!(
			files.finish(
				Ok(()),
				|| Err(ExportError::OutputValidation),
				|_, _| panic!("publication before validation"),
				|| panic!("completed failed validation")
			),
			Err(ExportError::OutputValidation)
		);
		fixture.sentinel();
	}

	#[test]
	fn publication_failure_preserves_destination_and_never_completes() {
		let fixture = Fixture::new();
		let files = fixture.files();
		assert_eq!(
			files.finish(
				Ok(()),
				|| Ok(()),
				|_, _| Err(std::io::Error::from(std::io::ErrorKind::PermissionDenied)),
				|| panic!("completed failed publication")
			),
			Err(ExportError::Publication)
		);
		fixture.sentinel();
	}

	#[test]
	fn exit_zero_missing_empty_or_nonregular_stage_is_not_success() {
		let fixture = Fixture::new();
		let files = fixture.files();
		std::fs::write(&files.stage, b"").unwrap();
		assert_eq!(
			files.finish(
				Ok(()),
				|| panic!("validate empty file"),
				|_, _| panic!("publish empty"),
				|| panic!()
			),
			Err(ExportError::OutputValidation)
		);
		std::fs::remove_file(&files.stage).unwrap();
		assert_eq!(files.validate_file(), Err(ExportError::OutputValidation));
		std::fs::create_dir(&files.stage).unwrap();
		assert_eq!(files.validate_file(), Err(ExportError::OutputValidation));
		std::fs::remove_dir(&files.stage).unwrap();
		fixture.sentinel();
	}

	#[test]
	fn malformed_and_missing_required_media_structure_is_not_success() {
		let expected = MediaExpectation::composite(&settings("mp4-h264"), "16:9", None);
		for bytes in [b"not json".as_slice(), b"{}", b"{\"streams\":[]}"] {
			assert_eq!(expected.validate(bytes), Err(ExportError::OutputValidation));
		}
		for bytes in [
			media("h264", None, "mov,mp4", 1920, 1080),
			media("hevc", Some("aac"), "mov,mp4", 1920, 1080),
			media("h264", Some("aac"), "webm", 1920, 1080),
			media("h264", Some("aac"), "mov,mp4", 0, 1080),
			media("h264", Some("aac"), "mov,mp4", 1280, 720),
		] {
			assert_eq!(
				expected.validate(&bytes),
				Err(ExportError::OutputValidation)
			);
		}
	}

	#[test]
	fn reservation_collision_never_claims_or_removes_foreign_stage() {
		let fixture = Fixture::new();
		let files = ExportFiles::reserve(&fixture.destination(), "mp4-h264", "collision").unwrap();
		std::fs::write(&files.stage, b"foreign").unwrap();
		assert!(ExportFiles::reserve(&fixture.destination(), "mp4-h264", "collision").is_err());
		assert_eq!(std::fs::read(&files.stage).unwrap(), b"foreign");
	}

	#[test]
	fn cleanup_only_removes_invocation_owned_files() {
		let fixture = Fixture::new();
		let mut first = fixture.files();
		let second =
			ExportFiles::reserve(&fixture.destination(), "mp4-h264", &invocation_token()).unwrap();
		assert_ne!(first.stage, second.stage);
		let text = first.reserve_text("text", &invocation_token(), 0).unwrap();
		let foreign = fixture.0.join("foreign.txt");
		std::fs::write(&foreign, b"foreign").unwrap();
		assert!(!first.cleanup());
		assert!(!first.stage.exists());
		assert!(!Path::new(&text).exists());
		assert!(second.stage.exists());
		std::fs::write(&first.stage, b"new owner after cleanup").unwrap();
		assert!(!first.cleanup());
		assert_eq!(
			std::fs::read(&first.stage).unwrap(),
			b"new owner after cleanup"
		);
		assert!(foreign.exists());
		fixture.sentinel();
	}

	#[test]
	fn cleanup_failure_preserves_primary_sanitized_error() {
		let fixture = Fixture::new();
		let mut files = fixture.files();
		std::fs::remove_file(&files.stage).unwrap();
		std::fs::create_dir(&files.stage).unwrap();
		let message = finish_cleanup(&mut files, Err(ExportError::Process)).unwrap_err();
		assert!(message.starts_with(ExportError::Process.message()));
		assert!(message.contains("cleanup also failed"));
		assert!(!message.contains(fixture.0.to_str().unwrap()));
		std::fs::remove_dir(&files.stage).unwrap();
	}

	#[test]
	fn destination_source_identity_is_rejected_before_process() {
		let fixture = Fixture::new();
		assert_eq!(
			reject_source_destination(&fixture.destination(), &[&fixture.destination()]),
			Err(ExportError::InvalidRequest)
		);
		std::fs::write(fixture.destination(), b"source").unwrap();
		assert_eq!(
			reject_source_destination(&fixture.destination(), &[&fixture.destination()]),
			Err(ExportError::InvalidRequest)
		);
		assert!(reject_source_destination(
			&fixture.0.join("different.mp4"),
			&[&fixture.destination()]
		)
		.is_ok());
	}

	#[test]
	fn text_collision_never_claims_foreign_file_and_stage_extension_matches_format() {
		let fixture = Fixture::new();
		for format in ["mp4-h264", "mp4-h265", "mov-h264", "webm-vp9", "gif"] {
			let files = ExportFiles::reserve(&fixture.destination(), format, &invocation_token()).unwrap();
			assert_eq!(files.stage.extension().unwrap(), output_extension(format));
			assert_eq!(files.stage.parent(), files.destination.parent());
		}
		let mut owner = fixture.files();
		let token = invocation_token();
		let path = owner.reserve_text("foreign", &token, 0).unwrap();
		let mut other = ExportFiles::reserve(&fixture.destination(), "mp4-h264", &invocation_token()).unwrap();
		assert!(other.reserve_text("overwrite", &token, 0).is_err());
		assert!(!other.cleanup());
		assert_eq!(std::fs::read(&path).unwrap(), b"foreign");
	}

	#[test]
	fn actual_encode_infrastructure_errors_do_not_retry() {
		for error in [ExportError::Sidecar, ExportError::Spawn, ExportError::Event, ExportError::MissingTermination] {
			let calls = std::cell::Cell::new(0);
			let result = tauri::async_runtime::block_on(encode_once(|| async {
				calls.set(calls.get() + 1);
				Err(error)
			}));
			assert_eq!(result, Err(error));
			assert_eq!(calls.get(), 1);
		}
	}

	#[cfg(unix)]
	#[test]
	fn provable_symlink_and_hardlink_aliases_are_rejected() {
		let fixture = Fixture::new();
		let source = fixture.0.join("source.mp4");
		std::fs::write(&source, b"source").unwrap();
		let hard = fixture.0.join("hard.mp4");
		std::fs::hard_link(&source, &hard).unwrap();
		let sym = fixture.0.join("sym.mp4");
		std::os::unix::fs::symlink(&source, &sym).unwrap();
		for dest in [hard, sym] {
			assert_eq!(
				reject_source_destination(&dest, &[&source]),
				Err(ExportError::InvalidRequest)
			);
		}
	}

	#[test]
	fn copy_expectations_keep_mp3_vp8_and_silent_cases() {
		for (format, video, audio, container) in [
			("mp4-h264", "h264", Some("mp3"), "mov,mp4,m4a,3gp,3g2,mj2"),
			("mov-h264", "h264", Some("aac"), "mov,mp4,m4a,3gp,3g2,mj2"),
			("mp4-h265", "hevc", None, "mov,mp4,m4a,3gp,3g2,mj2"),
			("webm-vp9", "vp8", Some("vorbis"), "matroska,webm"),
			("webm-vp9", "vp9", None, "matroska,webm"),
		] {
			assert!(copy_compatible(format, video, audio));
			assert!(
				MediaExpectation::copied(format, Some((641, 359)), video, audio)
					.validate(&media(video, audio, container, 641, 359))
					.is_ok()
			);
		}
	}

	#[test]
	fn composite_expectations_keep_all_formats_audio_and_dimensions() {
		for (format, video, audio, container) in [
			("mp4-h264", "h264", Some("aac"), "mov,mp4"),
			("mp4-h265", "hevc", Some("aac"), "mov,mp4"),
			("mov-h264", "h264", Some("aac"), "mov,mp4"),
			("webm-vp9", "vp9", Some("opus"), "matroska,webm"),
			("gif", "gif", None, "gif"),
		] {
			assert!(MediaExpectation::composite(&settings(format), "16:9", None)
				.validate(&media(video, audio, container, 1920, 1080))
				.is_ok());
		}
	}

	#[test]
	fn copy_and_composite_write_explicit_muxer_and_keep_selection() {
		for format in ["mp4-h264", "mp4-h265", "mov-h264", "webm-vp9", "gif"] {
			let clips = vec![clip()];
			let settings = settings(format);
			let (args, _) = build_args(
				&clips,
				&[0],
				"original",
				&settings,
				&[false],
				Some((640, 360)),
				&[None],
				30.0,
				"stage",
			)
			.unwrap();
			assert_eq!(
				&args[args.len() - 3..],
				["-f", output_muxer(format), "stage"]
			);
			if format != "gif" {
				assert!(copy_candidate(&clips, "original", &settings).is_some());
				let (args, _) = build_copy_args(&clips[0], format, "stage");
				assert!(args.windows(2).any(|p| p == ["-c", "copy"]));
				assert_eq!(
					&args[args.len() - 3..],
					["-f", output_muxer(format), "stage"]
				);
			}
		}
	}

	#[test]
	fn encoding_progress_cannot_report_completion() {
		for seconds in [-1.0, 0.0, 1.0, 10.0, f64::INFINITY, f64::NAN] {
			assert!(encoding_progress(seconds, 1.0) < 1.0);
		}
	}

	#[test]
	fn publication_follows_validation_and_completion_follows_rename() {
		use std::cell::Cell;
		let fixture = Fixture::new();
		let files = fixture.files();
		let phase = Cell::new(0);
		files
			.finish(
				Ok(()),
				|| {
					fixture.sentinel();
					phase.set(1);
					Ok(())
				},
				|stage, destination| {
					assert_eq!(phase.get(), 1);
					fixture.sentinel();
					// A deterministic publication seam; native replacement separately below.
					assert_eq!(std::fs::read(stage).unwrap(), b"candidate");
					assert_eq!(destination, fixture.destination());
					phase.set(2);
					Ok(())
				},
				|| {
					assert_eq!(phase.get(), 2);
					phase.set(3);
				},
			)
			.unwrap();
		assert_eq!(phase.get(), 3);
	}

	#[cfg(unix)]
	#[test]
	fn native_same_directory_rename_replaces_only_after_validation() {
		let fixture = Fixture::new();
		let files = fixture.files();
		files
			.finish(
				Ok(()),
				|| {
					fixture.sentinel();
					Ok(())
				},
				|stage, dest| std::fs::rename(stage, dest),
				|| {
					assert_eq!(std::fs::read(fixture.destination()).unwrap(), b"candidate");
				},
			)
			.unwrap();
		assert!(!files.stage.exists());
	}

	#[test]
	fn nvenc_quality_mapping_is_distinct() {
		assert_eq!(nvenc_cq("high"), "19");
		assert_eq!(nvenc_cq("medium"), "23");
		assert_eq!(nvenc_cq("low"), "28");
	}

	#[test]
	fn h264_backends_do_not_share_codec_specific_flags() {
		let cpu = h264_video_args("cpu", "high").unwrap();
		let nvenc = h264_video_args("nvenc", "high").unwrap();

		assert!(cpu.iter().any(|arg| arg == "libx264"));
		assert!(cpu.iter().any(|arg| arg == "-crf"));
		assert!(!cpu.iter().any(|arg| arg == "h264_nvenc"));

		assert!(nvenc.iter().any(|arg| arg == "h264_nvenc"));
		assert!(nvenc.iter().any(|arg| arg == "-cq"));
		assert!(!nvenc.iter().any(|arg| arg == "-crf"));
		assert!(!nvenc.iter().any(|arg| arg == "veryfast"));
	}

	#[test]
	fn nvenc_is_scoped_to_mp4_h264() {
		assert!(validate_encoder("mp4-h264", "nvenc").is_ok());
		assert!(validate_encoder("mp4-h264", "cpu").is_ok());
		assert!(validate_encoder("mov-h264", "nvenc").is_err());
		assert!(validate_encoder("mp4-h264", "unknown").is_err());
	}

	#[test]
	fn unknown_export_formats_fail_before_ffmpeg_runs() {
		let settings = ExportSettings {
			format: "unknown".into(),
			resolution: "source".into(),
			quality: "high".into(),
			encoder: "cpu".into(),
		};
		assert!(validate_export_settings(&settings).is_err());
	}
}

/// Minimal event seam shared by actual export, ffprobe and synthetic probes.
/// Error and missing terminal events always win over an apparent exit zero.
#[derive(Default)]
struct ProcessEvents {
	code: Option<i32>,
	error: Option<ExportError>,
	stdout: Vec<u8>,
}

impl ProcessEvents {
	fn accept(&mut self, event: &CommandEvent, capture: bool) {
		match event {
			CommandEvent::Error(_) => self.error = Some(ExportError::Event),
			CommandEvent::Terminated(payload) => {
				if payload.code.is_none() || self.code.is_some() {
					self.error = Some(ExportError::MissingTermination);
				}
				self.code = payload.code;
			}
			CommandEvent::Stdout(bytes) if capture => {
				// JSON output is bounded; overflow is an event failure, not success.
				if self.stdout.len() + bytes.len() + 1 <= 1024 * 1024 {
					self.stdout.extend(bytes);
					self.stdout.push(b'\n');
				} else {
					self.error = Some(ExportError::Event);
				}
			}
			_ => {} // stderr is discarded, never classified or exposed
		}
	}

	fn finish(self) -> Result<ProcessOutput, ExportError> {
		if let Some(e) = self.error {
			return Err(e);
		}
		Ok(ProcessOutput {
			code: self.code.ok_or(ExportError::MissingTermination)?,
			stdout: self.stdout,
		})
	}
}

async fn run_media(
	app: &AppHandle,
	tool: &str,
	args: Vec<String>,
	total: Option<f64>,
) -> Result<ProcessOutput, ExportError> {
	let cmd = app
		.shell()
		.sidecar(tool)
		.map_err(|_| ExportError::Sidecar)?;
	let (mut rx, _child) = cmd.args(args).spawn().map_err(|_| ExportError::Spawn)?;
	let mut events = ProcessEvents::default();
	while let Some(event) = rx.recv().await {
		if let (Some(total), CommandEvent::Stdout(bytes)) = (total, &event) {
			parse_progress(app, &String::from_utf8_lossy(bytes), total);
		}
		events.accept(&event, tool == "ffprobe");
	}
	events.finish()
}

fn successful_process(outcome: Result<ProcessOutput, ExportError>) -> Result<(), ExportError> {
	match outcome {
		Ok(out) if out.code == 0 => Ok(()),
		Ok(_) => Err(ExportError::Process),
		Err(e) => Err(e),
	}
}

/// FnOnce is the local runner seam: actual encoding has one invocation and no
/// retry contract. Synthetic control probes use their separate foundation path.
async fn encode_once<F>(run: impl FnOnce() -> F) -> Result<(), ExportError>
where
	F: std::future::Future<Output = Result<ProcessOutput, ExportError>>,
{
	successful_process(run().await)
}

/// Actual export has exactly one process attempt. Preflight is not consulted,
/// and no failure (including NVENC-like stderr) can trigger a CPU retry.
async fn run_export(
	app: &AppHandle,
	files: &ExportFiles,
	args: Vec<String>,
	total: f64,
	expected: &MediaExpectation,
) -> Result<(), ExportError> {
	let process = encode_once(|| run_media(app, "ffmpeg", args, Some(total))).await;
	process?;
	files.validate_file()?;
	let args: Vec<String> = [
		"-v",
		"error",
		"-show_entries",
		"stream=codec_type,codec_name,width,height:format=format_name",
		"-of",
		"json",
	]
	.iter()
	.map(|s| s.to_string())
	.chain(std::iter::once(files.stage.to_string_lossy().into_owned()))
	.collect();
	let probe = run_media(app, "ffprobe", args, None).await?;
	if probe.code != 0 {
		return Err(ExportError::OutputValidation);
	}
	files.finish(
		Ok(()),
		|| expected.validate(&probe.stdout),
		|stage, destination| std::fs::rename(stage, destination),
		|| {
			let _ = app.emit("export:progress", 1.0_f64);
		},
	)
}

/// Parse a `-progress` stdout line and emit a 0..1 fraction.
fn parse_progress(app: &AppHandle, line: &str, total: f64) {
	let line = line.trim();
	if let Some(v) = line.strip_prefix("out_time_us=") {
		if let Ok(us) = v.trim().parse::<f64>() {
			emit_progress(app, us / 1_000_000.0, total);
		}
	} else if let Some(v) = line.strip_prefix("out_time_ms=") {
		// Some builds report microseconds under this key; treat as us.
		if let Ok(us) = v.trim().parse::<f64>() {
			emit_progress(app, us / 1_000_000.0, total);
		}
	}
}

fn emit_progress(app: &AppHandle, seconds: f64, total: f64) {
	let pct = encoding_progress(seconds, total);
	let _ = app.emit("export:progress", pct);
}

/// Reserve completion for the validated publication boundary.
fn encoding_progress(seconds: f64, total: f64) -> f64 {
	if seconds.is_finite() && total.is_finite() && total > 0.0 {
		(seconds / total).clamp(0.0, 0.999)
	} else {
		0.0
	}
}
