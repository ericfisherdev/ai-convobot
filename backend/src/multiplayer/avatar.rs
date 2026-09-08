//! Validates and stores an avatar transferred over the join handshake.
//!
//! `validate_avatar` is pure (no I/O); `store_participant_avatar` is the one
//! function here that touches disk, and takes `&ParticipantId` (not `&str`)
//! for the same path-traversal reason `paths::participant_avatar_path` does
//! — the id grammar rules out `..`/`/` by construction.

use std::fmt;
use std::fs;
use std::io;
use std::path::PathBuf;

use crate::multiplayer::protocol::AvatarUpload;
use crate::participants::ParticipantId;
use crate::paths;

/// The largest decoded avatar this server accepts.
pub const MAX_AVATAR_BYTES: usize = 2 * 1024 * 1024;

/// An image format detected from magic bytes, never from the client's
/// claimed MIME type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvatarFormat {
    Png,
    Jpeg,
}

impl AvatarFormat {
    /// The file extension `store_participant_avatar` writes.
    pub fn extension(self) -> &'static str {
        match self {
            AvatarFormat::Png => "png",
            AvatarFormat::Jpeg => "jpg",
        }
    }

    /// The `Content-Type` the avatar-serving handler responds with.
    pub fn content_type(self) -> &'static str {
        match self {
            AvatarFormat::Png => "image/png",
            AvatarFormat::Jpeg => "image/jpeg",
        }
    }

    /// Detects a format from the leading bytes of a decoded image, or
    /// `None` if they match neither supported magic number.
    fn detect(bytes: &[u8]) -> Option<Self> {
        const PNG_MAGIC: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];
        if bytes.starts_with(&PNG_MAGIC) {
            Some(AvatarFormat::Png)
        } else if bytes.starts_with(&JPEG_MAGIC) {
            Some(AvatarFormat::Jpeg)
        } else {
            None
        }
    }
}

/// An avatar upload that has passed [`validate_avatar`]: decoded bytes with
/// a magic-byte-confirmed format.
#[derive(Debug, Clone)]
pub struct ValidatedAvatar {
    pub format: AvatarFormat,
    pub bytes: Vec<u8>,
}

/// Everything that can go wrong validating an [`AvatarUpload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AvatarError {
    NotBase64,
    /// `bytes` is the base64-decoded size that would have resulted, or the
    /// pre-decode estimate when rejected before decoding.
    TooLarge {
        bytes: usize,
    },
    UnsupportedFormat,
}

impl fmt::Display for AvatarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AvatarError::NotBase64 => write!(f, "avatar data is not valid base64"),
            AvatarError::TooLarge { bytes } => write!(
                f,
                "avatar is {} bytes, which exceeds the {} byte limit",
                bytes, MAX_AVATAR_BYTES
            ),
            AvatarError::UnsupportedFormat => {
                write!(f, "avatar is not a PNG or JPEG image")
            }
        }
    }
}

impl std::error::Error for AvatarError {}

/// Validates `upload`: rejects an over-length base64 payload before
/// decoding (no allocation for a payload that could not fit even if it
/// decoded losslessly), then decodes and checks the decoded size and the
/// magic bytes.
///
/// # Errors
/// [`AvatarError::TooLarge`] if the base64 payload's length implies more
/// than [`MAX_AVATAR_BYTES`] decoded bytes, or the decoded bytes exceed it;
/// [`AvatarError::NotBase64`] if `data_base64` fails to decode;
/// [`AvatarError::UnsupportedFormat`] if the decoded bytes match neither a
/// PNG nor a JPEG magic number.
pub fn validate_avatar(upload: &AvatarUpload) -> Result<ValidatedAvatar, AvatarError> {
    // Base64 expands 3 bytes to 4, so a payload longer than this could not
    // decode to <= MAX_AVATAR_BYTES bytes even before attempting to decode
    // it.
    const MAX_BASE64_LEN: usize = MAX_AVATAR_BYTES * 4 / 3 + 4;
    if upload.data_base64.len() > MAX_BASE64_LEN {
        return Err(AvatarError::TooLarge {
            bytes: upload.data_base64.len(),
        });
    }

    let bytes =
        crate::multiplayer::handshake::decode(&upload.data_base64).ok_or(AvatarError::NotBase64)?;

    if bytes.len() > MAX_AVATAR_BYTES {
        return Err(AvatarError::TooLarge { bytes: bytes.len() });
    }

    let format = AvatarFormat::detect(&bytes).ok_or(AvatarError::UnsupportedFormat)?;

    Ok(ValidatedAvatar { format, bytes })
}

/// Writes `avatar` under `paths::participant_avatars_dir()`, creating the
/// directory if needed, then removes the other extension's file if present
/// so a bot that switches from JPEG to PNG (or back) does not leave a stale
/// file behind for the avatar-serving handler to pick up by mistake.
///
/// Goes through `paths::participant_avatar_path`, not a manual join, so the
/// path-traversal guarantee (the `&ParticipantId` grammar rules out `..`
/// and `/`) lives in exactly one place.
pub fn store_participant_avatar(
    id: &ParticipantId,
    avatar: &ValidatedAvatar,
) -> io::Result<PathBuf> {
    fs::create_dir_all(paths::participant_avatars_dir())?;
    let path = paths::participant_avatar_path(id, avatar.format.extension());
    fs::write(&path, &avatar.bytes)?;
    remove_other_extension(id, avatar.format)?;
    Ok(path)
}

fn remove_other_extension(id: &ParticipantId, format: AvatarFormat) -> io::Result<()> {
    let other_extension = match format {
        AvatarFormat::Png => "jpg",
        AvatarFormat::Jpeg => "png",
    };
    let other_path = paths::participant_avatar_path(id, other_extension);
    if other_path.exists() {
        fs::remove_file(&other_path)?;
    }
    Ok(())
}

/// Best-effort removal of any stored avatar for `id` (both extensions).
/// Used to roll back a file `store_participant_avatar` partially wrote when
/// admission fails after storage (`host.rs::admit`): a missing file is not
/// an error (there may never have been one to remove), and any other I/O
/// error is logged rather than propagated since this itself runs on an
/// already-failing path with nothing left to report the error to.
pub fn remove_stored_avatar(id: &ParticipantId) {
    for format in [AvatarFormat::Png, AvatarFormat::Jpeg] {
        let path = paths::participant_avatar_path(id, format.extension());
        if let Err(e) = fs::remove_file(&path) {
            if e.kind() != io::ErrorKind::NotFound {
                eprintln!("multiplayer: failed to remove {}: {}", path.display(), e);
            }
        }
    }
}

/// Looks for a previously stored avatar for `id`, trying PNG then JPEG
/// (the two formats `store_participant_avatar` ever writes). Used by the
/// avatar-serving HTTP handler (`main.rs::multiplayer_participant_avatar`).
/// Not used to decide a fresh join's `avatar_url` (`host.rs::admit` sets
/// that only from its own successful store) since a file found here is not
/// scoped to any particular join.
pub fn find_stored_avatar(id: &ParticipantId) -> Option<(AvatarFormat, PathBuf)> {
    for format in [AvatarFormat::Png, AvatarFormat::Jpeg] {
        let path = paths::participant_avatar_path(id, format.extension());
        if path.exists() {
            return Some((format, path));
        }
    }
    None
}

/// The directory-parametrised twin of [`store_participant_avatar`], used
/// only by this module's own tests so they never have to call the
/// process-wide `paths::init`.
#[cfg(test)]
use std::path::Path;

#[cfg(test)]
fn store_participant_avatar_in(
    dir: &Path,
    id: &ParticipantId,
    avatar: &ValidatedAvatar,
) -> io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(format!("{}.{}", id.as_str(), avatar.format.extension()));
    fs::write(&path, &avatar.bytes)?;

    let other_extension = match avatar.format {
        AvatarFormat::Png => "jpg",
        AvatarFormat::Jpeg => "png",
    };
    let other_path = dir.join(format!("{}.{}", id.as_str(), other_extension));
    if other_path.exists() {
        fs::remove_file(&other_path)?;
    }

    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine as _;

    const PNG_MAGIC: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    const JPEG_MAGIC: [u8; 3] = [0xFF, 0xD8, 0xFF];

    fn upload_of(bytes: &[u8]) -> AvatarUpload {
        AvatarUpload {
            mime: "image/png".to_string(),
            data_base64: BASE64.encode(bytes),
        }
    }

    fn id(s: &str) -> ParticipantId {
        ParticipantId::parse(s).unwrap()
    }

    #[test]
    fn accepts_a_minimal_png_header() {
        let mut bytes = PNG_MAGIC.to_vec();
        bytes.extend_from_slice(b"rest of file");
        let validated = validate_avatar(&upload_of(&bytes)).unwrap();
        assert_eq!(validated.format, AvatarFormat::Png);
        assert_eq!(validated.bytes, bytes);
    }

    #[test]
    fn accepts_a_minimal_jpeg_header() {
        let mut bytes = JPEG_MAGIC.to_vec();
        bytes.extend_from_slice(b"rest of file");
        let validated = validate_avatar(&upload_of(&bytes)).unwrap();
        assert_eq!(validated.format, AvatarFormat::Jpeg);
    }

    #[test]
    fn rejects_a_gif() {
        let bytes = b"GIF89a...".to_vec();
        assert_eq!(
            validate_avatar(&upload_of(&bytes)).unwrap_err(),
            AvatarError::UnsupportedFormat
        );
    }

    #[test]
    fn rejects_more_than_the_byte_limit() {
        let mut bytes = PNG_MAGIC.to_vec();
        bytes.resize(MAX_AVATAR_BYTES + 1, 0);
        let err = validate_avatar(&upload_of(&bytes)).unwrap_err();
        assert!(matches!(err, AvatarError::TooLarge { .. }));
    }

    #[test]
    fn accepts_exactly_the_byte_limit() {
        let mut bytes = PNG_MAGIC.to_vec();
        bytes.resize(MAX_AVATAR_BYTES, 0);
        assert!(validate_avatar(&upload_of(&bytes)).is_ok());
    }

    #[test]
    fn rejects_invalid_base64() {
        let upload = AvatarUpload {
            mime: "image/png".to_string(),
            data_base64: "not valid base64!!".to_string(),
        };
        assert_eq!(
            validate_avatar(&upload).unwrap_err(),
            AvatarError::NotBase64
        );
    }

    #[test]
    fn store_then_switch_format_leaves_exactly_one_file() {
        let dir = tempfile::tempdir().unwrap();
        let bot1 = id("bot1");

        let mut png_bytes = PNG_MAGIC.to_vec();
        png_bytes.extend_from_slice(b"png data");
        let png = ValidatedAvatar {
            format: AvatarFormat::Png,
            bytes: png_bytes,
        };
        let png_path = store_participant_avatar_in(dir.path(), &bot1, &png).unwrap();
        assert!(png_path.exists());

        let mut jpeg_bytes = JPEG_MAGIC.to_vec();
        jpeg_bytes.extend_from_slice(b"jpeg data");
        let jpeg = ValidatedAvatar {
            format: AvatarFormat::Jpeg,
            bytes: jpeg_bytes,
        };
        let jpeg_path = store_participant_avatar_in(dir.path(), &bot1, &jpeg).unwrap();
        assert!(jpeg_path.exists());
        assert!(
            !png_path.exists(),
            "switching format should remove the old extension's file"
        );

        let entries: Vec<_> = fs::read_dir(dir.path()).unwrap().collect();
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn format_extension_and_content_type() {
        assert_eq!(AvatarFormat::Png.extension(), "png");
        assert_eq!(AvatarFormat::Png.content_type(), "image/png");
        assert_eq!(AvatarFormat::Jpeg.extension(), "jpg");
        assert_eq!(AvatarFormat::Jpeg.content_type(), "image/jpeg");
    }
}
