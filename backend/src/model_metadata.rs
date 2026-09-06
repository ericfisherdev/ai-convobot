//! Reads static facts (architecture, layer count, on-disk size) out of a
//! GGUF file's metadata header, without loading any tensor weights.
//!
//! `GgufContext` opens the file with `no_alloc: true`, so this only parses
//! the header — no `LlamaBackend` is needed and no memory is spent on
//! tensor data.

use std::fmt;
use std::fs;
use std::path::Path;

use llama_cpp_2::gguf::GgufContext;
use serde::Serialize;

/// Static facts about a GGUF model, read from its metadata header only.
#[derive(Debug, Clone, Serialize)]
pub struct ModelFacts {
    pub path: String,
    pub architecture: String,
    pub layer_count: u32,
    pub file_size_bytes: u64,
}

impl ModelFacts {
    /// On-disk size in MB, rounded up, with a floor of 1 so it is never
    /// used as a zero divisor.
    pub fn size_mb(&self) -> u64 {
        self.file_size_bytes.div_ceil(1024 * 1024).max(1)
    }
}

/// Failure modes for `read_model_facts`.
#[derive(Debug)]
pub enum ModelFactsError {
    /// The file could not be opened or `stat`'d.
    Io(std::io::Error),
    /// The path is not a valid GGUF file (or contains a null byte).
    NotGguf(String),
    /// A required metadata key was not present in the header.
    MissingKey(String),
    /// A key was present but stored as a type this reader doesn't expect.
    UnexpectedType { key: String, kv_type: u32 },
}

impl fmt::Display for ModelFactsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFactsError::Io(e) => write!(f, "failed to read model file: {}", e),
            ModelFactsError::NotGguf(path) => write!(f, "not a valid GGUF file: {}", path),
            ModelFactsError::MissingKey(key) => write!(f, "missing GGUF metadata key: {}", key),
            ModelFactsError::UnexpectedType { key, kv_type } => write!(
                f,
                "GGUF metadata key '{}' has unexpected type {}",
                key, kv_type
            ),
        }
    }
}

impl std::error::Error for ModelFactsError {}

impl From<std::io::Error> for ModelFactsError {
    fn from(e: std::io::Error) -> Self {
        ModelFactsError::Io(e)
    }
}

/// Reads architecture, layer count, and on-disk size from a GGUF file's
/// metadata header.
///
/// The layer count comes from `<arch>.block_count`, where `<arch>` is
/// `general.architecture` — the same key `llama_model_base::load_hparams`
/// reads in llama.cpp. `kv_type` is always checked before calling a typed
/// getter: `val_u32`/`val_i32` abort inside llama.cpp if the stored type
/// doesn't match.
pub fn read_model_facts(path: &Path) -> Result<ModelFacts, ModelFactsError> {
    let file_size_bytes = fs::metadata(path)?.len();

    let ctx = GgufContext::from_file(path)
        .ok_or_else(|| ModelFactsError::NotGguf(path.display().to_string()))?;

    let architecture = read_string(&ctx, "general.architecture")?;
    let block_count_key = format!("{}.block_count", architecture);
    let layer_count = read_layer_count(&ctx, &block_count_key)?;

    Ok(ModelFacts {
        path: path.display().to_string(),
        architecture,
        layer_count,
        file_size_bytes,
    })
}

fn read_string(ctx: &GgufContext, key: &str) -> Result<String, ModelFactsError> {
    let idx = ctx.find_key(key);
    if idx < 0 {
        return Err(ModelFactsError::MissingKey(key.to_string()));
    }
    let kv_type = ctx.kv_type(idx);
    if kv_type != llama_cpp_sys_2::GGUF_TYPE_STRING {
        return Err(ModelFactsError::UnexpectedType {
            key: key.to_string(),
            kv_type,
        });
    }
    ctx.val_str(idx)
        .map(str::to_string)
        .ok_or_else(|| ModelFactsError::UnexpectedType {
            key: key.to_string(),
            kv_type,
        })
}

/// Reads a key stored as `GGUF_TYPE_UINT32` or `GGUF_TYPE_INT32`.
/// `<arch>.block_count` is written as `uint32_t` by llama.cpp's own writer,
/// but the GGUF spec allows either signedness for integer KV entries, so
/// both are accepted here.
fn read_layer_count(ctx: &GgufContext, key: &str) -> Result<u32, ModelFactsError> {
    let idx = ctx.find_key(key);
    if idx < 0 {
        return Err(ModelFactsError::MissingKey(key.to_string()));
    }
    let kv_type = ctx.kv_type(idx);
    if kv_type == llama_cpp_sys_2::GGUF_TYPE_UINT32 {
        Ok(ctx.val_u32(idx))
    } else if kv_type == llama_cpp_sys_2::GGUF_TYPE_INT32 {
        u32::try_from(ctx.val_i32(idx)).map_err(|_| ModelFactsError::UnexpectedType {
            key: key.to_string(),
            kv_type,
        })
    } else {
        Err(ModelFactsError::UnexpectedType {
            key: key.to_string(),
            kv_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    const GGUF_TYPE_UINT32: i32 = 4;
    const GGUF_TYPE_STRING: i32 = 8;

    fn gguf_string(s: &str) -> Vec<u8> {
        let mut buf = (s.len() as u64).to_le_bytes().to_vec();
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    /// Builds the bytes of a minimal GGUF v3 file with no tensors and the
    /// given metadata key/value pairs, mirroring the header format read by
    /// `gguf_init_from_reader` in `ggml/src/gguf.cpp`: magic, version,
    /// tensor count, kv count, then each kv as (string key, i32 type, raw
    /// value bytes).
    fn build_gguf(kvs: &[(&str, i32, Vec<u8>)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0i64.to_le_bytes()); // n_tensors
        buf.extend_from_slice(&(kvs.len() as i64).to_le_bytes()); // n_kv
        for (key, kv_type, value) in kvs {
            buf.extend(gguf_string(key));
            buf.extend_from_slice(&kv_type.to_le_bytes());
            buf.extend_from_slice(value);
        }
        buf
    }

    fn write_fixture(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let mut file = fs::File::create(&path).unwrap();
        file.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn happy_path_returns_arch_layers_and_size() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = build_gguf(&[
            (
                "general.architecture",
                GGUF_TYPE_STRING,
                gguf_string("llama"),
            ),
            (
                "llama.block_count",
                GGUF_TYPE_UINT32,
                32u32.to_le_bytes().to_vec(),
            ),
        ]);
        let path = write_fixture(&dir, "model.gguf", &bytes);

        let facts = read_model_facts(&path).unwrap();

        assert_eq!(facts.architecture, "llama");
        assert_eq!(facts.layer_count, 32);
        assert_eq!(facts.file_size_bytes, bytes.len() as u64);
    }

    #[test]
    fn missing_file_is_io_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.gguf");

        let err = read_model_facts(&path).unwrap_err();

        assert!(matches!(err, ModelFactsError::Io(_)));
    }

    #[test]
    fn non_gguf_bytes_is_not_gguf_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fixture(&dir, "not-a-model.gguf", b"this is not a gguf file");

        let err = read_model_facts(&path).unwrap_err();

        assert!(matches!(err, ModelFactsError::NotGguf(_)));
    }

    #[test]
    fn missing_block_count_is_missing_key_error() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = build_gguf(&[(
            "general.architecture",
            GGUF_TYPE_STRING,
            gguf_string("llama"),
        )]);
        let path = write_fixture(&dir, "model.gguf", &bytes);

        let err = read_model_facts(&path).unwrap_err();

        assert!(matches!(err, ModelFactsError::MissingKey(key) if key == "llama.block_count"));
    }

    #[test]
    fn block_count_stored_as_string_is_unexpected_type_error() {
        let dir = tempfile::tempdir().unwrap();
        let bytes = build_gguf(&[
            (
                "general.architecture",
                GGUF_TYPE_STRING,
                gguf_string("llama"),
            ),
            ("llama.block_count", GGUF_TYPE_STRING, gguf_string("32")),
        ]);
        let path = write_fixture(&dir, "model.gguf", &bytes);

        let err = read_model_facts(&path).unwrap_err();

        assert!(matches!(
            err,
            ModelFactsError::UnexpectedType { key, .. } if key == "llama.block_count"
        ));
    }
}
