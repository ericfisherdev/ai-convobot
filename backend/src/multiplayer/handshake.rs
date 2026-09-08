//! The HMAC join-proof both host and joiner compute, and the base64
//! encoding used for both `Challenge.nonce` and `Join.proof` on the wire.
//!
//! Pure, no I/O, no `Database` dependency. `host.rs::authenticate` calls
//! [`verify_join_proof`]; the joiner (#130) calls [`new_nonce`] is host-only
//! (it never runs on the joiner side) and [`join_proof`]/[`encode`]/
//! [`decode`] to compute and send its proof.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use hmac::{Hmac, KeyInit, Mac};
use rand::RngCore;
use sha2::Sha256;

use crate::participants::ParticipantId;

type HmacSha256 = Hmac<Sha256>;

/// The length, in bytes, of a join nonce.
pub const NONCE_BYTES: usize = 32;

/// Generates a fresh random nonce for one join attempt. Never reused: a
/// replayed proof from an earlier connection has no matching nonce on this
/// one.
pub fn new_nonce() -> [u8; NONCE_BYTES] {
    let mut nonce = [0u8; NONCE_BYTES];
    rand::thread_rng().fill_bytes(&mut nonce);
    nonce
}

/// Computes the HMAC-SHA256 proof over `nonce || id.as_bytes()`, keyed with
/// `password`. The nonce is fixed-length raw bytes, so the concatenation is
/// unambiguous without a separator: no id can produce the same
/// `nonce || id` byte string as a different (nonce, id) pair when the nonce
/// length is fixed.
#[allow(dead_code)] // wired up by #130: the joiner computes its own proof with this
pub fn join_proof(password: &str, nonce: &[u8; NONCE_BYTES], id: &ParticipantId) -> [u8; 32] {
    let mut mac =
        HmacSha256::new_from_slice(password.as_bytes()).expect("HMAC accepts a key of any length");
    mac.update(nonce);
    mac.update(id.as_str().as_bytes());
    mac.finalize().into_bytes().into()
}

/// Verifies a base64-encoded proof against the expected `(password, nonce,
/// id)` triple. Any base64 decode failure is treated as a mismatch rather
/// than propagated, so callers have exactly one failure case to handle.
///
/// Constant-time by construction: `Mac::verify_slice` does the comparison,
/// not a manual `==`.
pub fn verify_join_proof(
    password: &str,
    nonce: &[u8; NONCE_BYTES],
    id: &ParticipantId,
    proof_base64: &str,
) -> bool {
    let Some(proof_bytes) = decode(proof_base64) else {
        return false;
    };
    let mut mac =
        HmacSha256::new_from_slice(password.as_bytes()).expect("HMAC accepts a key of any length");
    mac.update(nonce);
    mac.update(id.as_str().as_bytes());
    mac.verify_slice(&proof_bytes).is_ok()
}

/// Standard (padded) base64 encoding, the one encoding this module and the
/// wire protocol use for both `Challenge.nonce` and `Join.proof`.
pub fn encode(bytes: &[u8]) -> String {
    BASE64.encode(bytes)
}

/// The inverse of [`encode`]. `None` on any decode error.
pub fn decode(value: &str) -> Option<Vec<u8>> {
    BASE64.decode(value).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(s: &str) -> ParticipantId {
        ParticipantId::parse(s).unwrap()
    }

    // Known-answer vector, computed once and pinned: HMAC-SHA256 over
    // 32 zero bytes || b"bot1", keyed with "hunter2".
    const KNOWN_PASSWORD: &str = "hunter2";
    const KNOWN_NONCE: [u8; NONCE_BYTES] = [0u8; NONCE_BYTES];

    fn known_proof_base64() -> String {
        let proof = join_proof(KNOWN_PASSWORD, &KNOWN_NONCE, &id("bot1"));
        encode(&proof)
    }

    #[test]
    fn known_answer_vector_verifies() {
        let proof_b64 = known_proof_base64();
        assert!(verify_join_proof(
            KNOWN_PASSWORD,
            &KNOWN_NONCE,
            &id("bot1"),
            &proof_b64
        ));
    }

    #[test]
    fn known_answer_vector_is_pinned() {
        // Regression pin: if the HMAC construction ever changes, this
        // constant must be updated deliberately, not silently.
        assert_eq!(
            known_proof_base64(),
            "InsAgt1Ect9E7a5GdJTsdsxXvrX6c1InKFVqJnRBXUk="
        );
    }

    #[test]
    fn wrong_password_fails() {
        let proof_b64 = known_proof_base64();
        assert!(!verify_join_proof(
            "wrong",
            &KNOWN_NONCE,
            &id("bot1"),
            &proof_b64
        ));
    }

    #[test]
    fn wrong_nonce_fails() {
        let proof_b64 = known_proof_base64();
        let other_nonce = [1u8; NONCE_BYTES];
        assert!(!verify_join_proof(
            KNOWN_PASSWORD,
            &other_nonce,
            &id("bot1"),
            &proof_b64
        ));
    }

    #[test]
    fn wrong_id_fails() {
        let proof_b64 = known_proof_base64();
        assert!(!verify_join_proof(
            KNOWN_PASSWORD,
            &KNOWN_NONCE,
            &id("bot2"),
            &proof_b64
        ));
    }

    #[test]
    fn truncated_proof_fails() {
        let mut proof_b64 = known_proof_base64();
        proof_b64.truncate(proof_b64.len() - 4);
        assert!(!verify_join_proof(
            KNOWN_PASSWORD,
            &KNOWN_NONCE,
            &id("bot1"),
            &proof_b64
        ));
    }

    #[test]
    fn non_base64_proof_fails() {
        assert!(!verify_join_proof(
            KNOWN_PASSWORD,
            &KNOWN_NONCE,
            &id("bot1"),
            "not valid base64!!"
        ));
    }

    #[test]
    fn empty_password_still_produces_a_deterministic_proof() {
        // The host refuses empty passwords at the handler level (host.rs),
        // not here: this module just has to behave deterministically.
        let a = join_proof("", &KNOWN_NONCE, &id("bot1"));
        let b = join_proof("", &KNOWN_NONCE, &id("bot1"));
        assert_eq!(a, b);
    }

    #[test]
    fn encode_decode_round_trips() {
        let bytes = new_nonce();
        assert_eq!(decode(&encode(&bytes)).unwrap(), bytes.to_vec());
    }

    #[test]
    fn decode_rejects_invalid_base64() {
        assert_eq!(decode("not valid base64!!"), None);
    }

    #[test]
    fn new_nonce_is_not_trivially_constant() {
        // Not a cryptographic randomness test, just a sanity check that two
        // calls do not return the same all-zero (or otherwise identical)
        // buffer.
        assert_ne!(new_nonce(), new_nonce());
    }
}
