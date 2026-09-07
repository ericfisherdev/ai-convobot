use crate::database::{CompanionAttitude, Database};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub companion_id: i32,
    pub user_id: Option<i32>,
    pub created_at: DateTime<Utc>,
    pub last_activity: DateTime<Utc>,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct SessionManager {
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    session_timeout_minutes: i64,
}

impl SessionManager {
    pub fn new(timeout_minutes: i64) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            session_timeout_minutes: timeout_minutes,
        }
    }

    /// Create a new session, sweeping expired entries first so the map
    /// never grows past what one timeout window's worth of sessions needs.
    pub fn create_session(
        &self,
        companion_id: i32,
        user_id: Option<i32>,
    ) -> Result<Session, String> {
        let session_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let session = Session {
            id: session_id.clone(),
            companion_id,
            user_id,
            created_at: now,
            last_activity: now,
            is_active: true,
        };

        // Store session in memory
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        self.retain_unexpired(&mut sessions, now);
        sessions.insert(session_id.clone(), session.clone());

        println!("📦 Session created: {}", session_id);

        Ok(session)
    }

    /// Get a session by ID, refreshing its activity timestamp so the
    /// frontend's periodic keepalive actually extends the session's window.
    pub fn get_session(&self, session_id: &str) -> Result<Session, String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        let now = Utc::now();

        let session = sessions
            .get_mut(session_id)
            .filter(|s| s.is_active && !self.is_expired_at(s, now))
            .ok_or_else(|| format!("Session {} not found or expired", session_id))?;

        session.last_activity = now;
        Ok(session.clone())
    }

    /// Update attitude state for a session's companion and persist to database.
    /// The database is the single authoritative store for attitude state;
    /// this only bumps the session's activity timestamp and writes through.
    pub fn update_attitude(
        &self,
        session_id: &str,
        attitude: CompanionAttitude,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = Utc::now();

            // Persist to database
            Database::create_or_update_attitude(
                attitude.companion_id,
                attitude.target_id,
                &attitude.target_type,
                &attitude,
            )
            .map_err(|e| format!("Failed to persist attitude: {}", e))?;

            println!(
                "💾 Attitude updated for session {} and persisted to database",
                session_id
            );
            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// End a session by removing it entirely, rather than leaving a
    /// permanently inactive entry behind for nothing to ever prune.
    pub fn end_session(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        if sessions.remove(session_id).is_some() {
            println!("🔚 Session {} ended", session_id);
            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Check whether a session has expired as of `now`.
    fn is_expired_at(&self, session: &Session, now: DateTime<Utc>) -> bool {
        let timeout = Duration::minutes(self.session_timeout_minutes);
        now - session.last_activity > timeout
    }

    /// Drop every entry in `sessions` that has expired as of `now`. Shared by
    /// the amortised sweep in `create_session` and `sweep_expired_at`.
    fn retain_unexpired(&self, sessions: &mut HashMap<String, Session>, now: DateTime<Utc>) {
        sessions.retain(|_, s| !self.is_expired_at(s, now));
    }

    /// Remove every session expired as of `now`, returning the count removed.
    fn sweep_expired_at(&self, now: DateTime<Utc>) -> Result<usize, String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        let initial_count = sessions.len();
        self.retain_unexpired(&mut sessions, now);

        let removed_count = initial_count - sessions.len();
        if removed_count > 0 {
            println!("🧹 Cleaned up {} expired sessions", removed_count);
        }

        Ok(removed_count)
    }

    /// Clean up expired sessions
    pub fn cleanup_expired_sessions(&self) -> Result<usize, String> {
        self.sweep_expired_at(Utc::now())
    }

    /// Get statistics about active sessions
    pub fn get_session_stats(&self) -> Result<SessionStats, String> {
        let sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        let now = Utc::now();

        let active_count = sessions
            .values()
            .filter(|s| s.is_active && !self.is_expired_at(s, now))
            .count();

        Ok(SessionStats {
            active_sessions: active_count,
            total_sessions: sessions.len(),
        })
    }
}

#[cfg(test)]
impl SessionManager {
    /// Move a session's `last_activity` back by `by`, for deterministic
    /// expiry tests that don't depend on the wall clock or sleeping.
    fn backdate(&self, session_id: &str, by: Duration) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity -= by;
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SessionStats {
    pub active_sessions: usize,
    pub total_sessions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, Some(1)).unwrap();

        assert_eq!(session.companion_id, 1);
        assert_eq!(session.user_id, Some(1));
        assert!(session.is_active);
    }

    #[test]
    fn test_session_retrieval() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, Some(1)).unwrap();
        let retrieved = manager.get_session(&session.id).unwrap();

        assert_eq!(session.id, retrieved.id);
    }

    #[test]
    fn create_session_sweeps_expired_entries() {
        let manager = SessionManager::new(30);
        let a = manager.create_session(1, None).unwrap();
        manager.backdate(&a.id, Duration::minutes(31));
        let b = manager.create_session(1, None).unwrap();

        assert!(manager.get_session(&a.id).is_err());
        assert!(manager.get_session(&b.id).is_ok());
        assert_eq!(manager.get_session_stats().unwrap().total_sessions, 1);
    }

    #[test]
    fn create_session_keeps_live_entries() {
        let manager = SessionManager::new(30);
        let a = manager.create_session(1, None).unwrap();
        manager.backdate(&a.id, Duration::minutes(29));
        manager.create_session(1, None).unwrap();

        assert_eq!(manager.get_session_stats().unwrap().total_sessions, 2);
    }

    #[test]
    fn end_session_removes_the_entry() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, None).unwrap();

        manager.end_session(&session.id).unwrap();
        assert_eq!(manager.get_session_stats().unwrap().total_sessions, 0);

        let err = manager.end_session(&session.id).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn cleanup_expired_sessions_returns_removed_count() {
        let manager = SessionManager::new(30);
        let a = manager.create_session(1, None).unwrap();
        manager.create_session(1, None).unwrap();
        manager.backdate(&a.id, Duration::minutes(31));

        assert_eq!(manager.cleanup_expired_sessions().unwrap(), 1);
        assert_eq!(manager.cleanup_expired_sessions().unwrap(), 0);
    }

    #[test]
    fn get_session_refreshes_last_activity() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, None).unwrap();
        manager.backdate(&session.id, Duration::minutes(20));

        let refreshed = manager.get_session(&session.id).unwrap();
        assert!(refreshed.last_activity > session.last_activity);

        // Still within the window from the refreshed timestamp, so the
        // keepalive genuinely extends the session's life.
        manager.backdate(&session.id, Duration::minutes(20));
        assert!(manager.get_session(&session.id).is_ok());
    }

    #[test]
    fn stats_do_not_count_expired_sessions_as_active() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, None).unwrap();
        manager.backdate(&session.id, Duration::minutes(31));

        let stats = manager.get_session_stats().unwrap();
        assert_eq!(stats.active_sessions, 0);
        assert_eq!(stats.total_sessions, 1);
    }

    #[test]
    #[ignore = "needs a mock database before update_attitude can be exercised; see #63"]
    fn test_attitude_update() {
        let manager = SessionManager::new(30);
        let session = manager.create_session(1, Some(1)).unwrap();

        let attitude = CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 10.0,
            trust: 20.0,
            respect: 15.0,
            curiosity: 25.0,
            fear: 0.0,
            surprise: 5.0,
            anger: 0.0,
            joy: 30.0,
            sorrow: 0.0,
            disgust: 0.0,
            empathy: 20.0,
            gratitude: 15.0,
            jealousy: 0.0,
            suspicion: 0.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(50.0),
            last_updated: Utc::now().to_string(),
            created_at: Utc::now().to_string(),
        };

        manager.update_attitude(&session.id, attitude).unwrap();
    }
}
