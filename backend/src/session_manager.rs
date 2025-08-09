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
    pub attitude_state: Vec<CompanionAttitude>,
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

    /// Create a new session and load existing attitudes from database
    pub fn create_session(
        &self,
        companion_id: i32,
        user_id: Option<i32>,
    ) -> Result<Session, String> {
        let session_id = Uuid::new_v4().to_string();

        // Load existing attitudes from database
        let attitude_state = match Database::get_all_companion_attitudes(companion_id) {
            Ok(attitudes) => attitudes,
            Err(e) => {
                eprintln!(
                    "Failed to load attitudes for companion {}: {}",
                    companion_id, e
                );
                Vec::new() // Start with empty attitudes if load fails
            }
        };

        let session = Session {
            id: session_id.clone(),
            companion_id,
            user_id,
            created_at: Utc::now(),
            last_activity: Utc::now(),
            attitude_state,
            is_active: true,
        };

        // Store session in memory
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        sessions.insert(session_id.clone(), session.clone());

        println!(
            "📦 Session created: {} with {} attitudes loaded",
            session_id,
            session.attitude_state.len()
        );

        Ok(session)
    }

    /// Get an existing session or create a new one
    pub fn get_or_create_session(
        &self,
        session_id: Option<&str>,
        companion_id: i32,
        user_id: Option<i32>,
    ) -> Result<Session, String> {
        // Clean up expired sessions first
        self.cleanup_expired_sessions()?;

        if let Some(id) = session_id {
            if let Ok(session) = self.get_session(id) {
                // Update last activity
                self.update_activity(id)?;
                return Ok(session);
            }
        }

        // Create new session if not found or no ID provided
        self.create_session(companion_id, user_id)
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: &str) -> Result<Session, String> {
        let sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        sessions
            .get(session_id)
            .filter(|s| s.is_active && !self.is_session_expired(s))
            .cloned()
            .ok_or_else(|| format!("Session {} not found or expired", session_id))
    }

    /// Update session activity timestamp
    pub fn update_activity(&self, session_id: &str) -> Result<(), String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = Utc::now();
            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// Update attitude state in session and persist to database
    pub fn update_attitude(
        &self,
        session_id: &str,
        attitude: CompanionAttitude,
    ) -> Result<(), String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        if let Some(session) = sessions.get_mut(session_id) {
            // Update in-memory attitude state
            let existing_idx = session.attitude_state.iter().position(|a| {
                a.target_id == attitude.target_id && a.target_type == attitude.target_type
            });

            if let Some(idx) = existing_idx {
                session.attitude_state[idx] = attitude.clone();
            } else {
                session.attitude_state.push(attitude.clone());
            }

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

    /// Get current attitude state for a session
    pub fn get_attitude_state(&self, session_id: &str) -> Result<Vec<CompanionAttitude>, String> {
        let sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        sessions
            .get(session_id)
            .map(|s| s.attitude_state.clone())
            .ok_or_else(|| format!("Session {} not found", session_id))
    }

    /// Save session state to database before expiration
    pub fn persist_session(&self, session_id: &str) -> Result<(), String> {
        let sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        if let Some(session) = sessions.get(session_id) {
            // Persist all attitudes to database
            for attitude in &session.attitude_state {
                Database::create_or_update_attitude(
                    attitude.companion_id,
                    attitude.target_id,
                    &attitude.target_type,
                    attitude,
                )
                .map_err(|e| format!("Failed to persist attitude: {}", e))?;
            }

            println!(
                "💾 Session {} persisted with {} attitudes",
                session_id,
                session.attitude_state.len()
            );
            Ok(())
        } else {
            Err(format!("Session {} not found", session_id))
        }
    }

    /// End a session and persist state
    pub fn end_session(&self, session_id: &str) -> Result<(), String> {
        // Persist session state first
        self.persist_session(session_id)?;

        // Mark session as inactive
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        if let Some(session) = sessions.get_mut(session_id) {
            session.is_active = false;
            println!("🔚 Session {} ended", session_id);
        }

        Ok(())
    }

    /// Check if a session has expired
    fn is_session_expired(&self, session: &Session) -> bool {
        let timeout = Duration::minutes(self.session_timeout_minutes);
        Utc::now() - session.last_activity > timeout
    }

    /// Clean up expired sessions
    pub fn cleanup_expired_sessions(&self) -> Result<usize, String> {
        let mut sessions = self.sessions.lock().map_err(|e| e.to_string())?;
        let initial_count = sessions.len();

        // Find expired sessions
        let expired_ids: Vec<String> = sessions
            .iter()
            .filter(|(_, session)| self.is_session_expired(session))
            .map(|(id, _)| id.clone())
            .collect();

        // Persist and remove expired sessions
        for session_id in &expired_ids {
            if let Some(session) = sessions.get(&session_id.clone()) {
                // Persist attitudes before removal
                for attitude in &session.attitude_state {
                    let _ = Database::create_or_update_attitude(
                        attitude.companion_id,
                        attitude.target_id,
                        &attitude.target_type,
                        attitude,
                    );
                }
            }
            sessions.remove(session_id);
        }

        let removed_count = initial_count - sessions.len();
        if removed_count > 0 {
            println!("🧹 Cleaned up {} expired sessions", removed_count);
        }

        Ok(removed_count)
    }

    /// Get statistics about active sessions
    pub fn get_session_stats(&self) -> Result<SessionStats, String> {
        let sessions = self.sessions.lock().map_err(|e| e.to_string())?;

        let active_count = sessions.values().filter(|s| s.is_active).count();
        let total_attitudes: usize = sessions.values().map(|s| s.attitude_state.len()).sum();

        Ok(SessionStats {
            active_sessions: active_count,
            total_sessions: sessions.len(),
            total_attitudes_cached: total_attitudes,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct SessionStats {
    pub active_sessions: usize,
    pub total_sessions: usize,
    pub total_attitudes_cached: usize,
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
            relationship_score: Some(50.0),
            last_updated: Utc::now().to_string(),
            created_at: Utc::now().to_string(),
        };

        // Note: This test would need a mock database to fully work
        // manager.update_attitude(&session.id, attitude).unwrap();
    }
}
