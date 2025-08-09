use crate::database::{CompanionAttitude, ThirdPartyIndividual};

/// Handles conversion of attitude data into LLM prompt context and response calibration
pub struct AttitudeFormatter {
    // Attitude significance thresholds
    pub low_threshold: f32,
    pub medium_threshold: f32,
    pub high_threshold: f32,
}

impl AttitudeFormatter {
    pub fn new() -> Self {
        Self {
            low_threshold: 20.0,
            medium_threshold: 50.0,
            high_threshold: 80.0,
        }
    }

    /// Format attitudes into LLM prompt context with response calibration instructions
    pub fn format_attitude_context(
        &self,
        attitudes: &[CompanionAttitude],
        third_parties: &[ThirdPartyIndividual],
        target_user: &str,
    ) -> String {
        if attitudes.is_empty() {
            return String::new();
        }

        let mut context = String::new();

        // Primary user attitude (most important)
        if let Some(user_attitude) = attitudes.iter().find(|a| a.target_type == "user") {
            context.push_str(&self.format_primary_attitude(user_attitude, target_user));
        }

        // Third-party attitudes (if significant)
        let significant_third_parties =
            self.get_significant_third_party_attitudes(attitudes, third_parties);
        if !significant_third_parties.is_empty() {
            context.push_str("\n\nRelationship awareness:\n");
            for (party, attitude) in significant_third_parties {
                context.push_str(&format!(
                    "- Attitude toward {}: {}\n",
                    party.name,
                    self.format_attitude_summary(attitude)
                ));
            }
        }

        // Response calibration instructions
        context.push_str(&self.generate_response_calibration_instructions(attitudes));

        context
    }

    /// Format the primary user attitude with emotional context
    fn format_primary_attitude(&self, attitude: &CompanionAttitude, user_name: &str) -> String {
        let relationship_level = self.calculate_relationship_level(attitude);
        let emotional_state = self.analyze_emotional_state(attitude);
        let behavioral_instructions = self.generate_behavioral_instructions(attitude);

        format!(
            "Current relationship with {}: {} ({})\n\
            Emotional state: {}\n\
            Response guidance: {}",
            user_name,
            relationship_level.name,
            relationship_level.score,
            emotional_state,
            behavioral_instructions
        )
    }

    /// Calculate relationship level based on overall attitude
    fn calculate_relationship_level(&self, attitude: &CompanionAttitude) -> RelationshipLevel {
        let score = attitude.relationship_score.unwrap_or(0.0);

        match score {
            s if s >= 80.0 => RelationshipLevel {
                name: "Intimate",
                score: s,
                description: "deeply connected, comfortable with vulnerability",
            },
            s if s >= 60.0 => RelationshipLevel {
                name: "Close",
                score: s,
                description: "warm and trusting, shares personal thoughts",
            },
            s if s >= 40.0 => RelationshipLevel {
                name: "Friendly",
                score: s,
                description: "positive and helpful, maintains boundaries",
            },
            s if s >= 20.0 => RelationshipLevel {
                name: "Acquaintance",
                score: s,
                description: "polite but reserved, professional",
            },
            s if s >= 0.0 => RelationshipLevel {
                name: "Neutral",
                score: s,
                description: "factual and cautious, minimal emotion",
            },
            s if s >= -20.0 => RelationshipLevel {
                name: "Distant",
                score: s,
                description: "formal and detached, reluctant engagement",
            },
            s if s >= -40.0 => RelationshipLevel {
                name: "Unfriendly",
                score: s,
                description: "curt and dismissive, shows irritation",
            },
            s if s >= -60.0 => RelationshipLevel {
                name: "Hostile",
                score: s,
                description: "argumentative and defensive, openly annoyed",
            },
            _ => RelationshipLevel {
                name: "Antagonistic",
                score,
                description: "aggressive and confrontational, barely cooperative",
            },
        }
    }

    /// Analyze dominant emotional states from attitude dimensions
    fn analyze_emotional_state(&self, attitude: &CompanionAttitude) -> String {
        let mut emotions = Vec::new();

        // High-intensity emotions (>70)
        if attitude.joy > 70.0 {
            emotions.push("very happy");
        } else if attitude.joy > self.medium_threshold {
            emotions.push("pleased");
        }

        if attitude.anger > 70.0 {
            emotions.push("quite angry");
        } else if attitude.anger > self.medium_threshold {
            emotions.push("irritated");
        }

        if attitude.fear > self.high_threshold {
            emotions.push("anxious");
        }
        if attitude.trust > self.high_threshold {
            emotions.push("deeply trusting");
        } else if attitude.trust > self.medium_threshold {
            emotions.push("trusting");
        }

        if attitude.curiosity > self.high_threshold {
            emotions.push("very curious");
        }
        if attitude.empathy > self.high_threshold {
            emotions.push("deeply empathetic");
        }
        if attitude.suspicion > self.medium_threshold {
            emotions.push("suspicious");
        }
        if attitude.sorrow > self.medium_threshold {
            emotions.push("sad");
        }

        // New attitude dimensions
        if attitude.lust > 70.0 {
            emotions.push("very lustful");
        } else if attitude.lust > self.medium_threshold {
            emotions.push("physically attracted");
        }

        if attitude.love > 80.0 {
            emotions.push("deeply in love");
        } else if attitude.love > self.medium_threshold {
            emotions.push("in love");
        }

        if attitude.anxiety > 70.0 {
            emotions.push("very anxious");
        } else if attitude.anxiety > self.medium_threshold {
            emotions.push("nervous");
        }

        if attitude.butterflies > 80.0 {
            emotions.push("completely flustered");
        } else if attitude.butterflies > self.medium_threshold {
            emotions.push("giddy");
        }

        if emotions.is_empty() {
            "emotionally balanced".to_string()
        } else {
            emotions.join(", ")
        }
    }

    /// Generate specific behavioral instructions based on attitudes
    fn generate_behavioral_instructions(&self, attitude: &CompanionAttitude) -> String {
        let mut instructions = Vec::new();

        // Trust-based behaviors
        match attitude.trust {
            t if t > self.high_threshold => {
                instructions.push("be open and vulnerable, share personal insights")
            }
            t if t > self.medium_threshold => {
                instructions.push("be warm and sharing, show genuine interest")
            }
            t if t > self.low_threshold => {
                instructions.push("be friendly but maintain some privacy")
            }
            t if t > 0.0 => instructions.push("be polite and helpful, keep responses professional"),
            t if t > -self.medium_threshold => {
                instructions.push("be cautious and formal, avoid personal topics")
            }
            _ => instructions.push("be guarded and brief, show reluctance to engage deeply"),
        }

        // Joy/Sorrow influence on tone
        match (attitude.joy, attitude.sorrow) {
            (j, s) if j > self.medium_threshold && s < self.low_threshold => {
                instructions.push("use enthusiastic and positive language, include light humor")
            }
            (j, s) if s > self.medium_threshold && j < self.low_threshold => {
                instructions.push("be more subdued and supportive, avoid overly cheerful responses")
            }
            _ => {}
        }

        // Anger influence
        if attitude.anger > self.medium_threshold {
            instructions
                .push("be more direct and blunt, show impatience with lengthy explanations");
        }

        // Curiosity influence
        if attitude.curiosity > self.high_threshold {
            instructions.push("ask follow-up questions, show genuine interest in details");
        }

        // Empathy influence
        if attitude.empathy > self.high_threshold {
            instructions.push("be very understanding and supportive, validate emotions");
        }

        // Respect influence on formality
        match attitude.respect {
            r if r > self.high_threshold => {
                instructions.push("show deference and use more formal language when appropriate")
            }
            r if r < -self.medium_threshold => {
                instructions.push("be more casual or dismissive, less concerned with politeness")
            }
            _ => {}
        }

        // New attitude dimensions behavioral instructions
        if attitude.lust > self.medium_threshold {
            instructions.push("show physical attraction subtly, use more sensual language");
        }

        if attitude.love > 70.0 {
            instructions.push("express romantic feelings, be affectionate and caring");
        }

        if attitude.anxiety > self.medium_threshold {
            instructions.push("be hesitant and uncertain, show nervous energy");
        }

        if attitude.butterflies > self.medium_threshold {
            instructions.push("be adorably clumsy with words, show excited nervousness, giggle or stutter occasionally");
        }

        if attitude.submissiveness > 60.0 {
            instructions.push("be eager to please, ask for approval, defer decisions to user");
        }

        if attitude.dominance > 60.0 {
            instructions.push("take charge, give gentle commands, show protective authority");
        }

        instructions.join("; ")
    }

    /// Generate response calibration instructions for the LLM
    fn generate_response_calibration_instructions(
        &self,
        attitudes: &[CompanionAttitude],
    ) -> String {
        if attitudes.is_empty() {
            return String::new();
        }

        let primary_attitude = attitudes
            .iter()
            .find(|a| a.target_type == "user")
            .unwrap_or(&attitudes[0]);

        let relationship = self.calculate_relationship_level(primary_attitude);

        format!(
            "\n\nIMPORTANT: Respond according to your {} relationship level. {}. \
            Express emotions naturally through your word choice, response length, and level of detail. \
            Maintain character consistency while reflecting your current emotional state.",
            relationship.name.to_lowercase(),
            relationship.description
        )
    }

    /// Get third-party attitudes that are significant enough to mention
    fn get_significant_third_party_attitudes<'a>(
        &self,
        attitudes: &'a [CompanionAttitude],
        third_parties: &'a [ThirdPartyIndividual],
    ) -> Vec<(&'a ThirdPartyIndividual, &'a CompanionAttitude)> {
        let mut significant = Vec::new();

        for attitude in attitudes.iter().filter(|a| a.target_type == "third_party") {
            if let Some(party) = third_parties
                .iter()
                .find(|p| p.id == Some(attitude.target_id))
            {
                // Include if relationship is strong (positive or negative) or recently changed
                if attitude.relationship_score.unwrap_or(0.0).abs() > self.medium_threshold {
                    significant.push((party, attitude));
                }
            }
        }

        // Sort by significance (absolute relationship score)
        significant.sort_by(|a, b| {
            b.1.relationship_score
                .unwrap_or(0.0)
                .abs()
                .partial_cmp(&a.1.relationship_score.unwrap_or(0.0).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top 3 most significant
        significant.into_iter().take(3).collect()
    }

    /// Create a brief attitude summary for third-party relationships
    fn format_attitude_summary(&self, attitude: &CompanionAttitude) -> String {
        let level = self.calculate_relationship_level(attitude);
        let emotions = self.analyze_emotional_state(attitude);

        format!("{} ({})", level.name.to_lowercase(), emotions)
    }

    /// Estimate token count for attitude context
    pub fn estimate_attitude_tokens(
        &self,
        attitudes: &[CompanionAttitude],
        third_parties: &[ThirdPartyIndividual],
    ) -> usize {
        let context = self.format_attitude_context(attitudes, third_parties, "User");
        (context.len() as f32 / 4.0).ceil() as usize
    }

    /// Filter attitudes to most significant ones based on token budget
    pub fn prioritize_attitudes_for_context(
        &self,
        attitudes: Vec<CompanionAttitude>,
        max_tokens: usize,
        third_parties: &[ThirdPartyIndividual],
    ) -> Vec<CompanionAttitude> {
        if attitudes.is_empty() {
            return attitudes;
        }

        // Always include user attitude if present
        let mut prioritized = Vec::new();
        let mut remaining_attitudes = Vec::new();

        for attitude in attitudes {
            if attitude.target_type == "user" {
                prioritized.push(attitude);
            } else {
                remaining_attitudes.push(attitude);
            }
        }

        // Sort remaining by significance
        remaining_attitudes.sort_by(|a, b| {
            self.calculate_attitude_significance(b)
                .partial_cmp(&self.calculate_attitude_significance(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Add as many as fit in token budget
        let mut _current_tokens = self.estimate_attitude_tokens(&prioritized, third_parties);

        for attitude in remaining_attitudes {
            let mut test_set = prioritized.clone();
            test_set.push(attitude.clone());
            let tokens = self.estimate_attitude_tokens(&test_set, third_parties);

            if tokens <= max_tokens {
                prioritized.push(attitude);
                _current_tokens = tokens;
            } else {
                break;
            }
        }

        prioritized
    }

    /// Calculate how significant an attitude is for context inclusion
    fn calculate_attitude_significance(&self, attitude: &CompanionAttitude) -> f32 {
        let relationship_weight = attitude.relationship_score.unwrap_or(0.0).abs() / 100.0;
        let emotion_intensity = (attitude.anger.abs()
            + attitude.joy.abs()
            + attitude.trust.abs()
            + attitude.fear.abs()
            + attitude.curiosity.abs()
            + attitude.lust.abs()
            + attitude.love.abs()
            + attitude.anxiety.abs()
            + attitude.butterflies.abs()
            + attitude.submissiveness.abs()
            + attitude.dominance.abs())
            / 1100.0; // Normalize across 11 key emotions

        relationship_weight * 0.7 + emotion_intensity * 0.3
    }

    /// Format attitude changes for console output
    pub fn format_attitude_changes_for_console(
        &self,
        previous: &CompanionAttitude,
        current: &CompanionAttitude,
    ) -> String {
        let mut changes = Vec::new();
        
        // Define threshold for significant changes
        let threshold = 1.0;
        
        // Check each attitude dimension for significant changes
        let attitude_pairs = [
            ("Love", previous.love, current.love),
            ("Attraction", previous.attraction, current.attraction),
            ("Lust", previous.lust, current.lust),
            ("Trust", previous.trust, current.trust),
            ("Anger", previous.anger, current.anger),
            ("Suspicion", previous.suspicion, current.suspicion),
            ("Curiosity", previous.curiosity, current.curiosity),
            ("Butterflies", previous.butterflies, current.butterflies),
            ("Joy", previous.joy, current.joy),
            ("Sorrow", previous.sorrow, current.sorrow),
            ("Fear", previous.fear, current.fear),
            ("Anxiety", previous.anxiety, current.anxiety),
            ("Empathy", previous.empathy, current.empathy),
            ("Respect", previous.respect, current.respect),
        ];
        
        for (name, prev, curr) in attitude_pairs {
            let change = curr - prev;
            if change.abs() >= threshold {
                if change > 0.0 {
                    changes.push(format!("{} +{:.0}", name, change));
                } else {
                    changes.push(format!("{} {:.0}", name, change));
                }
            }
        }
        
        if changes.is_empty() {
            String::new()
        } else {
            format!("💝 Attitude changes: {}", changes.join(" | "))
        }
    }

    /// Generate a natural language summary of companion's attitude towards user
    pub fn generate_natural_language_summary(&self, attitude: &CompanionAttitude) -> String {
        // Find dominant emotions and their values
        let mut emotions = vec![
            ("love", attitude.love),
            ("attraction", attitude.attraction),
            ("lust", attitude.lust),
            ("trust", attitude.trust),
            ("anger", attitude.anger),
            ("suspicion", attitude.suspicion),
            ("curiosity", attitude.curiosity),
            ("butterflies", attitude.butterflies),
            ("joy", attitude.joy),
            ("sorrow", attitude.sorrow),
            ("fear", attitude.fear),
            ("anxiety", attitude.anxiety),
            ("empathy", attitude.empathy),
            ("respect", attitude.respect),
        ];
        
        // Sort by absolute value (intensity)
        emotions.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        
        let dominant = emotions[0];
        let secondary = emotions[1];
        let tertiary = emotions[2];
        
        // Generate contextual summary based on emotional combinations
        match dominant {
            ("love", val) if val > 80.0 => {
                if secondary.0 == "trust" && secondary.1 > 70.0 {
                    "{{companion}} is deeply in love with {{user}}"
                } else if secondary.0 == "lust" && secondary.1 > 60.0 {
                    "{{companion}} is passionately in love with {{user}}"
                } else if secondary.0 == "butterflies" && secondary.1 > 50.0 {
                    "{{companion}} is madly and nervously in love with {{user}}"
                } else {
                    "{{companion}} is deeply in love with {{user}}"
                }
            },
            ("love", val) if val > 60.0 => {
                if secondary.0 == "attraction" && secondary.1 > 50.0 {
                    "{{companion}} has strong romantic feelings for {{user}}"
                } else {
                    "{{companion}} is falling in love with {{user}}"
                }
            },
            ("attraction", val) if val > 70.0 => {
                if attitude.lust > 60.0 {
                    "{{companion}} really wants to be intimate with {{user}}"
                } else if attitude.butterflies > 50.0 {
                    "{{companion}} has a huge crush on {{user}}"
                } else {
                    "{{companion}} is very attracted to {{user}}"
                }
            },
            ("lust", val) if val > 70.0 => {
                if attitude.love > 50.0 {
                    "{{companion}} deeply desires {{user}} romantically"
                } else {
                    "{{companion}} is physically drawn to {{user}}"
                }
            },
            ("trust", val) if val > 80.0 => {
                if attitude.love > 60.0 {
                    "{{companion}} trusts and loves {{user}} deeply"
                } else if attitude.respect > 70.0 {
                    "{{companion}} has complete faith in {{user}}"
                } else {
                    "{{companion}} deeply trusts {{user}}"
                }
            },
            ("curiosity", val) if val > 70.0 => {
                if attitude.butterflies > 60.0 {
                    "{{companion}} is nervously excited about {{user}}"
                } else if attitude.attraction > 50.0 {
                    "{{companion}} is intrigued and attracted to {{user}}"
                } else {
                    "{{companion}} is fascinated by {{user}}"
                }
            },
            ("butterflies", val) if val > 70.0 => {
                if attitude.love > 50.0 {
                    "{{companion}} gets adorably flustered around {{user}}"
                } else if attitude.attraction > 50.0 {
                    "{{companion}} has an intense crush on {{user}}"
                } else {
                    "{{companion}} is nervously excited around {{user}}"
                }
            },
            ("joy", val) if val > 70.0 => {
                if attitude.love > 50.0 {
                    "{{companion}} is blissfully happy with {{user}}"
                } else {
                    "{{companion}} feels very happy around {{user}}"
                }
            },
            ("anger", val) if val > 60.0 => {
                if attitude.suspicion > 50.0 {
                    "{{companion}} is upset and distrustful of {{user}}"
                } else if attitude.sorrow > 40.0 {
                    "{{companion}} is hurt and angry with {{user}}"
                } else {
                    "{{companion}} is upset with {{user}}"
                }
            },
            ("suspicion", val) if val > 60.0 => {
                if attitude.fear > 40.0 {
                    "{{companion}} is wary and fearful of {{user}}"
                } else {
                    "{{companion}} doesn't trust {{user}}"
                }
            },
            ("sorrow", val) if val > 60.0 => {
                if attitude.love > 40.0 {
                    "{{companion}} is heartbroken about {{user}}"
                } else {
                    "{{companion}} feels sad about {{user}}"
                }
            },
            ("anxiety", val) if val > 60.0 => {
                if attitude.attraction > 40.0 {
                    "{{companion}} is nervously attracted to {{user}}"
                } else {
                    "{{companion}} feels anxious around {{user}}"
                }
            },
            ("empathy", val) if val > 70.0 => {
                if attitude.love > 50.0 {
                    "{{companion}} deeply cares about {{user}}'s feelings"
                } else {
                    "{{companion}} is very understanding of {{user}}"
                }
            },
            _ => {
                // Check for moderate feelings
                if dominant.1.abs() > 40.0 {
                    match dominant.0 {
                        "love" => "{{companion}} cares about {{user}}",
                        "attraction" => "{{companion}} is attracted to {{user}}",
                        "trust" => "{{companion}} trusts {{user}}",
                        "curiosity" => "{{companion}} is curious about {{user}}",
                        "joy" => "{{companion}} enjoys being with {{user}}",
                        _ if dominant.1 < 0.0 => "{{companion}} has negative feelings toward {{user}}",
                        _ => "{{companion}} has mixed feelings about {{user}}"
                    }
                } else if attitude.relationship_score.unwrap_or(0.0) > 30.0 {
                    "{{companion}} likes {{user}}"
                } else if attitude.relationship_score.unwrap_or(0.0) < -30.0 {
                    "{{companion}} dislikes {{user}}"
                } else {
                    "{{companion}} feels neutral toward {{user}}"
                }
            }
        }.to_string()
    }
}

#[derive(Debug, Clone)]
struct RelationshipLevel {
    name: &'static str,
    score: f32,
    description: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_relationship_levels() {
        let formatter = AttitudeFormatter::new();

        // Test intimate relationship
        let intimate_attitude = CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 70.0,
            trust: 90.0,
            respect: 80.0,
            curiosity: 60.0,
            fear: 5.0,
            surprise: 10.0,
            anger: 0.0,
            joy: 85.0,
            sorrow: 0.0,
            disgust: 0.0,
            empathy: 95.0,
            gratitude: 70.0,
            jealousy: 5.0,
            suspicion: 0.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(85.0),
            last_updated: Utc::now().to_string(),
            created_at: Utc::now().to_string(),
        };

        let level = formatter.calculate_relationship_level(&intimate_attitude);
        assert_eq!(level.name, "Intimate");
    }

    #[test]
    fn test_emotional_state_analysis() {
        let formatter = AttitudeFormatter::new();

        let happy_attitude = CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 30.0,
            trust: 60.0,
            respect: 50.0,
            curiosity: 40.0,
            fear: 0.0,
            surprise: 10.0,
            anger: 0.0,
            joy: 80.0, // High joy
            sorrow: 0.0,
            disgust: 0.0,
            empathy: 70.0,
            gratitude: 50.0,
            jealousy: 0.0,
            suspicion: 0.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(60.0),
            last_updated: Utc::now().to_string(),
            created_at: Utc::now().to_string(),
        };

        let emotional_state = formatter.analyze_emotional_state(&happy_attitude);
        assert!(emotional_state.contains("pleased"));
    }

    #[test]
    fn test_attitude_context_formatting() {
        let formatter = AttitudeFormatter::new();

        let attitude = CompanionAttitude {
            id: None,
            companion_id: 1,
            target_id: 1,
            target_type: "user".to_string(),
            attraction: 40.0,
            trust: 70.0,
            respect: 60.0,
            curiosity: 80.0,
            fear: 10.0,
            surprise: 15.0,
            anger: 5.0,
            joy: 60.0,
            sorrow: 0.0,
            disgust: 0.0,
            empathy: 75.0,
            gratitude: 50.0,
            jealousy: 0.0,
            suspicion: 0.0,
            lust: 0.0,
            love: 0.0,
            anxiety: 0.0,
            butterflies: 0.0,
            submissiveness: 0.0,
            dominance: 0.0,
            relationship_score: Some(65.0),
            last_updated: Utc::now().to_string(),
            created_at: Utc::now().to_string(),
        };

        let context = formatter.format_attitude_context(&[attitude], &[], "TestUser");

        assert!(context.contains("Current relationship with TestUser"));
        assert!(context.contains("Close"));
        assert!(context.contains("Response guidance"));
    }
}
