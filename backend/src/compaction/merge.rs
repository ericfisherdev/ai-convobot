//! The production [`SummaryMerger`]: folds a checkpoint's previous rolling
//! summary and detail into one compressed rolling summary, via the same
//! [`Extractor`] seam (#183) #173's extraction pass uses. Unit-tested with
//! `crate::llm::FakeExtractor`, no GGUF needed.

use crate::compaction::commit::{MergeError, SummaryMerger};
use crate::llm::Extractor;

/// Wraps an [`Extractor`] — in production, `llm::ResidentExtractor`, which
/// prefers the configured extraction model and falls back to the chat model
/// — to compress a rolling summary that has grown past its token budget.
/// Greedy, no grammar: this is prose compression, not structured
/// extraction.
pub struct LlmSummaryMerger<'a> {
    pub extractor: &'a dyn Extractor,
}

impl SummaryMerger for LlmSummaryMerger<'_> {
    fn merge(&self, rolling: &str, budget_tokens: usize) -> Result<String, MergeError> {
        let max_words = budget_tokens * 3 / 4;
        let prompt = format!(
            "Compress the following story summary to at most {max_words} words. Keep every named person, place, injury, promise and open question. No dates or clock times.\n\n{rolling}"
        );
        self.extractor
            .complete(&prompt, budget_tokens)
            .map_err(|e| MergeError::ModelUnavailable(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::FakeExtractor;

    #[test]
    fn merge_returns_the_extractors_completion() {
        let extractor = FakeExtractor::returning(vec![Ok("compressed summary".to_string())]);
        let merger = LlmSummaryMerger {
            extractor: &extractor,
        };

        let result = merger.merge("a very long rolling summary", 100).unwrap();

        assert_eq!(result, "compressed summary");
        assert_eq!(extractor.prompts.lock().unwrap().len(), 1);
        assert!(extractor.prompts.lock().unwrap()[0].contains("a very long rolling summary"));
        assert!(extractor.grammars.lock().unwrap()[0].is_none());
    }

    #[test]
    fn merge_maps_an_extractor_error_to_model_unavailable() {
        let extractor =
            FakeExtractor::returning(vec![Err(std::io::Error::other("model not loaded"))]);
        let merger = LlmSummaryMerger {
            extractor: &extractor,
        };

        let err = merger.merge("a rolling summary", 100).unwrap_err();

        assert!(
            matches!(err, MergeError::ModelUnavailable(msg) if msg.contains("model not loaded"))
        );
    }

    #[test]
    fn merge_asks_for_at_most_three_quarters_of_the_budget_in_words() {
        let extractor = FakeExtractor::returning(vec![Ok("ok".to_string())]);
        let merger = LlmSummaryMerger {
            extractor: &extractor,
        };

        merger.merge("text", 100).unwrap();

        assert!(extractor.prompts.lock().unwrap()[0].contains("at most 75 words"));
    }
}
