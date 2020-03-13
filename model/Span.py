def get_span_candidates(text_len, max_sentence_length, max_mention_width):
        """Get a list of candidate spans up to length W.
        Args:
            text_len: Tensor of [num_sentences,]
            max_sentence_length: Integer scalar.
            max_mention_width: Integer.
        """
        candidate_starts = 0
        candidate_ends = 0
        candidate_mask = 0
        return candidate_starts, candidate_ends, candidate_mask