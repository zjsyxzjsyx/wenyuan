from eval.build_multiturn import build_row

ROW = {"id": "so-1", "library": "redis", "question": "How do I make a Redis key expire?", "gold_urls": ["u"],
       "source": "s"}


class Writer:
    def __init__(self, reply):
        self.reply = reply

    def chat_json(self, messages, temperature=0.0):
        return self.reply


def test_followup_hides_its_subject_and_keeps_gold():
    row = build_row(Writer({"first_turn": "What is a Redis key?", "assistant_reply": "A name for a value.",
                            "followup": "How do I make it expire?", "subject": "a Redis key"}), ROW)
    assert row["question"] == "How do I make it expire?" and row["gold_urls"] == ["u"]
    assert row["history"][0] == {"role": "user", "content": "What is a Redis key?"}
    assert row["original_question"] == ROW["question"]


def test_followup_that_still_names_the_subject_is_dropped():
    leaky = {"first_turn": "x", "assistant_reply": "y", "followup": "How do I make a Redis key expire?",
             "subject": "Redis key"}
    assert build_row(Writer(leaky), ROW) is None
    assert build_row(Writer({"first_turn": "x", "followup": "it?"}), ROW) is None      # no subject reported



def test_hiding_only_the_library_name_is_dropped():
    only_library = {"first_turn": "What is Redis?", "assistant_reply": "A data store.",
                    "followup": "How do I make a key expire in it?", "subject": "Redis"}
    assert build_row(Writer(only_library), ROW) is None
