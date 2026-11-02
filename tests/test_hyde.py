from app.query.hyde import hypothetical_passage


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat_json(self, messages, temperature=0.0):
        self.prompts.append(messages[0]["content"])
        return self.reply


def test_passage_is_asked_for_in_the_libraries_own_words():
    llm = FakeLLM({"passage": "  Use lifespan events to start work when the app starts.  "})
    assert hypothetical_passage(llm, "How do I run a thread in the background?", "fastapi") == \
        "Use lifespan events to start work when the app starts."
    assert "official fastapi documentation" in llm.prompts[0] and "run a thread" in llm.prompts[0]


def test_missing_passage_is_empty_not_an_error():
    assert hypothetical_passage(FakeLLM({}), "q") == ""
