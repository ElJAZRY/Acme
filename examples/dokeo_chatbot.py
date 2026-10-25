"""A tiny, trainable chatbot: bag-of-words + multinomial Naive Bayes.

Run with Python 3.11+: `python intent_chatbot.py` or `python intent_chatbot.py --demo`.
No packages, network, API keys, message logging or saved model files are needed.

The labelled training sentences below are synthetic examples. The model learns
which intent best fits a message, then selects a fixed reply. It is not an LLM.
Scores are uncalibrated model estimates, not verified confidence or accuracy.
This demonstration has no production evaluation, fairness assessment or approval.
Unrecognised words and low scores trigger abstention, but this does not reliably
detect every out-of-domain or misleading message. Do not use it for legal,
medical, financial, access-control or other consequential decisions.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import re

MODEL_VERSION = "0.1-demo"
TRAINING_DATA = (
    ("hello hi good morning", "greeting"),
    ("hey there nice to meet you", "greeting"),
    ("hello good evening", "greeting"),
    ("what time does the shop open", "hours"),
    ("tell me your opening hours", "hours"),
    ("when are you open on weekdays", "hours"),
    ("how can I return an item for a refund", "returns"),
    ("I need help returning my order", "returns"),
    ("what is your refund and return policy", "returns"),
    ("goodbye see you later", "goodbye"),
    ("bye thanks that is all", "goodbye"),
    ("thank you goodbye", "goodbye"),
)
REPLIES = {
    "greeting": "Hello! Ask me about this fictional shop's hours or returns.",
    "hours": "Demo answer: this fictional shop opens Monday to Friday, 09:00–17:00.",
    "returns": "Demo answer: contact the fictional shop's support team about a return.",
    "goodbye": "Goodbye!",
}
FALLBACK = "I don't know. This demo only covers greetings, shop hours and returns."


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


@dataclass(frozen=True)
class Prediction:
    intent: str | None
    score: float
    reply: str


class IntentChatbot:
    def __init__(self, minimum_score: float = 0.65):
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between 0 and 1")
        self.minimum_score = minimum_score
        self.documents: Counter[str] = Counter()
        self.words: dict[str, Counter[str]] = defaultdict(Counter)
        self.vocabulary: set[str] = set()
        self.word_totals: Counter[str] = Counter()

    def train(self, examples=TRAINING_DATA) -> "IntentChatbot":
        """Fit a fresh model; each training call replaces earlier training counts."""
        self.documents.clear()
        self.words.clear()
        self.vocabulary.clear()
        self.word_totals.clear()
        for sentence, intent in examples:
            if intent not in REPLIES:
                raise ValueError("Each training intent needs a fixed reply")
            observed = tokens(sentence)
            if not observed:
                raise ValueError("Each training sentence needs at least one word")
            self.documents[intent] += 1
            self.words[intent].update(observed)
            self.word_totals[intent] += len(observed)
            self.vocabulary.update(observed)
        if not self.documents:
            raise ValueError("Training examples cannot be empty")
        return self

    def predict(self, text: str) -> Prediction:
        if not self.documents:
            raise ValueError("Train the model before prediction")
        observed = Counter(word for word in tokens(text) if word in self.vocabulary)
        if not observed:
            return Prediction(None, 0.0, FALLBACK)
        total_documents = sum(self.documents.values())
        scores = {}
        for intent, count in self.documents.items():
            # Log probabilities avoid numerical underflow; +1 is Laplace smoothing.
            log_score = math.log(count / total_documents)
            denominator = self.word_totals[intent] + len(self.vocabulary)
            log_score += sum(
                n * math.log((self.words[intent][word] + 1) / denominator)
                for word, n in observed.items()
            )
            scores[intent] = log_score
        best = max(scores, key=scores.get)
        normalizer = sum(math.exp(value - scores[best]) for value in scores.values())
        score = 1.0 / normalizer
        if score < self.minimum_score:
            return Prediction(None, score, FALLBACK)
        return Prediction(best, score, REPLIES[best])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo", action="store_true", help="Show sample predictions and exit"
    )
    args = parser.parse_args()
    model = IntentChatbot().train()
    if args.demo:
        for message in (
            "hello hi",
            "your opening hours",
            "return my order",
            "bye goodbye see you later",
            "quantum teleportation",
        ):
            answer = model.predict(message)
            print(
                json.dumps(
                    {
                        "model_version": MODEL_VERSION,
                        "message": message,
                        "intent": answer.intent,
                        "uncalibrated_score": round(answer.score, 4),
                        "reply": answer.reply,
                    },
                    ensure_ascii=False,
                )
            )
        return
    print("Local ML chatbot demo. Type /quit to exit. Messages are not stored.")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message == "/quit":
            break
        print("Bot:", model.predict(message).reply)


if __name__ == "__main__":
    main()
