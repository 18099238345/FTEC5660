#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        reasoning_effort="low",
        # The vision model spends part of this budget on hidden reasoning, even
        # at low effort, before emitting the compact JSON response.
        max_tokens=8192,
        timeout=120,
        max_retries=2,
    )

    extraction_rules = """
You are an exact OCR accountant reading ONE Hong Kong supermarket receipt.
Extract the values needed for the homework. Read the printed transaction itself;
ignore loyalty-point balances, card balances, dates, phone numbers, quantities,
barcodes, device/reference numbers, change, and repeated payment-card details.

Definitions:
- final_payment: the transaction total AFTER the receipt's ROUNDING line. It is
  normally the OCTOPUS, VISA, CASH, or other tender amount beside the summary.
  Do not use cash tendered, change, remaining card value, or a repeated
  "amount deducted" line lower on the receipt.
- subtotal: the receipt's SUBTOTAL / Chinese 小計 amount BEFORE ROUNDING. It is
  already after discounts.
- rounding: the signed ROUNDING adjustment; use 0.00 when absent.
- discounts: every genuine discount/promotion/coupon/member/app/packaging-
  damage reduction applied in the merchandise section, including item-level
  "Buy N Save", percentage-off, member-price and app-upgrade reductions.
  Record each discount once as a POSITIVE amount. Do not include ROUNDING,
  change, tender/payment lines, balances, ordinary positive item prices, or
  zero-value coupon lines.
- positive_item_total: independently add every positive merchandise line total
  above SUBTOTAL exactly once. Use the right-column extended price, not a unit
  price printed after SP and not the quantity. Exclude SUBTOTAL, payments,
  balances, change, deposits, dates, identifiers, and every negative line.
  A positive PLASTIC BAG CHARGING / bag fee (commonly $1.00 or $2.00) is part
  of the transaction and MUST be included exactly once. A $0.00 coupon line
  contributes nothing and is not a discount.

Scan the complete receipt. First locate the bottom summary, then scan the
merchandise section line by line for all discount amounts. Check that
subtotal + rounding = final_payment (allow ordinary cent rounding), and check
your discount arithmetic before answering. Independently sum the positive item
line totals as a cross-check; do not recompute or replace the printed subtotal.

Return JSON only, with decimal strings and exactly this shape:
{{"final_payment":"0.00","subtotal":"0.00","rounding":"0.00",
 "discounts":[{{"label":"printed label","amount":"0.00"}}],
 "discount_total":"0.00","positive_item_total":"0.00",
 "amount_without_discounts":"0.00"}}
discount_total must equal the sum of the discount amounts, and
positive_item_total and amount_without_discounts must both equal
subtotal + discount_total.
""".strip()

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_rules),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Extract and cross-check this receipt. Return JSON only.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}", "detail": "high"},
                    },
                ],
            ),
        ]
    )
    extractor = extraction_prompt | model | StrOutputParser()

    review_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", extraction_rules),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": (
                            "Independently re-read the receipt and audit the draft below. "
                            "Correct every OCR, classification, omission, sign, and arithmetic "
                            "error. Pay special attention to all negative merchandise lines and "
                            "never count ROUNDING as a discount. Return only the corrected JSON.\n\n"
                            "DRAFT:\n{draft}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}", "detail": "high"},
                    },
                ],
            ),
        ]
    )

    # Preserve both passes. Experimental vision models can occasionally return
    # an empty audit response even when extraction succeeded, so the caller can
    # safely fall back to the still-validated first pass instead of losing it.
    with_draft = RunnablePassthrough.assign(draft=extractor)
    reviewer = review_prompt | model | StrOutputParser()
    return with_draft.assign(review=reviewer)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def money(value: Any, field: str) -> Decimal:
        if value is None:
            raise ValueError(f"missing {field}")
        cleaned = str(value).strip().upper().replace("HK$", "")
        cleaned = cleaned.replace("$", "").replace(",", "").replace(" ", "")
        amount = Decimal(cleaned).quantize(Decimal("0.01"))
        if not amount.is_finite():
            raise ValueError(f"non-finite {field}")
        return amount

    def parse_receipt(value: Any) -> tuple[Decimal, Decimal]:
        candidates = (
            [value.get("review"), value.get("draft")]
            if isinstance(value, dict)
            else [value]
        )

        def parse_candidate(candidate: Any) -> tuple[Decimal, Decimal]:
            text = response_text(candidate)
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("model response did not contain a JSON object")
            data = json.loads(text[start : end + 1])

            final_payment = money(data.get("final_payment"), "final_payment")
            subtotal = money(data.get("subtotal"), "subtotal")
            rounding = money(data.get("rounding", "0.00"), "rounding")
            if final_payment < 0 or subtotal < 0:
                raise ValueError("receipt totals must be non-negative")
            if abs((subtotal + rounding) - final_payment) > Decimal("0.11"):
                raise ValueError("subtotal, rounding, and final payment are inconsistent")

            raw_discounts = data.get("discounts", [])
            if not isinstance(raw_discounts, list):
                raise ValueError("discounts must be a list")
            discounts: list[Decimal] = []
            for index, item in enumerate(raw_discounts):
                raw_amount = item.get("amount") if isinstance(item, dict) else item
                amount = abs(money(raw_amount, f"discounts[{index}]"))
                if amount:
                    discounts.append(amount)

            discount_total = sum(discounts, Decimal("0.00")).quantize(
                Decimal("0.01")
            )
            stated_discount_total = data.get("discount_total")
            if stated_discount_total is not None:
                stated = abs(money(stated_discount_total, "discount_total"))
                if abs(stated - discount_total) > Decimal("0.01"):
                    raise ValueError("discount list does not match discount_total")

            without_discounts = (subtotal + discount_total).quantize(Decimal("0.01"))
            positive_item_total = money(
                data.get("positive_item_total"), "positive_item_total"
            )
            if abs(positive_item_total - without_discounts) > Decimal("0.01"):
                raise ValueError("positive item total does not confirm discounts")
            stated_without = data.get("amount_without_discounts")
            if stated_without is not None:
                stated = money(stated_without, "amount_without_discounts")
                if abs(stated - without_discounts) > Decimal("0.01"):
                    raise ValueError("amount_without_discounts is inconsistent")
            return final_payment, without_discounts

        parsed: list[tuple[Decimal, Decimal]] = []
        parse_error: Exception | None = None
        for candidate in candidates:
            try:
                parsed.append(parse_candidate(candidate))
            except (ValueError, TypeError, json.JSONDecodeError, InvalidOperation) as exc:
                parse_error = exc
        if not parsed:
            raise ValueError("no model pass contained a valid receipt extraction") from parse_error
        if any(result != parsed[0] for result in parsed[1:]):
            raise ValueError("the extraction and audit passes disagree")
        return parsed[0]

    requests = [{"image_url": image_data_url(path)} for path in images]
    outputs = chain.batch(
        requests,
        # Sequential calls are slower but avoid quality drift observed when the
        # experimental vision endpoint handles several long receipts at once.
        config={"max_concurrency": 1},
        return_exceptions=True,
    )

    paid_total = Decimal("0.00")
    no_discount_total = Decimal("0.00")
    for path, request, output in zip(images, requests, outputs, strict=True):
        last_error: Exception | None = output if isinstance(output, Exception) else None
        if not isinstance(output, Exception):
            try:
                paid, no_discount = parse_receipt(output)
            except (ValueError, TypeError, json.JSONDecodeError, InvalidOperation) as exc:
                last_error = exc
            else:
                paid_total += paid
                no_discount_total += no_discount
                continue

        # Retry only the problematic receipt. This covers transient API failures,
        # truncated output, malformed JSON, and disagreement between the two passes.
        for _ in range(2):
            try:
                paid, no_discount = parse_receipt(chain.invoke(request))
            except Exception as exc:
                last_error = exc
                continue
            paid_total += paid
            no_discount_total += no_discount
            break
        else:
            raise RuntimeError(
                f"could not extract a valid result from {path.name}: {last_error}"
            ) from last_error

    return {
        QUERY_1: f"HK${paid_total.quantize(Decimal('0.01')):.2f}",
        QUERY_2: f"HK${no_discount_total.quantize(Decimal('0.01')):.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
