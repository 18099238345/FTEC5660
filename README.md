# FTEC5660 Homework 1: Receipt Chain

Build a LangChain pipeline that reads every supermarket receipt in a folder
with the vision-capable DeepSeek Flash model and answers these two questions:

1. How much money did I spend in total for these bills?
2. How much would I have had to pay without the discount?

For this homework, **amount spent** means the final payment after the receipt's
rounding line. **Without the discount** means the sum of the original positive
item prices: add back every promotion, coupon, member, app, packaging-damage,
and percentage discount, but do not add back rounding.

## Student task

Only edit the two functions in `hw1.py` that contain `### YOUR CODE HERE`:

- `build_chain()` creates your LangChain chain.
- `answer_queries()` runs the chain on the receipt images and returns one final
  response for each question.

You may use prompt chaining, routing, parallel calls, reflection, or a
combination. Your final responses should each contain one HKD amount. Do not
hard-code filenames or public answers; grading uses unseen receipt folders.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your DeepSeek key after `DEEPSEEK_API_KEY=` in `.env`, then run:

```bash
python3 hw1.py --image-folder public_test
```

The program creates `results.csv` in the current directory. Its columns are
`query`, `model_response`, and `correctness`. The public answers are in
`public_test/ground_truth.json`. The starter intentionally returns the dummy
response `please design your chain to answer these two queries.` so it runs
before you add any API code.

The required model is `deepseek-v4-flash-vision-exp`, the vision-capable
DeepSeek Flash model. JPEG, PNG, GIF, and WebP inputs are accepted by the
homework runner.


## Homework 1 solution

```mermaid
flowchart LR
    A[Receipt images] --> B[Encode each image as a data URL]
    B --> C[DeepSeek vision extraction pass]
    C --> D[DeepSeek vision audit pass]
    D --> E[Validate JSON and monetary identities]
    E --> F[Decimal aggregation in Python]
    F --> G[Two exact HKD responses]
    C -. fallback if the audit is empty or invalid .-> E
```

The solution builds one LangChain pipeline around
`deepseek-v4-flash-vision-exp` and applies it independently to every receipt.
The first vision pass extracts the final payment, subtotal, rounding adjustment,
and each genuine discount line into JSON; a second pass re-reads the same image
and audits the extraction. Python then validates the JSON and the monetary
identities, falls back to a valid first pass if necessary, and uses `Decimal` to
sum the final payments and the subtotals plus discounts without floating-point
rounding errors. Bounded retries handle transient or malformed model responses,
while the final dictionary contains only the two required `HK$0.00`-style
amounts.
