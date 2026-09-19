# Feedback schema

Each line is JSON:

```json
{"prompt":"Write a function...","chosen":"def ...","rejected":"...","weight":1.0,"source":"human|tests|teacher"}
```

- chosen is required.
- rejected is optional; when absent, use SFT.
- when rejected is present, DPO-style pairwise tuning can be used.
- keep a stable replay sample of prior high-quality rows during continual training to reduce catastrophic forgetting.
- every feedback batch should carry provenance and an evaluation result before it is allowed to change the promoted checkpoint.
