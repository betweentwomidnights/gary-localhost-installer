# One-time legacy SA3 adapter export

`lora_ckpt_export.py` is copied verbatim from sa3.cpp at commit
`7d5c35c7d853702ffa7cd93d6762d25796a649bd`, under `tools/` in
https://github.com/betweentwomidnights/sa3.cpp.

Keep generic export changes and tests upstream; refresh this copy together.
Gary embeds it at build time, records its SHA-256 with every exported checkpoint,
and runs it on CPU through the existing SA3 environment before cleanup. Native
inference, training and reuse of verified exports do not require Python.
