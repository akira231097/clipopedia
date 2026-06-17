# Contributing

Thanks for taking a look! This is primarily a reference project, but improvements
are welcome.

## Setup

```bash
pip install -e ".[dev]"     # core + dev tooling (enough for the demo and tests)
pytest                      # run the suite
ruff check . && ruff format .
mypy src
```

## Guidelines

- **Depend on ports, not vendors.** New pipeline logic should rely on the
  `Protocol`s in `ports.py`. A new external service is a new adapter in
  `adapters/` plus a line in `factory.py`.
- **Keep the core import-light.** Heavy SDKs are imported lazily so the demo and
  tests run without them. Don't add a top-level import of a `live`-only package
  to a core module.
- **Add a test.** New retrieval logic should come with a unit test that runs on
  the in-memory backend.
- **No secrets, ever.** Configuration is environment-driven; `.env` is
  git-ignored and only `.env.example` (placeholders) is committed.
