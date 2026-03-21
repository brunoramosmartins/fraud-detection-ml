# ADR-005 — `app.state` for Runtime Model Storage

## Status

Accepted

## Context

The scoring API must load a trained model at startup and make it available
to request handlers. Three patterns were considered for storing the runtime
model reference:

1. **Module-level global variable:** define `_model = None` at the top of
   `app/main.py` and populate it in the `startup` event.
2. **`app.state`:** use FastAPI's application state object to attach the
   model and related metadata to the application instance.
3. **Dependency injection with `Depends`:** create a callable that loads or
   returns the cached model, injected into each endpoint via `Depends`.

## Decision

`app.state` populated during the `lifespan` context manager.

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load_deployed_model(app)   # writes to app.state
    yield
    # shutdown cleanup (if needed)

def _load_deployed_model(app: FastAPI) -> None:
    # ...
    app.state.model = model
    app.state.feature_list = meta["feature_list"]
    app.state.threshold = meta["metrics"]["best_threshold"]
```

Handlers access the model via `request.app.state.model`.

## Consequences

**Positive:**
- **Test isolation:** in tests, the `TestClient` receives the application
  instance. Monkeypatching `_load_deployed_model` before the lifespan runs
  allows injecting a stub model cleanly — no global state leaks between
  tests.
  ```python
  monkeypatch.setattr("app.main._load_deployed_model", lambda app: ...)
  ```
- **Explicitness:** the scope of the model is the application lifecycle,
  not the module import. Starting a second application instance (e.g., for
  testing) creates a second independent `app.state` — no cross-contamination.
- **`lifespan` compliance:** the FastAPI maintainers deprecated `on_event`
  decorators in favor of `lifespan`. Using `app.state` with `lifespan` is
  the idiomatic pattern in FastAPI ≥0.93.
- **No dependency injection overhead:** for an object that is loaded once
  at startup and is effectively immutable during the process lifetime,
  `Depends` adds request-level overhead with no benefit.

**Negative:**
- `request.app.state` is accessed by attribute name (string), which is not
  type-checked by mypy without additional stubs. A typed wrapper class would
  improve IDE support.
- The model is loaded synchronously in the async lifespan context. For very
  large models, this blocks the event loop during startup. An async load
  (e.g., `asyncio.to_thread`) would be cleaner but adds complexity.
- `app.state` is not thread-safe for writes during request handling. This
  is acceptable because the model is written once at startup and is read-only
  thereafter. If live model swapping were required, a `threading.Lock`
  would be necessary.

**Why not module-level globals:** the core issue with module-level globals
is that they make tests flaky. If `_model` is set in one test, it leaks
into subsequent tests unless explicitly reset. This happened during
development when `TestClient` was created at module scope — the model
loaded in one test's setup was visible to all other tests. `app.state`
scopes the model to the application instance and eliminates this class of bug.
