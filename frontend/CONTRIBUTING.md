# Contributing

Thanks for considering a contribution. For anything larger than a small
fix, open an issue first so the scope is agreed before you write the
code.

## Getting started

```bash
npm install
cp .env.example .env   # point PUBLIC_TRITON_API_URL at your backend
npm run dev             # http://localhost:5173
```

You need a running OpenProcessor backend (OpenSearch + the curation API,
served under `PUBLIC_API_PREFIX`) for the app to do anything useful — see
that repo's README. There is no local database or mock-data mode.

## Before opening a PR

```bash
npm run check   # svelte-check + tsc, strict mode, must be 0 errors
npm run lint     # prettier + eslint
npm test         # vitest
npm run build    # production build must succeed
```

All four must pass — CI runs the same checks, plus the stubbed
Playwright suite (`npm run test:e2e`) and a gitleaks secret scan, and
will block merge otherwise.

## Screenshots and sample data

Screenshots and sample data must come from public datasets only (for
example COCO val2017). Never commit an image from a private dataset, a
photograph of a person or vehicle you don't have the rights to, or a
screenshot showing a real deployment's host names, paths or class counts.

## Conventions

See `CLAUDE.md` for the authoritative reference on:

- Architecture (SvelteKit 2, Svelte 5 runes only — no `writable()`/
  `readable()`), route table, keyboard-shortcut reservations.
- Style: 2-space indent, dark theme only, no emoji, no gradients,
  Apple system colors only.
- Data-integrity rules (optimistic UI + rollback, OCC conflict
  handling, test-holdout filtering) — don't bypass these to "simplify"
  a change.

If you change a route, a keyboard shortcut, or add a feature flag,
update `CLAUDE.md`'s tables in the same PR — they're meant to stay
authoritative, not drift from the code.

## Commit messages

Conventional commits: `<type>(<scope>): <summary>`, imperative mood.
Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `build`,
`chore`, `security`, `perf`. Update `CHANGELOG.md` under `[Unreleased]`
for anything user-visible.

## Reporting bugs / requesting features

Use the issue templates — they ask for what's actually needed to
reproduce or scope the work (repro steps, environment, screenshots).

## Code of conduct

Be respectful and constructive. Disagreements about approach are fine
and expected; personal attacks are not.
