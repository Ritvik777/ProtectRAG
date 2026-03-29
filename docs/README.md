# ProtectRAG documentation folder

## Interactive website (GitHub Pages)

This folder includes a **static microsite** for newcomers:

- **[index.html](index.html)** — overview, Mermaid diagrams, install snippet, and a **browser-only Q&amp;A chat** (scripted answers; no backend).
- **`css/site.css`** · **`js/site.js`** — styles and chat logic.

### Enable GitHub Pages

1. Push the `docs/` folder to your default branch.
2. On GitHub: **Settings → Pages**.
3. **Build and deployment → Source:** *Deploy from a branch*.
4. **Branch:** your default branch, **folder:** `/docs`, Save.

The site will be available at:

`https://<your-username>.github.io/ProtectRAG/`

(Repository name must match if using project Pages; adjust if your repo name differs.)

**Note:** [`.nojekyll`](.nojekyll) is present so GitHub serves files as static assets without Jekyll processing.

---

## Long-form docs

**Start here:** [README.md](../README.md) in the repository root — installation, architecture, usage, integrations, observability, evals, and publishing.

**Contributors:** [CONTRIBUTING.md](../CONTRIBUTING.md) · **Security:** [SECURITY.md](../SECURITY.md)
