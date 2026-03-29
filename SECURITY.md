# Security policy

## Supported versions

We address security issues in the **latest release** on PyPI and on the **`main`** branch of this repository. Older versions may not receive backports.

## Reporting a vulnerability

If you believe you have found a **security vulnerability** in ProtectRAG (for example: a reliable way to evade screening that you do not want to disclose publicly yet, or an issue that could compromise applications using the library in an unexpected way):

1. **Do not** open a public GitHub issue with exploit details.
2. Use **[GitHub Security Advisories](https://github.com/Ritvik777/ProtectRAG/security/advisories/new)** to report privately, or contact the maintainers through the contact options available on the repository.

Please include:

- A short description of the issue and its impact
- Steps to reproduce (code or scenario)
- Affected version(s) if known

We will work with you on a coordinated disclosure timeline when possible.

## Scope notes

ProtectRAG is a **library**: it does not run as a hosted service. “Vulnerabilities” might include logic flaws in detection, unsafe defaults in integrations, or documentation that leads to insecure deployment. Heuristic and LLM-based detection are **not** guarantees; limitations are described in the [README](README.md).

## Prompt injection vs. library security

Many users care about **prompt-injection effectiveness** (false negatives/positives). Those are important product issues and are usually fine to discuss in **public issues** and PRs. Use **private disclosure** when the issue is a **defect in the library or its integrations** that could harm users who deploy it in good faith (e.g. crash, data leak, or a documented guarantee that is provably broken).
