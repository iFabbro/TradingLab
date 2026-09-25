# Security Policy

## Scope

TradingLab is a research and supervised-execution project. It is not presented as a production trading system, and no live brokerage credentials belong in the repository.

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability reporting / Security Advisories for this repository rather than opening a public issue.

Include, when available:

- affected file or component;
- concise description of the issue;
- reproduction steps or proof of concept;
- potential impact;
- suggested mitigation.

## Secrets

Never commit API keys, access tokens, passwords, private keys, broker credentials, `.env` files, or local runtime state. Use environment variables or local configuration that is excluded by `.gitignore`.

If a secret is accidentally committed, treat it as compromised: revoke or rotate it first, then remove it from the repository history.
