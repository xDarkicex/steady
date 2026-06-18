# Contributing to steady

## Issues

- Search existing issues before opening a new one
- Include a minimal reproduction case
- For bug reports, include Go version and platform

## Pull Requests

- Keep PRs focused on a single change
- Run `go test -race ./...` and `staticcheck ./...` before submitting
- Every exported function must have a Go doc comment starting with the name
- Cyclomatic complexity must not exceed 10 per function
- All allocations must use `github.com/xDarkicex/memory` — no `make` or `new` for data planes
- Add tests for new functionality; coverage must not decrease

## Code Style

- Follow standard Go conventions (`gofmt`, `go vet`)
- Go doc comments on all exported types, functions, methods, and constants
- No external dependencies beyond `xDarkicex/memory`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
