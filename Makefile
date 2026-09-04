.DEFAULT_GOAL := check
.NOTPARALLEL:
.PHONY: fmt fmt-check lint typecheck test-fe build-fe clippy test-be check

fmt:
	cd backend && cargo fmt

fmt-check:
	cd backend && cargo fmt --check

lint:
	npm run lint

typecheck:
	npx tsc --noEmit

test-fe:
	npm run test:run

build-fe:
	npm run build

clippy:
	cd backend && cargo clippy --all-targets -- -D warnings

test-be:
	cd backend && cargo test

check: fmt-check lint typecheck test-fe build-fe clippy test-be
