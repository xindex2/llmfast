BINARY := llmfast
PKG    := ./cmd/gateway

.PHONY: help build build-linux run mock dev agent plan test test-race lint fmt clean models

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## Build the gateway, agent and planner for the host platform
	go build -trimpath -ldflags="-s -w" -o dist/$(BINARY) $(PKG)
	go build -trimpath -ldflags="-s -w" -o dist/$(BINARY)-agent ./cmd/agent
	go build -trimpath -ldflags="-s -w" -o dist/llmplan ./cmd/llmplan

build-linux: ## Cross-compile everything for a Linux server (pure Go, no cgo)
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
		go build -trimpath -ldflags="-s -w" -o dist/$(BINARY)-linux-amd64 $(PKG)
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
		go build -trimpath -ldflags="-s -w" -o dist/$(BINARY)-agent-linux-amd64 ./cmd/agent
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
		go build -trimpath -ldflags="-s -w" -o dist/llmplan-linux-amd64 ./cmd/llmplan

agent: ## Run the node agent locally (detects this machine's hardware)
	LLMFAST_AGENT_TOKEN=$${LLMFAST_AGENT_TOKEN:-dev-agent-token} \
		go run ./cmd/agent -listen 127.0.0.1:9900 -name local -state-dir /tmp/llmfast-agent

hardware: ## Print the hardware the agent detects on this machine
	@go run ./cmd/agent -hardware -state-dir /tmp/llmfast-agent

plan: ## Check whether a model fits: make plan MODEL=Qwen/Qwen3-8B
	@go run ./cmd/llmplan $(MODEL) --compare

mock: ## Run the mock vLLM backend on :8000
	go run ./cmd/mockvllm -addr :8000

run: ## Run the gateway against config/dev.yaml
	go run $(PKG) -config config/dev.yaml

dev: ## Run mock backend and gateway together
	@go run ./cmd/mockvllm -addr :8000 & \
	 trap 'kill %1' EXIT; \
	 sleep 1; go run $(PKG) -config config/dev.yaml

test: ## Run the test suite
	go test ./...

test-race: ## Run the test suite under the race detector
	go test -race ./...

lint: ## Vet and check formatting
	go vet ./...
	@test -z "$$(gofmt -l . | tee /dev/stderr)" || (echo "run: make fmt"; exit 1)

fmt: ## Format all Go source
	gofmt -w .

models: ## Print the OpenRouter model document the gateway would publish
	@curl -s localhost:8080/v1/models | python3 -m json.tool

clean:
	rm -rf dist *.db *.db-wal *.db-shm
