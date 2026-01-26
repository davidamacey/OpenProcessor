# ==================================================================================
# Makefile for Triton Inference Server YOLO Deployment
# ==================================================================================
# This Makefile provides convenient shortcuts for common development tasks
# with unified service management.
# ==================================================================================

# Default shell
SHELL := /bin/bash

# Variables
COMPOSE := docker compose
API_SERVICE := yolo-api
TRITON_SERVICE := triton-api
OPENSEARCH_SERVICE := opensearch
BENCHMARK_DIR := benchmarks
SCRIPTS_DIR := scripts

# Port configurations
API_PORT := 4603
TRITON_HTTP_PORT := 4600
TRITON_GRPC_PORT := 4601
TRITON_METRICS_PORT := 4602
PROMETHEUS_PORT := 4604
GRAFANA_PORT := 4605
LOKI_PORT := 4606
OPENSEARCH_PORT := 4607
OPENSEARCH_DASH_PORT := 4608

# Default target
.DEFAULT_GOAL := help

# ==================================================================================
# Help
# ==================================================================================

.PHONY: help
help: ## Show this help message
	@echo "==================================================================================="
	@echo "Triton Inference Server - YOLO Deployment Makefile"
	@echo "==================================================================================="
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick Start:"
	@echo "  make up          # Start all services"
	@echo "  make status      # Check service status"
	@echo "  make bench-quick # Run quick benchmark"
	@echo "  make logs        # View all logs"
	@echo ""

# ==================================================================================
# Service Management
# ==================================================================================

.PHONY: up
up: ## Start all services (Triton + API + Monitoring + OpenSearch)
	@echo "Starting all services..."
	$(COMPOSE) up -d
	@echo ""
	@echo "Services starting. Check status with: make status"
	@echo "API available at: http://localhost:$(API_PORT)"
	@echo "Grafana dashboard: http://localhost:$(GRAFANA_PORT) (admin/admin)"

.PHONY: down
down: ## Stop all services
	@echo "Stopping all services..."
	$(COMPOSE) down

.PHONY: restart
restart: ## Restart all services
	@echo "Restarting all services..."
	$(COMPOSE) restart
	@echo "Services restarted. Check status with: make status"

.PHONY: restart-triton
restart-triton: ## Restart only Triton server (after model changes)
	@echo "Restarting Triton server..."
	$(COMPOSE) restart $(TRITON_SERVICE)
	@sleep 5
	@echo "Triton restarted. Checking model status..."
	@$(MAKE) status

.PHONY: restart-api
restart-api: ## Restart only API service
	@echo "Restarting API service..."
	$(COMPOSE) restart $(API_SERVICE)

.PHONY: build
build: ## Build all containers
	@echo "Building containers..."
	$(COMPOSE) build

.PHONY: rebuild
rebuild: ## Rebuild containers without cache
	@echo "Rebuilding containers (no cache)..."
	$(COMPOSE) build --no-cache

# ==================================================================================
# Logs and Monitoring
# ==================================================================================

.PHONY: logs
logs: ## Follow logs from all services
	$(COMPOSE) logs -f

.PHONY: logs-triton
logs-triton: ## Follow Triton server logs
	$(COMPOSE) logs -f $(TRITON_SERVICE)

.PHONY: logs-api
logs-api: ## Follow API service logs
	$(COMPOSE) logs -f $(API_SERVICE)

.PHONY: logs-opensearch
logs-opensearch: ## Follow OpenSearch logs
	$(COMPOSE) logs -f $(OPENSEARCH_SERVICE)

.PHONY: status
status: ## Check health of all services
	@bash $(SCRIPTS_DIR)/check_services.sh

.PHONY: health
health: status ## Alias for status

.PHONY: ps
ps: ## Show running containers
	$(COMPOSE) ps

# ==================================================================================
# Testing
# ==================================================================================

.PHONY: test-detect
test-detect: ## Test object detection endpoint
	curl -X POST http://localhost:$(API_PORT)/detect -F "image=@test_images/sample.jpg" | jq

.PHONY: test-faces
test-faces: ## Test face detection/recognition
	curl -X POST http://localhost:$(API_PORT)/faces/recognize -F "image=@test_images/faces/sample.jpg" | jq

.PHONY: test-embed
test-embed: ## Test embedding endpoints
	curl -X POST http://localhost:$(API_PORT)/embed/image -F "image=@test_images/sample.jpg" | jq

.PHONY: test-ocr
test-ocr: ## Test OCR endpoint
	curl -X POST http://localhost:$(API_PORT)/ocr/predict -F "image=@test_images/ocr_sample.jpg" | jq

.PHONY: test-analyze
test-analyze: ## Test combined analysis
	curl -X POST http://localhost:$(API_PORT)/analyze -F "image=@test_images/sample.jpg" | jq

.PHONY: test-search
test-search: ## Test image search
	curl -X POST http://localhost:$(API_PORT)/search/image -F "image=@test_images/sample.jpg" | jq

.PHONY: test-ingest
test-ingest: ## Test image ingestion
	curl -X POST http://localhost:$(API_PORT)/ingest -F "file=@test_images/sample.jpg" | jq

.PHONY: test-all
test-all: ## Run all endpoint tests
	@echo "Testing all endpoints..."
	$(MAKE) test-detect
	$(MAKE) test-faces
	$(MAKE) test-embed
	$(MAKE) test-ocr
	$(MAKE) test-analyze

.PHONY: test-api-health
test-api-health: ## Test API health
	@echo "Testing API health (port $(API_PORT))..."
	@curl -sf http://localhost:$(API_PORT)/health && echo " OK" || echo " FAILED"

.PHONY: test-inference
test-inference: ## Test inference on all tracks (shell script)
	@echo "Testing inference on all tracks..."
	@bash tests/test_inference.sh

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "Running integration tests..."
	$(COMPOSE) exec $(API_SERVICE) python /app/scripts/test_integration.py

.PHONY: test-patch
test-patch: ## Verify End2End TRT NMS patch is applied
	@echo "Verifying End2End TensorRT NMS patch..."
	$(COMPOSE) exec $(API_SERVICE) python /app/tests/test_end2end_patch.py

.PHONY: test-onnx
test-onnx: ## Test ONNX End2End model locally (bypasses Triton)
	@echo "Testing ONNX End2End model locally..."
	$(COMPOSE) exec $(API_SERVICE) python /app/tests/test_onnx_end2end.py

.PHONY: test-shared-client
test-shared-client: ## Test shared vs per-request client performance
	@echo "Testing shared vs per-request client..."
	@bash tests/test_shared_vs_per_request.sh

# ==================================================================================
# Benchmarking
# ==================================================================================

# Benchmark configuration
BENCH_DURATION := 30s
BENCH_CLIENTS := 32
BENCH_REQUESTS := 1000
TEST_IMAGE := test_images/sample.jpg

.PHONY: bench-detect
bench-detect: ## Benchmark detection endpoint with wrk
	@echo "Benchmarking /detect ($(BENCH_DURATION), $(BENCH_CLIENTS) clients)..."
	@which wrk > /dev/null 2>&1 || (echo "Install wrk: apt install wrk"; exit 1)
	@echo "POST http://localhost:$(API_PORT)/detect" > /tmp/bench_detect.lua
	@echo 'wrk.method = "POST"' >> /tmp/bench_detect.lua
	@echo 'wrk.body = ""' >> /tmp/bench_detect.lua
	@echo 'wrk.headers["Content-Type"] = "multipart/form-data"' >> /tmp/bench_detect.lua
	wrk -t4 -c$(BENCH_CLIENTS) -d$(BENCH_DURATION) -s $(BENCHMARK_DIR)/scripts/detect.lua http://localhost:$(API_PORT)/detect

.PHONY: bench-faces
bench-faces: ## Benchmark face recognition with ab
	@echo "Benchmarking /faces/recognize ($(BENCH_REQUESTS) requests, $(BENCH_CLIENTS) concurrent)..."
	@which ab > /dev/null 2>&1 || (echo "Install ab: apt install apache2-utils"; exit 1)
	ab -n $(BENCH_REQUESTS) -c $(BENCH_CLIENTS) -p $(TEST_IMAGE) -T "image/jpeg" \
		http://localhost:$(API_PORT)/faces/recognize

.PHONY: bench-embed
bench-embed: ## Benchmark image embedding with ab
	@echo "Benchmarking /embed/image ($(BENCH_REQUESTS) requests, $(BENCH_CLIENTS) concurrent)..."
	ab -n $(BENCH_REQUESTS) -c $(BENCH_CLIENTS) -p $(TEST_IMAGE) -T "image/jpeg" \
		http://localhost:$(API_PORT)/embed/image

.PHONY: bench-ingest
bench-ingest: ## Benchmark image ingestion with ab
	@echo "Benchmarking /ingest ($(BENCH_REQUESTS) requests, $(BENCH_CLIENTS) concurrent)..."
	ab -n $(BENCH_REQUESTS) -c $(BENCH_CLIENTS) -p $(TEST_IMAGE) -T "image/jpeg" \
		http://localhost:$(API_PORT)/ingest

.PHONY: bench-search
bench-search: ## Benchmark image search with ab
	@echo "Benchmarking /search/image ($(BENCH_REQUESTS) requests, $(BENCH_CLIENTS) concurrent)..."
	ab -n $(BENCH_REQUESTS) -c $(BENCH_CLIENTS) -p $(TEST_IMAGE) -T "image/jpeg" \
		http://localhost:$(API_PORT)/search/image

.PHONY: bench-quick
bench-quick: ## Quick benchmark of all endpoints (1000 requests each)
	@echo "==================================================================================="
	@echo "Quick Benchmark - All Endpoints"
	@echo "==================================================================================="
	@echo ""
	@echo "--- /detect ---"
	@curl -s -w "Time: %{time_total}s\n" -o /dev/null -X POST http://localhost:$(API_PORT)/detect -F "image=@$(TEST_IMAGE)"
	@echo ""
	@echo "--- /faces/recognize ---"
	@curl -s -w "Time: %{time_total}s\n" -o /dev/null -X POST http://localhost:$(API_PORT)/faces/recognize -F "image=@$(TEST_IMAGE)"
	@echo ""
	@echo "--- /embed/image ---"
	@curl -s -w "Time: %{time_total}s\n" -o /dev/null -X POST http://localhost:$(API_PORT)/embed/image -F "image=@$(TEST_IMAGE)"
	@echo ""
	@echo "--- /ocr/predict ---"
	@curl -s -w "Time: %{time_total}s\n" -o /dev/null -X POST http://localhost:$(API_PORT)/ocr/predict -F "image=@$(TEST_IMAGE)"
	@echo ""
	@echo "Quick benchmark complete."

.PHONY: bench-results
bench-results: ## Show recent benchmark results
	@echo "Recent benchmark results:"
	@ls -lt $(BENCHMARK_DIR)/results/ 2>/dev/null | head -n 10 || echo "No results yet"

.PHONY: bench-python
bench-python: ## Run Python benchmark script with httpx
	@echo "Running Python benchmark..."
	$(COMPOSE) exec $(API_SERVICE) python /app/benchmarks/scripts/benchmark.py

# ==================================================================================
# Model Management
# ==================================================================================

.PHONY: models-list
models-list: ## List loaded Triton models
	curl -s http://localhost:$(API_PORT)/models | jq

.PHONY: models-status
models-status: ## Check model health
	curl -s http://localhost:$(API_PORT)/health | jq

.PHONY: models-reload
models-reload: ## Reload all models
	docker compose exec triton-api tritonserver --model-control-mode=explicit --load-model=*

# ==================================================================================
# Development and Testing
# ==================================================================================

.PHONY: shell-api
shell-api: ## Open shell in API container
	$(COMPOSE) exec $(API_SERVICE) /bin/bash

.PHONY: shell-triton
shell-triton: ## Open shell in Triton container
	$(COMPOSE) exec $(TRITON_SERVICE) /bin/bash

.PHONY: shell-opensearch
shell-opensearch: ## Open shell in OpenSearch container
	$(COMPOSE) exec $(OPENSEARCH_SERVICE) /bin/bash

.PHONY: profile-api
profile-api: ## Profile API with py-spy (DURATION=30, OUTPUT=profile.svg)
	@echo "======================================"
	@echo "FastAPI Performance Profiling"
	@echo "======================================"
	@DURATION=$${DURATION:-30}; OUTPUT=$${OUTPUT:-profile.svg}; \
	echo "Duration: $${DURATION} seconds"; \
	echo "Output: $${OUTPUT}"; \
	CONTAINER_PID=$$($(COMPOSE) exec $(API_SERVICE) pgrep -f "uvicorn src.main:app" | head -1 | tr -d '[:space:]'); \
	if [ -z "$$CONTAINER_PID" ]; then \
		echo "ERROR: Could not find uvicorn process"; \
		exit 1; \
	fi; \
	echo "Found process: PID $$CONTAINER_PID"; \
	FORMAT="flamegraph"; \
	case "$$OUTPUT" in *.speedscope) FORMAT="speedscope";; esac; \
	echo "Generating $$FORMAT visualization..."; \
	$(COMPOSE) exec $(API_SERVICE) py-spy record --pid $$CONTAINER_PID --duration $$DURATION --rate 100 --format $$FORMAT --output /tmp/$$OUTPUT --subprocesses; \
	docker cp $$(docker compose ps -q $(API_SERVICE)):/tmp/$$OUTPUT ./$$OUTPUT; \
	echo "Profile saved to: $$OUTPUT"

.PHONY: resize-images
resize-images: ## Resize images for testing (SOURCE_DIR, OUTPUT_DIR, SIZE)
	@echo "Resizing images..."
	@. .venv/bin/activate && python scripts/resize_images.py \
		--source $${SOURCE_DIR:-test_images} \
		--output $${OUTPUT_DIR:-test_images_resized} \
		--size $${SIZE:-640}

.PHONY: test-create-images
test-create-images: ## Generate test images in various sizes (SOURCE required)
	@echo "Creating test images..."
	@if [ -z "$(SOURCE)" ]; then \
		echo "Error: SOURCE parameter required"; \
		echo "Example: make test-create-images SOURCE=/path/to/image.jpg"; \
		exit 1; \
	fi
	python tests/create_test_images.py --source "$(SOURCE)"

# ==================================================================================
# Model Management API (Dynamic Upload & Export)
# ==================================================================================

.PHONY: api-upload-model
api-upload-model: ## Upload a model via API (usage: make api-upload-model MODEL=/path/to/model.pt [NAME=custom_name])
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make api-upload-model MODEL=/path/to/model.pt [NAME=custom_name]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make api-upload-model MODEL=./my_model.pt"; \
		echo "  make api-upload-model MODEL=./my_model.pt NAME=vehicle_detector"; \
		exit 1; \
	fi
	@NAME_ARG=""; \
	if [ -n "$(NAME)" ]; then NAME_ARG="-F triton_name=$(NAME)"; fi; \
	echo "Uploading model $(MODEL) via API..."; \
	curl -s -X POST http://localhost:$(API_PORT)/models/upload \
		-F "file=@$(MODEL)" \
		$$NAME_ARG | jq '.'

.PHONY: api-export-status
api-export-status: ## Check export task status (usage: make api-export-status ID=task_id)
	@if [ -z "$(ID)" ]; then \
		echo "Error: ID parameter required"; \
		echo "Usage: make api-export-status ID=task_id"; \
		exit 1; \
	fi
	@curl -s http://localhost:$(API_PORT)/models/export/$(ID) | jq '.'

.PHONY: api-exports
api-exports: ## List all export tasks
	@echo "Export tasks:"
	@curl -s http://localhost:$(API_PORT)/models/exports | jq '.'

.PHONY: api-models
api-models: ## List all models in Triton repository
	@echo "Models in Triton repository:"
	@curl -s http://localhost:$(API_PORT)/models/ | jq '.'

.PHONY: api-load-model
api-load-model: ## Load a model into Triton (usage: make api-load-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-load-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Loading model $(NAME) into Triton..."
	@curl -s -X POST http://localhost:$(API_PORT)/models/$(NAME)/load | jq '.'

.PHONY: api-unload-model
api-unload-model: ## Unload a model from Triton (usage: make api-unload-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-unload-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Unloading model $(NAME) from Triton..."
	@curl -s -X POST http://localhost:$(API_PORT)/models/$(NAME)/unload | jq '.'

.PHONY: api-delete-model
api-delete-model: ## Delete a model from repository (usage: make api-delete-model NAME=model_name)
	@if [ -z "$(NAME)" ]; then \
		echo "Error: NAME parameter required"; \
		echo "Usage: make api-delete-model NAME=model_name"; \
		exit 1; \
	fi
	@echo "Deleting model $(NAME)..."
	@curl -s -X DELETE http://localhost:$(API_PORT)/models/$(NAME) | jq '.'

.PHONY: api-health
api-health: ## Check if API is healthy and ready
	@echo "Checking API health..."
	@curl -sf http://localhost:$(API_PORT)/health > /dev/null && echo "API is healthy" || (echo "API not ready"; exit 1)

.PHONY: api-wait-ready
api-wait-ready: ## Wait for API to be ready (up to 60 seconds)
	@echo "Waiting for API to be ready..."
	@for i in $$(seq 1 12); do \
		if curl -sf http://localhost:$(API_PORT)/health > /dev/null 2>&1; then \
			echo "API is ready!"; \
			exit 0; \
		fi; \
		echo "  Attempt $$i/12 - waiting 5 seconds..."; \
		sleep 5; \
	done; \
	echo "ERROR: API not ready after 60 seconds"; \
	exit 1

.PHONY: api-test-quick
api-test-quick: ## Quick API test (no export, just endpoint verification)
	@echo "==================================================================================="
	@echo "Model Management API - Quick Endpoint Test"
	@echo "==================================================================================="
	@echo ""
	@echo "--- Testing GET /models/ ---"
	@curl -s http://localhost:$(API_PORT)/models/ | jq '.triton_status, .total'
	@echo ""
	@echo "--- Testing GET /models/exports ---"
	@curl -s http://localhost:$(API_PORT)/models/exports | jq 'length'
	@echo ""
	@echo "--- Testing API Health ---"
	@curl -sf http://localhost:$(API_PORT)/health | jq '.status'
	@echo ""
	@echo "All quick tests passed!"

# ==================================================================================
# Model Export (CLI-based)
# ==================================================================================

.PHONY: export-models
export-models: ## Export YOLO models (TRT + End2End with normalized boxes)
	@echo "Exporting YOLO models to TensorRT formats (normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats trt trt_end2end --normalize-boxes --save-labels --generate-config

.PHONY: export-all
export-all: ## Export all models (nano through xlarge) in all formats
	@echo "Exporting all YOLO models in all formats (normalized boxes)..."
	@echo "WARNING: This will take 60-120 minutes depending on GPU"
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models nano small medium large xlarge --formats all --normalize-boxes --save-labels --generate-config

.PHONY: export-small
export-small: ## Quick export for small model (TRT + End2End with normalized boxes)
	@echo "Exporting small model (TRT + End2End, normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats trt trt_end2end --normalize-boxes --save-labels --generate-config

.PHONY: export-onnx
export-onnx: ## Export ONNX-only format (with normalized boxes)
	@echo "Exporting ONNX models only (normalized boxes)..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --models small --formats onnx onnx_end2end --normalize-boxes --save-labels

.PHONY: export-custom
export-custom: ## Export custom model (usage: make export-custom MODEL=/path/to/model.pt [NAME=custom_name] [BATCH=32])
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make export-custom MODEL=/path/to/model.pt [NAME=custom_name] [BATCH=32]"; \
		echo ""; \
		echo "Examples:"; \
		echo "  make export-custom MODEL=/app/pytorch_models/my_model.pt"; \
		echo "  make export-custom MODEL=/app/pytorch_models/my_model.pt NAME=my_detector BATCH=64"; \
		exit 1; \
	fi
	@CUSTOM_ARG="$(MODEL)"; \
	if [ -n "$(NAME)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:$(NAME)"; elif [ -n "$(BATCH)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:"; fi; \
	if [ -n "$(BATCH)" ]; then CUSTOM_ARG="$$CUSTOM_ARG:$(BATCH)"; fi; \
	echo "Exporting custom model: $$CUSTOM_ARG"; \
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--custom-model "$$CUSTOM_ARG" \
		--formats trt trt_end2end \
		--normalize-boxes \
		--save-labels \
		--generate-config

.PHONY: export-config
export-config: ## Export models from YAML config file (usage: make export-config CONFIG=/path/to/config.yaml)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG parameter required"; \
		echo "Usage: make export-config CONFIG=/path/to/config.yaml"; \
		echo ""; \
		echo "Example YAML format:"; \
		echo "  models:"; \
		echo "    my_model:"; \
		echo "      pt_file: /app/pytorch_models/my_model.pt"; \
		echo "      triton_name: my_custom_detector  # optional"; \
		echo "      max_batch: 32                    # optional"; \
		exit 1; \
	fi
	@echo "Exporting models from config: $(CONFIG)"
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py \
		--config-file "$(CONFIG)" \
		--formats trt trt_end2end \
		--normalize-boxes \
		--save-labels \
		--generate-config

.PHONY: export-list
export-list: ## List available built-in models
	@$(COMPOSE) exec $(API_SERVICE) python /app/export/export_models.py --list-models

.PHONY: export-mobileclip
export-mobileclip: ## Export MobileCLIP models
	@echo "Exporting MobileCLIP image encoder..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_mobileclip_image_encoder.py
	@echo "Exporting MobileCLIP text encoder..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_mobileclip_text_encoder.py
	@$(MAKE) restart-triton

.PHONY: export-status
export-status: ## Show status of all exported models
	@echo "==================================================================================="
	@echo "Model Export Status"
	@echo "==================================================================================="
	@echo ""
	@echo "PyTorch Models (pytorch_models/):"
	@ls -lh pytorch_models/*.pt 2>/dev/null || echo "  No PyTorch models found"
	@echo ""
	@echo "Triton Models (models/):"
	@for dir in models/yolov11*; do \
		if [ -d "$$dir" ]; then \
			name=$$(basename $$dir); \
			model=""; config=""; \
			[ -f "$$dir/1/model.onnx" ] && model="ONNX"; \
			[ -f "$$dir/1/model.plan" ] && model="TRT"; \
			[ -f "$$dir/config.pbtxt" ] && config="OK" || config="MISSING"; \
			printf "  %-35s model: %-5s config: %s\n" "$$name" "$${model:-NONE}" "$$config"; \
		fi \
	done
	@echo ""

.PHONY: validate-exports
validate-exports: ## Validate that Triton can load exported models
	@echo "Validating exported models with Triton..."
	@echo "Checking Triton model repository status..."
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models | jq -r '.models[]? | "\(.name): \(.state)"' 2>/dev/null || echo "Triton not running. Start with: make up"
	@echo ""
	@echo "Model details:"
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models/yolov11_small_trt/config 2>/dev/null | jq '.name, .max_batch_size' || echo "  yolov11_small_trt: Not loaded"
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models/yolov11_small_trt_end2end/config 2>/dev/null | jq '.name, .max_batch_size' || echo "  yolov11_small_trt_end2end: Not loaded"

# ==================================================================================
# Face Detection & Recognition
# ==================================================================================

.PHONY: download-face-models
download-face-models: ## Download ArcFace face recognition models
	@echo "Downloading face recognition models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/download_face_models.py

.PHONY: export-face-recognition
export-face-recognition: ## Export ArcFace to TensorRT
	@echo "Exporting ArcFace face recognition to TensorRT..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_face_recognition.py
	@$(MAKE) load-face-models

.PHONY: load-face-models
load-face-models: ## Load face models into Triton
	@echo "Loading face models into Triton..."
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/yolo11_face_small_trt_end2end/load" || true
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/arcface_w600k_r50/load" || true
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/yolo11_face_pipeline/load" || true
	@echo "Face models loaded."

.PHONY: setup-face-pipeline
setup-face-pipeline: download-face-models export-face-recognition download-yolo11-face export-yolo11-face ## Complete face pipeline setup
	@echo "Face pipeline setup complete!"
	@echo ""
	@echo "Loaded models:"
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/index" | grep -E "(yolo11_face|arcface)" || true

.PHONY: download-face-test-data
download-face-test-data: ## Download LFW and WIDER Face test datasets
	@echo "Downloading face test datasets..."
	@bash $(SCRIPTS_DIR)/setup_face_test_data.sh --all
	@echo ""
	@echo "Datasets downloaded to test_images/faces/"

# ==================================================================================
# YOLO11-Face (Alternative Face Detection Pipeline)
# ==================================================================================

.PHONY: download-yolo11-face
download-yolo11-face: ## Download YOLO11-face models from YapaLab/yolo-face
	@echo "Downloading YOLO11-face models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/scripts/download_yolo11_face.py --models small

.PHONY: export-yolo11-face
export-yolo11-face: ## Export YOLO11-face to TensorRT
	@echo "Exporting YOLO11-face to TensorRT..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_yolo11_face.py \
		--model /app/pytorch_models/yolo11_face/yolov11s-face.pt

.PHONY: load-yolo11-face
load-yolo11-face: ## Load YOLO11-face models into Triton
	@echo "Loading YOLO11-face models into Triton..."
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/yolo11_face_small_trt/load" || true
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/yolo11_face_pipeline/load" || true
	@echo "YOLO11-face models loaded."

.PHONY: setup-yolo11-face
setup-yolo11-face: download-yolo11-face export-yolo11-face restart-triton load-yolo11-face ## Complete YOLO11-face setup
	@echo "YOLO11-face setup complete!"
	@echo ""
	@echo "Test with:"
	@echo "  curl -X POST http://localhost:$(API_PORT)/faces/detect -F 'file=@test.jpg'"

# ==================================================================================
# OCR (PP-OCRv5)
# ==================================================================================

.PHONY: download-paddleocr
download-paddleocr: ## Download PP-OCRv5 ONNX models and dictionaries
	@echo "Downloading PP-OCRv5 models..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/download_paddleocr.py

.PHONY: export-paddleocr-det
export-paddleocr-det: ## Export PaddleOCR detection model to TensorRT
	@echo "Exporting PP-OCRv5 detection to TensorRT..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_paddleocr_det.py

.PHONY: export-paddleocr-rec
export-paddleocr-rec: ## Export PaddleOCR recognition model to TensorRT
	@echo "Exporting PP-OCRv5 recognition to TensorRT..."
	$(COMPOSE) exec $(API_SERVICE) python /app/export/export_paddleocr_rec.py

.PHONY: export-paddleocr
export-paddleocr: export-paddleocr-det export-paddleocr-rec ## Export both PaddleOCR models to TensorRT
	@echo "Both OCR models exported!"

.PHONY: setup-ocr
setup-ocr: download-paddleocr export-paddleocr restart-triton ## Complete OCR pipeline setup
	@echo "OCR pipeline setup complete!"

.PHONY: load-ocr-models
load-ocr-models: ## Load OCR models into Triton
	@echo "Loading OCR models into Triton..."
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/paddleocr_det_trt/load" || true
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/paddleocr_rec_trt/load" || true
	@echo "OCR models loaded."

# ==================================================================================
# Monitoring and Metrics
# ==================================================================================

.PHONY: open-grafana
open-grafana: ## Open Grafana dashboard in browser
	@echo "Opening Grafana dashboard..."
	@echo "URL: http://localhost:$(GRAFANA_PORT)"
	@echo "Login: admin/admin"
	@xdg-open http://localhost:$(GRAFANA_PORT) 2>/dev/null || open http://localhost:$(GRAFANA_PORT) 2>/dev/null || echo "Please open http://localhost:$(GRAFANA_PORT) in your browser"

.PHONY: open-prometheus
open-prometheus: ## Open Prometheus UI in browser
	@echo "Opening Prometheus..."
	@xdg-open http://localhost:$(PROMETHEUS_PORT) 2>/dev/null || open http://localhost:$(PROMETHEUS_PORT) 2>/dev/null || echo "Please open http://localhost:$(PROMETHEUS_PORT) in your browser"

.PHONY: open-opensearch
open-opensearch: ## Open OpenSearch Dashboards in browser
	@echo "Opening OpenSearch Dashboards..."
	@xdg-open http://localhost:$(OPENSEARCH_DASH_PORT) 2>/dev/null || open http://localhost:$(OPENSEARCH_DASH_PORT) 2>/dev/null || echo "Please open http://localhost:$(OPENSEARCH_DASH_PORT) in your browser"

.PHONY: metrics
metrics: ## Show Triton metrics
	@curl -s http://localhost:$(TRITON_METRICS_PORT)/metrics | grep -E "nv_inference_|nv_gpu_" | head -n 20

.PHONY: gpu
gpu: ## Show GPU status
	@nvidia-smi

.PHONY: gpu-watch
gpu-watch: ## Watch GPU status (updates every second)
	@watch -n 1 nvidia-smi

# ==================================================================================
# Triton Model Management
# ==================================================================================

.PHONY: triton-health
triton-health: ## Check Triton server health
	@echo "Checking Triton health..."
	@curl -sf http://localhost:$(TRITON_HTTP_PORT)/v2/health/ready > /dev/null && echo "Triton is ready" || (echo "Triton not ready"; exit 1)

.PHONY: triton-models-ready
triton-models-ready: ## List all READY models in Triton
	@echo "=== Triton Models (READY) ==="
	@curl -s -X POST http://localhost:$(TRITON_HTTP_PORT)/v2/repository/index 2>/dev/null | \
		python3 -c "import sys,json; models=json.load(sys.stdin); ready=[m['name'] for m in models if m.get('state')=='READY']; print(f'Total: {len(ready)} models'); [print(f'  - {n}') for n in sorted(ready)]" 2>/dev/null || echo "Error: Cannot connect to Triton"

.PHONY: triton-stats
triton-stats: ## Show Triton model statistics
	@echo "=== Triton Model Statistics ==="
	@curl -s http://localhost:$(TRITON_HTTP_PORT)/v2/models/stats 2>/dev/null | \
		python3 -c "import sys,json; stats=json.load(sys.stdin); ms=stats.get('model_stats',[]); [print(f\"{m['name']}: {m['inference_stats']['success']['count']} inferences\") for m in ms if m['inference_stats']['success']['count']>0]" 2>/dev/null || echo "Error: Cannot fetch stats"

.PHONY: triton-metrics
triton-metrics: ## Show key Triton metrics (inference counts, latencies)
	@echo "=== Triton Key Metrics ==="
	@curl -s http://localhost:$(TRITON_METRICS_PORT)/metrics 2>/dev/null | \
		grep -E "nv_inference_count|nv_inference_compute_infer_duration" | \
		grep -v "^#" | head -30

.PHONY: triton-unload-all
triton-unload-all: ## Unload all models from Triton server
	@echo "Unloading all models from Triton..."
	@for model in $$(curl -s -X POST http://localhost:$(TRITON_HTTP_PORT)/v2/repository/index | jq -r '.[].name'); do \
		echo "  Unloading $$model..."; \
		curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/$$model/unload" > /dev/null; \
	done
	@sleep 3
	@echo ""
	@echo "READY models remaining:"
	@curl -s -X POST http://localhost:$(TRITON_HTTP_PORT)/v2/repository/index | jq -r '.[] | select(.state == "READY") | .name'

.PHONY: triton-models
triton-models: ## List all Triton models and their state
	@echo "Triton Model Repository:"
	@curl -s -X POST http://localhost:$(TRITON_HTTP_PORT)/v2/repository/index | jq -r '.[] | "\(.name): \(.state)"'

.PHONY: triton-load
triton-load: ## Load a model into Triton (usage: make triton-load MODEL=model_name)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make triton-load MODEL=model_name"; \
		exit 1; \
	fi
	@echo "Loading $(MODEL) into Triton..."
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/$(MODEL)/load" | jq '.'

.PHONY: triton-unload
triton-unload: ## Unload a model from Triton (usage: make triton-unload MODEL=model_name)
	@if [ -z "$(MODEL)" ]; then \
		echo "Error: MODEL parameter required"; \
		echo "Usage: make triton-unload MODEL=model_name"; \
		exit 1; \
	fi
	@echo "Unloading $(MODEL) from Triton..."
	@curl -s -X POST "http://localhost:$(TRITON_HTTP_PORT)/v2/repository/models/$(MODEL)/unload" | jq '.'

# ==================================================================================
# Health & Status Checks
# ==================================================================================

.PHONY: check-all
check-all: ## Full system health check (API + Triton + OpenSearch)
	@echo "==================================================================================="
	@echo "System Health Check"
	@echo "==================================================================================="
	@echo ""
	@echo "--- API Service ---"
	@curl -sf http://localhost:$(API_PORT)/health > /dev/null && echo "API is healthy" || echo "API not responding"
	@echo ""
	@echo "--- Triton Server ---"
	@curl -sf http://localhost:$(TRITON_HTTP_PORT)/v2/health/ready > /dev/null && echo "Triton is ready" || echo "Triton not ready"
	@echo ""
	@echo "--- OpenSearch ---"
	@curl -sf http://localhost:$(OPENSEARCH_PORT)/_cluster/health > /dev/null && echo "OpenSearch is healthy" || echo "OpenSearch not responding"
	@echo ""
	@echo "--- GPU Status ---"
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "Cannot query GPU"
	@echo ""
	@echo "==================================================================================="

# ==================================================================================
# Cleanup
# ==================================================================================

.PHONY: clean
clean: ## Stop services and remove containers
	@echo "Stopping and removing containers..."
	$(COMPOSE) down

.PHONY: clean-all
clean-all: ## Stop services, remove containers and volumes (WARNING: deletes all data)
	@echo "WARNING: This will remove all containers, volumes, and data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	$(COMPOSE) down -v

.PHONY: clean-logs
clean-logs: ## Clear Docker logs
	@echo "Clearing Docker logs..."
	$(COMPOSE) down
	@docker system prune -f

.PHONY: clean-bench
clean-bench: ## Remove benchmark results
	@echo "Removing benchmark results..."
	@rm -rf $(BENCHMARK_DIR)/results/*
	@echo "Benchmark results cleared."

.PHONY: clean-exports
clean-exports: ## Clean old model exports (keeps configs, prepares for re-export)
	@echo "==================================================================================="
	@echo "Cleaning old model exports..."
	@echo "==================================================================================="
	@echo ""
	@echo "Backing up config.pbtxt files..."
	@mkdir -p models/backup_configs_$$(date +%Y%m%d_%H%M%S)
	@for config in models/*/config.pbtxt; do \
		if [ -f "$$config" ]; then \
			model_name=$$(dirname "$$config" | xargs basename); \
			cp "$$config" "models/backup_configs_$$(date +%Y%m%d_%H%M%S)/$${model_name}_config.pbtxt" 2>/dev/null || true; \
		fi; \
	done
	@echo ""
	@echo "Removing old ONNX and TRT files..."
	@for dir in models/yolov11_*/1; do \
		if [ -d "$$dir" ]; then \
			rm -f "$$dir/model.onnx" "$$dir/model.onnx.old" "$$dir/model.plan" "$$dir/model.plan.old" 2>/dev/null || true; \
		fi; \
	done
	@echo ""
	@echo "Clearing TRT cache..."
	@rm -rf trt_cache/* 2>/dev/null || true
	@echo ""
	@echo "Done! Run 'make export-status' to see current state."
	@echo "Then run 'make export-models' or 'make export-all' to re-export."

# ==================================================================================
# OpenSearch / Data Management
# ==================================================================================

.PHONY: opensearch-reset
opensearch-reset: ## Reset OpenSearch indices (WARNING: deletes all visual search data)
	@echo "WARNING: This will delete all OpenSearch indices and data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read
	@curl -X DELETE "http://localhost:$(OPENSEARCH_PORT)/_all" || echo "Failed to delete indices"
	@echo "OpenSearch indices cleared."

.PHONY: opensearch-status
opensearch-status: ## Show OpenSearch cluster status
	@echo "OpenSearch Cluster Status:"
	@curl -s http://localhost:$(OPENSEARCH_PORT)/_cluster/health?pretty

.PHONY: opensearch-indices
opensearch-indices: ## List OpenSearch indices
	@echo "OpenSearch Indices:"
	@curl -s http://localhost:$(OPENSEARCH_PORT)/_cat/indices?v

.PHONY: opensearch-reset-indexes
opensearch-reset-indexes: ## Reset all OpenSearch indexes (delete and recreate)
	@echo "Resetting OpenSearch indexes..."
	@curl -s -X DELETE "http://localhost:$(API_PORT)/index" | python3 -c "import sys,json; print(json.load(sys.stdin).get('message','deleted'))" 2>/dev/null || true
	@sleep 1
	@curl -s -X POST "http://localhost:$(API_PORT)/index/create" | python3 -c "import sys,json; print('Indexes created:', json.load(sys.stdin).get('status','unknown'))" 2>/dev/null
	@echo "Done."

# ==================================================================================
# Documentation
# ==================================================================================

.PHONY: info
info: ## Show service URLs and ports
	@echo "==================================================================================="
	@echo "Triton Inference Server - Unified Deployment"
	@echo "==================================================================================="
	@echo ""
	@echo "Quick Start:"
	@echo "  make up                    Start all services"
	@echo "  make status                Check service health"
	@echo "  make test-all              Test all endpoints"
	@echo ""
	@echo "Services:"
	@echo "  API:                       http://localhost:$(API_PORT)"
	@echo "  Triton HTTP:               http://localhost:$(TRITON_HTTP_PORT)"
	@echo "  Triton gRPC:               http://localhost:$(TRITON_GRPC_PORT)"
	@echo "  OpenSearch:                http://localhost:$(OPENSEARCH_PORT)"
	@echo ""
	@echo "Monitoring:"
	@echo "  Grafana:                   http://localhost:$(GRAFANA_PORT) (admin/admin)"
	@echo "  Prometheus:                http://localhost:$(PROMETHEUS_PORT)"
	@echo ""
	@echo "API Endpoints (port $(API_PORT)):"
	@echo "  Detection:                 POST /detect"
	@echo "  Face Recognition:          POST /faces/recognize"
	@echo "  Image Embedding:           POST /embed/image"
	@echo "  Text Embedding:            POST /embed/text"
	@echo "  OCR:                       POST /ocr/predict"
	@echo "  Image Search:              POST /search/image"
	@echo "  Text Search:               POST /search/text"
	@echo "  Ingest:                    POST /ingest"
	@echo "  Analyze:                   POST /analyze"
	@echo ""

.PHONY: docs
docs: info ## Alias for info

# ==================================================================================
# Reference Repositories (for attribution and development)
# ==================================================================================

.PHONY: clone-refs-essential
clone-refs-essential: ## Clone essential reference repos (ultralytics-end2end, ml-mobileclip)
	@echo "Cloning essential reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --essential

.PHONY: clone-refs-recommended
clone-refs-recommended: ## Clone essential + recommended reference repos
	@echo "Cloning recommended reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --recommended

.PHONY: clone-refs-all
clone-refs-all: ## Clone all reference repositories
	@echo "Cloning all reference repositories..."
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --all

.PHONY: clone-refs-list
clone-refs-list: ## List available reference repos and their status
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --list

.PHONY: clone-ref
clone-ref: ## Clone a specific reference repo (usage: make clone-ref REPO=ultralytics-end2end)
	@if [ -z "$(REPO)" ]; then \
		echo "Error: REPO parameter required"; \
		echo "Usage: make clone-ref REPO=repo_name"; \
		echo ""; \
		echo "Available repos:"; \
		bash $(SCRIPTS_DIR)/clone_reference_repos.sh --list; \
		exit 1; \
	fi
	@bash $(SCRIPTS_DIR)/clone_reference_repos.sh --repo $(REPO)

# ==================================================================================
# Phony targets (targets that don't create files)
# ==================================================================================

.PHONY: help up down restart restart-triton restart-api build rebuild \
        logs logs-triton logs-api logs-opensearch status health ps \
        test-detect test-faces test-embed test-ocr test-analyze test-search test-ingest test-all \
        test-api-health test-inference test-integration test-patch test-onnx test-shared-client \
        bench-quick bench-detect bench-faces bench-embed bench-ingest bench-search bench-results bench-python \
        models-list models-status models-reload \
        shell-api shell-triton shell-opensearch profile-api resize-images test-create-images \
        api-upload-model api-export-status api-exports api-models \
        api-load-model api-unload-model api-delete-model \
        api-health api-wait-ready api-test-quick \
        export-models export-all export-small export-onnx export-custom export-config export-list \
        export-mobileclip export-status validate-exports \
        download-face-models export-face-recognition load-face-models setup-face-pipeline download-face-test-data \
        download-yolo11-face export-yolo11-face load-yolo11-face setup-yolo11-face \
        download-paddleocr export-paddleocr-det export-paddleocr-rec export-paddleocr setup-ocr load-ocr-models \
        open-grafana open-prometheus open-opensearch metrics gpu gpu-watch \
        triton-health triton-models-ready triton-stats triton-metrics \
        triton-unload-all triton-models triton-load triton-unload \
        check-all \
        clean clean-all clean-logs clean-bench clean-exports \
        opensearch-reset opensearch-status opensearch-indices opensearch-reset-indexes \
        info docs \
        clone-refs-essential clone-refs-recommended clone-refs-all clone-refs-list clone-ref
