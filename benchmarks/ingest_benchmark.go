// Ingest Benchmark - High-performance parallel image ingestion benchmark
//
// Usage:
//   go run ingest_benchmark.go -dir /path/to/images -workers 64 -clear
//
// This script:
// - Recursively finds all images in a directory
// - Ingests them in parallel using multiple workers
// - Logs detailed real-time metrics
// - Reports comprehensive throughput statistics

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Config holds benchmark configuration
type Config struct {
	Directory        string
	APIHost          string
	APIPort          int
	Workers          int
	Timeout          time.Duration
	ClearIndex       bool
	ReportEvery      int
	LogFile          string
	EnableDetection  bool
	EnableFaces      bool
	EnableClip       bool
	EnableOCR        bool
}

// Stats holds benchmark statistics
type Stats struct {
	TotalImages       int64
	Processed         int64
	Successful        int64
	Failed            int64
	StartTime         time.Time
	Latencies         []float64
	LatenciesMutex    sync.Mutex
	LastReportTime    time.Time
	LastReportCount   int64
	IntervalRates     []float64
	IntervalRatesMu   sync.Mutex
}

// Result holds final benchmark results
type Result struct {
	// Summary
	TotalImages      int64   `json:"total_images"`
	Processed        int64   `json:"processed"`
	Successful       int64   `json:"successful"`
	Failed           int64   `json:"failed"`
	SuccessRate      float64 `json:"success_rate_percent"`

	// Timing
	TotalTimeSeconds float64 `json:"total_time_seconds"`
	TotalTimeHuman   string  `json:"total_time_human"`

	// Throughput
	ImagesPerSecond     float64 `json:"images_per_second"`
	ImagesPerMinute     float64 `json:"images_per_minute"`
	ImagesPerHour       float64 `json:"images_per_hour"`
	PeakImagesPerSecond float64 `json:"peak_images_per_second"`
	MinImagesPerSecond  float64 `json:"min_images_per_second"`
	AvgImagesPerSecond  float64 `json:"avg_images_per_second"`

	// Latency
	LatencyMeanMs float64 `json:"latency_mean_ms"`
	LatencyMinMs  float64 `json:"latency_min_ms"`
	LatencyMaxMs  float64 `json:"latency_max_ms"`
	LatencyP50Ms  float64 `json:"latency_p50_ms"`
	LatencyP90Ms  float64 `json:"latency_p90_ms"`
	LatencyP95Ms  float64 `json:"latency_p95_ms"`
	LatencyP99Ms  float64 `json:"latency_p99_ms"`

	// Configuration
	Workers   int    `json:"workers"`
	Directory string `json:"directory"`
	Timestamp string `json:"timestamp"`
}

// MetricLog for real-time logging
type MetricLog struct {
	Timestamp       string  `json:"timestamp"`
	ElapsedSeconds  float64 `json:"elapsed_seconds"`
	Processed       int64   `json:"processed"`
	Successful      int64   `json:"successful"`
	Failed          int64   `json:"failed"`
	ProgressPercent float64 `json:"progress_percent"`
	CurrentRate     float64 `json:"current_rate_per_sec"`
	OverallRate     float64 `json:"overall_rate_per_sec"`
	ETASeconds      float64 `json:"eta_seconds"`
	ETAHuman        string  `json:"eta_human"`
}

var imageExtensions = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".webp": true,
	".gif": true, ".bmp": true, ".tiff": true, ".tif": true,
}

var logger *log.Logger
var logFile *os.File

func setupLogging(logPath string) {
	var err error
	logFile, err = os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		log.Printf("Warning: Could not create log file: %v", err)
		logger = log.New(os.Stdout, "", 0)
		return
	}
	// Log to both file and stdout
	multiWriter := io.MultiWriter(os.Stdout, logFile)
	logger = log.New(multiWriter, "", 0)
}

func closeLogging() {
	if logFile != nil {
		logFile.Close()
	}
}

func isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return imageExtensions[ext]
}

func findAllImages(dir string) ([]string, error) {
	var images []string
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && isImageFile(path) {
			images = append(images, path)
		}
		return nil
	})
	return images, err
}

func ingestImage(client *http.Client, url string, imagePath string, config *Config) (float64, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return 0, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	part, err := writer.CreateFormFile("image", filepath.Base(imagePath))
	if err != nil {
		return 0, err
	}
	if _, err := io.Copy(part, file); err != nil {
		return 0, err
	}

	writer.WriteField("image_id", imagePath)
	writer.WriteField("image_path", imagePath)
	writer.WriteField("enable_ocr", fmt.Sprintf("%t", config.EnableOCR))
	writer.WriteField("enable_detection", fmt.Sprintf("%t", config.EnableDetection))
	writer.WriteField("enable_faces", fmt.Sprintf("%t", config.EnableFaces))
	writer.WriteField("enable_clip", fmt.Sprintf("%t", config.EnableClip))
	writer.Close()

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	start := time.Now()
	resp, err := client.Do(req)
	latency := float64(time.Since(start).Milliseconds())

	if err != nil {
		return latency, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return latency, fmt.Errorf("status %d: %s", resp.StatusCode, string(bodyBytes[:min(200, len(bodyBytes))]))
	}

	return latency, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func clearIndexes(opensearchPort int, apiHost string, apiPort int) error {
	osURL := fmt.Sprintf("http://localhost:%d", opensearchPort)
	indexes := []string{"visual_search_global", "visual_search_faces", "visual_search_vehicles", "visual_search_people"}
	client := &http.Client{Timeout: 30 * time.Second}

	// Delete existing indexes
	for _, index := range indexes {
		req, _ := http.NewRequest("DELETE", osURL+"/"+index, nil)
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		resp.Body.Close()
	}
	logger.Println("[SETUP] Deleted OpenSearch indexes")

	// Create indexes with proper k-NN mappings
	logger.Println("[SETUP] Creating indexes with k-NN vector mappings...")

	// Global index (CLIP embeddings)
	globalMapping := `{
		"settings": {
			"index": {
				"number_of_shards": 1,
				"number_of_replicas": 0,
				"knn": true,
				"refresh_interval": "-1"
			}
		},
		"mappings": {
			"properties": {
				"image_id": { "type": "keyword" },
				"image_path": { "type": "keyword" },
				"global_embedding": {
					"type": "knn_vector",
					"dimension": 512,
					"method": {
						"name": "hnsw",
						"space_type": "cosinesimil",
						"engine": "faiss",
						"parameters": { "ef_construction": 512, "m": 16 }
					}
				},
				"imohash": { "type": "keyword" },
				"file_size_bytes": { "type": "long" },
				"width": { "type": "integer" },
				"height": { "type": "integer" },
				"duplicate_group_id": { "type": "keyword" },
				"metadata": { "type": "object", "enabled": true },
				"indexed_at": { "type": "date" }
			}
		}
	}`

	// Faces index (ArcFace embeddings)
	facesMapping := `{
		"settings": {
			"index": {
				"number_of_shards": 1,
				"number_of_replicas": 0,
				"knn": true,
				"refresh_interval": "-1"
			}
		},
		"mappings": {
			"properties": {
				"face_id": { "type": "keyword" },
				"image_id": { "type": "keyword" },
				"image_path": { "type": "keyword" },
				"embedding": {
					"type": "knn_vector",
					"dimension": 512,
					"method": {
						"name": "hnsw",
						"space_type": "cosinesimil",
						"engine": "faiss",
						"parameters": { "ef_construction": 512, "m": 16 }
					}
				},
				"box": { "type": "float" },
				"landmarks": {
					"type": "object",
					"properties": {
						"left_eye": { "type": "float" },
						"right_eye": { "type": "float" },
						"nose": { "type": "float" },
						"left_mouth": { "type": "float" },
						"right_mouth": { "type": "float" }
					}
				},
				"confidence": { "type": "float" },
				"quality_score": { "type": "float" },
				"person_id": { "type": "keyword" },
				"person_name": { "type": "keyword" },
				"indexed_at": { "type": "date" }
			}
		}
	}`

	// Create global index
	req, _ := http.NewRequest("PUT", osURL+"/visual_search_global", strings.NewReader(globalMapping))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		logger.Printf("[SETUP] Warning: failed to create global index: %v\n", err)
	} else {
		resp.Body.Close()
		logger.Println("[SETUP] Created visual_search_global with k-NN mapping")
	}

	// Create faces index
	req, _ = http.NewRequest("PUT", osURL+"/visual_search_faces", strings.NewReader(facesMapping))
	req.Header.Set("Content-Type", "application/json")
	resp, err = client.Do(req)
	if err != nil {
		logger.Printf("[SETUP] Warning: failed to create faces index: %v\n", err)
	} else {
		resp.Body.Close()
		logger.Println("[SETUP] Created visual_search_faces with k-NN mapping")
	}

	time.Sleep(1 * time.Second)
	logger.Println("[SETUP] Indexes ready for ingestion")
	return nil
}

func worker(id int, jobs <-chan string, stats *Stats, config *Config, wg *sync.WaitGroup) {
	defer wg.Done()

	client := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        200,
			MaxIdleConnsPerHost: 200,
			IdleConnTimeout:     90 * time.Second,
			DisableKeepAlives:   false,
		},
	}

	url := fmt.Sprintf("http://%s:%d/ingest", config.APIHost, config.APIPort)

	for imagePath := range jobs {
		latency, err := ingestImage(client, url, imagePath, config)

		atomic.AddInt64(&stats.Processed, 1)

		if err != nil {
			atomic.AddInt64(&stats.Failed, 1)
		} else {
			atomic.AddInt64(&stats.Successful, 1)
			stats.LatenciesMutex.Lock()
			stats.Latencies = append(stats.Latencies, latency)
			stats.LatenciesMutex.Unlock()
		}
	}
}

func formatDuration(seconds float64) string {
	if seconds < 60 {
		return fmt.Sprintf("%.0fs", seconds)
	} else if seconds < 3600 {
		return fmt.Sprintf("%.0fm %.0fs", seconds/60, float64(int(seconds)%60))
	} else {
		hours := int(seconds / 3600)
		mins := int(seconds/60) % 60
		secs := int(seconds) % 60
		return fmt.Sprintf("%dh %dm %ds", hours, mins, secs)
	}
}

func progressReporter(stats *Stats, config *Config, done <-chan bool) {
	ticker := time.NewTicker(time.Duration(config.ReportEvery) * time.Second)
	defer ticker.Stop()

	stats.LastReportTime = stats.StartTime
	stats.LastReportCount = 0

	for {
		select {
		case <-ticker.C:
			now := time.Now()
			processed := atomic.LoadInt64(&stats.Processed)
			successful := atomic.LoadInt64(&stats.Successful)
			failed := atomic.LoadInt64(&stats.Failed)

			elapsed := now.Sub(stats.StartTime).Seconds()
			intervalElapsed := now.Sub(stats.LastReportTime).Seconds()
			intervalCount := processed - stats.LastReportCount

			currentRate := float64(intervalCount) / intervalElapsed
			overallRate := float64(processed) / elapsed
			progressPct := float64(processed) / float64(stats.TotalImages) * 100

			remaining := stats.TotalImages - processed
			etaSeconds := 0.0
			if overallRate > 0 {
				etaSeconds = float64(remaining) / overallRate
			}

			// Track interval rates for peak/min calculation
			if currentRate > 0 {
				stats.IntervalRatesMu.Lock()
				stats.IntervalRates = append(stats.IntervalRates, currentRate)
				stats.IntervalRatesMu.Unlock()
			}

			metric := MetricLog{
				Timestamp:       now.Format("15:04:05"),
				ElapsedSeconds:  elapsed,
				Processed:       processed,
				Successful:      successful,
				Failed:          failed,
				ProgressPercent: progressPct,
				CurrentRate:     currentRate,
				OverallRate:     overallRate,
				ETASeconds:      etaSeconds,
				ETAHuman:        formatDuration(etaSeconds),
			}

			// Log detailed metrics
			logger.Printf("[METRIC] %s | Progress: %d/%d (%.1f%%) | Rate: %.1f/s (current) %.1f/s (avg) | Success: %d | Failed: %d | ETA: %s",
				metric.Timestamp,
				processed, stats.TotalImages, progressPct,
				currentRate, overallRate,
				successful, failed,
				metric.ETAHuman,
			)

			stats.LastReportTime = now
			stats.LastReportCount = processed

		case <-done:
			return
		}
	}
}

func calculatePercentile(sorted []float64, percentile float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)) * percentile)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func runBenchmark(config *Config) (*Result, error) {
	logger.Println("╔══════════════════════════════════════════════════════════════╗")
	logger.Println("║            OPENPROCESSOR INGEST BENCHMARK                    ║")
	logger.Println("╚══════════════════════════════════════════════════════════════╝")
	logger.Println()
	logger.Printf("[CONFIG] Directory: %s\n", config.Directory)
	logger.Printf("[CONFIG] Workers: %d\n", config.Workers)
	logger.Printf("[CONFIG] Timeout: %s\n", config.Timeout)
	logger.Printf("[CONFIG] API: http://%s:%d/ingest\n", config.APIHost, config.APIPort)
	logger.Println()
	logger.Println("[PIPELINE] Enabled services:")
	logger.Printf("[PIPELINE]   Detection (YOLO):  %t\n", config.EnableDetection)
	logger.Printf("[PIPELINE]   CLIP embedding:    %t\n", config.EnableClip)
	logger.Printf("[PIPELINE]   Face (SCRFD+Arc):  %t\n", config.EnableFaces)
	logger.Printf("[PIPELINE]   OCR:               %t\n", config.EnableOCR)
	logger.Println()

	// Find all images
	logger.Println("[SCAN] Scanning directory for images...")
	scanStart := time.Now()
	images, err := findAllImages(config.Directory)
	if err != nil {
		return nil, fmt.Errorf("failed to scan directory: %w", err)
	}
	scanTime := time.Since(scanStart).Seconds()

	if len(images) == 0 {
		return nil, fmt.Errorf("no images found in %s", config.Directory)
	}

	logger.Printf("[SCAN] Found %d images in %.2f seconds\n", len(images), scanTime)

	// Clear indexes if requested
	if config.ClearIndex {
		clearIndexes(4607, config.APIHost, config.APIPort)
	}

	// Initialize stats
	stats := &Stats{
		TotalImages:   int64(len(images)),
		StartTime:     time.Now(),
		Latencies:     make([]float64, 0, len(images)),
		IntervalRates: make([]float64, 0, 1000),
	}

	// Create job channel
	jobs := make(chan string, config.Workers*4)

	// Start workers
	var wg sync.WaitGroup
	for i := 0; i < config.Workers; i++ {
		wg.Add(1)
		go worker(i, jobs, stats, config, &wg)
	}

	// Start progress reporter
	done := make(chan bool)
	go progressReporter(stats, config, done)

	// Send jobs
	logger.Println()
	logger.Printf("[START] Beginning ingest of %d images with %d workers...\n", len(images), config.Workers)
	logger.Println()

	for _, img := range images {
		jobs <- img
	}
	close(jobs)

	// Wait for completion
	wg.Wait()
	done <- true

	totalTime := time.Since(stats.StartTime).Seconds()

	// Calculate latency stats
	stats.LatenciesMutex.Lock()
	latencies := make([]float64, len(stats.Latencies))
	copy(latencies, stats.Latencies)
	stats.LatenciesMutex.Unlock()

	var meanLatency, minLatency, maxLatency, p50, p90, p95, p99 float64
	if len(latencies) > 0 {
		sort.Float64s(latencies)

		sum := 0.0
		for _, l := range latencies {
			sum += l
		}
		meanLatency = sum / float64(len(latencies))
		minLatency = latencies[0]
		maxLatency = latencies[len(latencies)-1]
		p50 = calculatePercentile(latencies, 0.50)
		p90 = calculatePercentile(latencies, 0.90)
		p95 = calculatePercentile(latencies, 0.95)
		p99 = calculatePercentile(latencies, 0.99)
	}

	// Calculate throughput stats
	stats.IntervalRatesMu.Lock()
	intervalRates := stats.IntervalRates
	stats.IntervalRatesMu.Unlock()

	var peakRate, minRate, avgRate float64
	if len(intervalRates) > 0 {
		sort.Float64s(intervalRates)
		peakRate = intervalRates[len(intervalRates)-1]
		minRate = intervalRates[0]
		sum := 0.0
		for _, r := range intervalRates {
			sum += r
		}
		avgRate = sum / float64(len(intervalRates))
	}

	successRate := 0.0
	if stats.Processed > 0 {
		successRate = float64(stats.Successful) / float64(stats.Processed) * 100
	}

	overallRate := float64(stats.Successful) / totalTime

	result := &Result{
		TotalImages:         stats.TotalImages,
		Processed:           stats.Processed,
		Successful:          stats.Successful,
		Failed:              stats.Failed,
		SuccessRate:         successRate,
		TotalTimeSeconds:    totalTime,
		TotalTimeHuman:      formatDuration(totalTime),
		ImagesPerSecond:     overallRate,
		ImagesPerMinute:     overallRate * 60,
		ImagesPerHour:       overallRate * 3600,
		PeakImagesPerSecond: peakRate,
		MinImagesPerSecond:  minRate,
		AvgImagesPerSecond:  avgRate,
		LatencyMeanMs:       meanLatency,
		LatencyMinMs:        minLatency,
		LatencyMaxMs:        maxLatency,
		LatencyP50Ms:        p50,
		LatencyP90Ms:        p90,
		LatencyP95Ms:        p95,
		LatencyP99Ms:        p99,
		Workers:             config.Workers,
		Directory:           config.Directory,
		Timestamp:           time.Now().Format(time.RFC3339),
	}

	return result, nil
}

func printResult(result *Result) {
	logger.Println()
	logger.Println("╔══════════════════════════════════════════════════════════════╗")
	logger.Println("║                    BENCHMARK RESULTS                         ║")
	logger.Println("╚══════════════════════════════════════════════════════════════╝")
	logger.Println()
	logger.Println("┌─────────────────────────────────────────────────────────────┐")
	logger.Println("│ SUMMARY                                                     │")
	logger.Println("├─────────────────────────────────────────────────────────────┤")
	logger.Printf("│ Total Images Found:        %d\n", result.TotalImages)
	logger.Printf("│ Images Processed:          %d\n", result.Processed)
	logger.Printf("│ Successful:                %d (%.1f%%)\n", result.Successful, result.SuccessRate)
	logger.Printf("│ Failed:                    %d\n", result.Failed)
	logger.Println("└─────────────────────────────────────────────────────────────┘")
	logger.Println()
	logger.Println("┌─────────────────────────────────────────────────────────────┐")
	logger.Println("│ TIMING                                                      │")
	logger.Println("├─────────────────────────────────────────────────────────────┤")
	logger.Printf("│ Total Time:                %.2f seconds (%s)\n", result.TotalTimeSeconds, result.TotalTimeHuman)
	logger.Println("└─────────────────────────────────────────────────────────────┘")
	logger.Println()
	logger.Println("┌─────────────────────────────────────────────────────────────┐")
	logger.Println("│ THROUGHPUT                                                  │")
	logger.Println("├─────────────────────────────────────────────────────────────┤")
	logger.Printf("│ Overall Rate:              %.2f images/second\n", result.ImagesPerSecond)
	logger.Printf("│ Per Minute:                %.0f images/minute\n", result.ImagesPerMinute)
	logger.Printf("│ Per Hour:                  %.0f images/hour\n", result.ImagesPerHour)
	logger.Printf("│ Peak Rate:                 %.2f images/second\n", result.PeakImagesPerSecond)
	logger.Printf("│ Min Rate:                  %.2f images/second\n", result.MinImagesPerSecond)
	logger.Println("└─────────────────────────────────────────────────────────────┘")
	logger.Println()
	logger.Println("┌─────────────────────────────────────────────────────────────┐")
	logger.Println("│ LATENCY                                                     │")
	logger.Println("├─────────────────────────────────────────────────────────────┤")
	logger.Printf("│ Mean:                      %.2f ms\n", result.LatencyMeanMs)
	logger.Printf("│ Min:                       %.2f ms\n", result.LatencyMinMs)
	logger.Printf("│ Max:                       %.2f ms\n", result.LatencyMaxMs)
	logger.Printf("│ P50 (median):              %.2f ms\n", result.LatencyP50Ms)
	logger.Printf("│ P90:                       %.2f ms\n", result.LatencyP90Ms)
	logger.Printf("│ P95:                       %.2f ms\n", result.LatencyP95Ms)
	logger.Printf("│ P99:                       %.2f ms\n", result.LatencyP99Ms)
	logger.Println("└─────────────────────────────────────────────────────────────┘")
	logger.Println()
	logger.Println("┌─────────────────────────────────────────────────────────────┐")
	logger.Println("│ CONFIGURATION                                               │")
	logger.Println("├─────────────────────────────────────────────────────────────┤")
	logger.Printf("│ Workers:                   %d\n", result.Workers)
	logger.Printf("│ Directory:                 %s\n", result.Directory)
	logger.Printf("│ Timestamp:                 %s\n", result.Timestamp)
	logger.Println("└─────────────────────────────────────────────────────────────┘")
}

func main() {
	config := &Config{}

	flag.StringVar(&config.Directory, "dir", "", "Directory containing images to ingest (required)")
	flag.StringVar(&config.APIHost, "host", "localhost", "API host")
	flag.IntVar(&config.APIPort, "port", 4603, "API port")
	flag.IntVar(&config.Workers, "workers", 64, "Number of concurrent workers")
	flag.DurationVar(&config.Timeout, "timeout", 120*time.Second, "Request timeout")
	flag.BoolVar(&config.ClearIndex, "clear", false, "Clear OpenSearch indexes before starting")
	flag.IntVar(&config.ReportEvery, "report", 10, "Report progress every N seconds")
	flag.StringVar(&config.LogFile, "log", "", "Log file path (default: benchmarks/results/ingest_<timestamp>.log)")

	// Pipeline control flags
	flag.BoolVar(&config.EnableDetection, "detection", true, "Enable YOLO object detection")
	flag.BoolVar(&config.EnableFaces, "faces", true, "Enable face detection and embedding (SCRFD + ArcFace)")
	flag.BoolVar(&config.EnableClip, "clip", true, "Enable CLIP image embedding (MobileCLIP)")
	flag.BoolVar(&config.EnableOCR, "ocr", false, "Enable OCR text extraction (disabled by default)")

	flag.Parse()

	if config.Directory == "" {
		fmt.Println("Error: -dir is required")
		flag.Usage()
		os.Exit(1)
	}

	// Setup logging
	timestamp := time.Now().Format("20060102_150405")
	if config.LogFile == "" {
		config.LogFile = fmt.Sprintf("benchmarks/results/ingest_%s.log", timestamp)
	}
	setupLogging(config.LogFile)
	defer closeLogging()

	result, err := runBenchmark(config)
	if err != nil {
		logger.Printf("[ERROR] Benchmark failed: %v\n", err)
		os.Exit(1)
	}

	printResult(result)

	// Save JSON results
	resultFile := fmt.Sprintf("benchmarks/results/ingest_%s.json", timestamp)
	jsonBytes, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile(resultFile, jsonBytes, 0644)
	logger.Printf("\n[SAVED] Results: %s\n", resultFile)
	logger.Printf("[SAVED] Log: %s\n", config.LogFile)
}
