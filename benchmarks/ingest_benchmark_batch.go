// Batch Ingest Benchmark - High-throughput parallel batch ingestion
//
// Usage:
//   go run ingest_benchmark_batch.go -dir /path/to/images -workers 16 -batch 32 -clear
//
// This uses batch ingestion to maximize GPU utilization by:
// - Pre-loading images into memory queue (producers)
// - Sending batches of images to /ingest/batch endpoint (consumers)
// - Keeping the inference pipeline constantly full

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

type Config struct {
	Directory       string
	APIHost         string
	APIPort         int
	Workers         int // Number of batch sender workers
	BatchSize       int // Images per batch (max 64)
	Timeout         time.Duration
	ClearIndex      bool
	ReportEvery     int
	LogFile         string
	EnableDetection bool
	EnableFaces     bool
	EnableClip      bool
	EnableOCR       bool
	QueueSize       int // Pre-load queue size
}

type ImageData struct {
	Path  string
	Bytes []byte
}

type Stats struct {
	TotalImages     int64
	Processed       int64
	Successful      int64
	Failed          int64
	BatchesSent     int64
	StartTime       time.Time
	Latencies       []float64
	LatenciesMutex  sync.Mutex
	IntervalRates   []float64
	IntervalRatesMu sync.Mutex
	LastReportTime  time.Time
	LastReportCount int64
}

type Result struct {
	TotalImages         int64   `json:"total_images"`
	Processed           int64   `json:"processed"`
	Successful          int64   `json:"successful"`
	Failed              int64   `json:"failed"`
	BatchesSent         int64   `json:"batches_sent"`
	SuccessRate         float64 `json:"success_rate_percent"`
	TotalTimeSeconds    float64 `json:"total_time_seconds"`
	TotalTimeHuman      string  `json:"total_time_human"`
	ImagesPerSecond     float64 `json:"images_per_second"`
	ImagesPerMinute     float64 `json:"images_per_minute"`
	ImagesPerHour       float64 `json:"images_per_hour"`
	PeakImagesPerSecond float64 `json:"peak_images_per_second"`
	LatencyMeanMs       float64 `json:"latency_mean_ms"`
	LatencyP50Ms        float64 `json:"latency_p50_ms"`
	LatencyP95Ms        float64 `json:"latency_p95_ms"`
	LatencyP99Ms        float64 `json:"latency_p99_ms"`
	Workers             int     `json:"workers"`
	BatchSize           int     `json:"batch_size"`
	Directory           string  `json:"directory"`
	Timestamp           string  `json:"timestamp"`
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

// imageLoader reads images from disk and puts them in the queue
func imageLoader(paths []string, queue chan<- ImageData, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(queue)

	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		queue <- ImageData{Path: path, Bytes: data}
	}
}

// batchCollector collects images into batches
func batchCollector(input <-chan ImageData, output chan<- []ImageData, batchSize int, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(output)

	batch := make([]ImageData, 0, batchSize)
	for img := range input {
		batch = append(batch, img)
		if len(batch) >= batchSize {
			// Send batch
			batchCopy := make([]ImageData, len(batch))
			copy(batchCopy, batch)
			output <- batchCopy
			batch = batch[:0]
		}
	}
	// Send remaining
	if len(batch) > 0 {
		output <- batch
	}
}

// sendBatch sends a batch of images to the API
func sendBatch(client *http.Client, url string, batch []ImageData, config *Config) (int, int, float64, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	// Add all images
	imageIDs := make([]string, len(batch))
	imagePaths := make([]string, len(batch))

	for i, img := range batch {
		part, err := writer.CreateFormFile("images", filepath.Base(img.Path))
		if err != nil {
			return 0, len(batch), 0, err
		}
		part.Write(img.Bytes)
		imageIDs[i] = img.Path
		imagePaths[i] = img.Path
	}

	// Add metadata
	idsJSON, _ := json.Marshal(imageIDs)
	pathsJSON, _ := json.Marshal(imagePaths)
	writer.WriteField("image_ids", string(idsJSON))
	writer.WriteField("image_paths", string(pathsJSON))
	writer.WriteField("skip_duplicates", "false")
	writer.WriteField("detect_near_duplicates", "false")
	writer.WriteField("enable_ocr", fmt.Sprintf("%t", config.EnableOCR))
	writer.WriteField("enable_detection", fmt.Sprintf("%t", config.EnableDetection))
	writer.WriteField("enable_faces", fmt.Sprintf("%t", config.EnableFaces))
	writer.WriteField("enable_clip", fmt.Sprintf("%t", config.EnableClip))
	writer.Close()

	req, err := http.NewRequest("POST", url, body)
	if err != nil {
		return 0, len(batch), 0, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	start := time.Now()
	resp, err := client.Do(req)
	latency := float64(time.Since(start).Milliseconds())

	if err != nil {
		return 0, len(batch), latency, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 201 {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return 0, len(batch), latency, fmt.Errorf("status %d: %s", resp.StatusCode, string(bodyBytes[:min(200, len(bodyBytes))]))
	}

	// Parse response to get success count
	var result map[string]interface{}
	json.NewDecoder(resp.Body).Decode(&result)

	processed := int(result["processed"].(float64))
	return processed, len(batch) - processed, latency, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// batchWorker sends batches to the API
func batchWorker(id int, batches <-chan []ImageData, stats *Stats, config *Config, wg *sync.WaitGroup) {
	defer wg.Done()

	client := &http.Client{
		Timeout: config.Timeout,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 100,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	url := fmt.Sprintf("http://%s:%d/ingest/batch", config.APIHost, config.APIPort)

	for batch := range batches {
		success, failed, latency, err := sendBatch(client, url, batch, config)

		atomic.AddInt64(&stats.BatchesSent, 1)
		atomic.AddInt64(&stats.Processed, int64(len(batch)))

		if err != nil {
			atomic.AddInt64(&stats.Failed, int64(len(batch)))
		} else {
			atomic.AddInt64(&stats.Successful, int64(success))
			atomic.AddInt64(&stats.Failed, int64(failed))

			stats.LatenciesMutex.Lock()
			stats.Latencies = append(stats.Latencies, latency)
			stats.LatenciesMutex.Unlock()
		}
	}
}

func clearIndexes(opensearchPort int) error {
	osURL := fmt.Sprintf("http://localhost:%d", opensearchPort)
	indexes := []string{"visual_search_global", "visual_search_faces", "visual_search_vehicles", "visual_search_people"}
	client := &http.Client{Timeout: 30 * time.Second}

	for _, index := range indexes {
		req, _ := http.NewRequest("DELETE", osURL+"/"+index, nil)
		resp, _ := client.Do(req)
		if resp != nil {
			resp.Body.Close()
		}
	}
	logger.Println("[SETUP] Deleted OpenSearch indexes")

	// Create global index with k-NN mapping
	globalMapping := `{
		"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": true, "refresh_interval": "-1"}},
		"mappings": {"properties": {
			"image_id": {"type": "keyword"}, "image_path": {"type": "keyword"},
			"global_embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss", "parameters": {"ef_construction": 512, "m": 16}}},
			"imohash": {"type": "keyword"}, "indexed_at": {"type": "date"}
		}}
	}`

	facesMapping := `{
		"settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0, "knn": true, "refresh_interval": "-1"}},
		"mappings": {"properties": {
			"face_id": {"type": "keyword"}, "image_id": {"type": "keyword"}, "image_path": {"type": "keyword"},
			"embedding": {"type": "knn_vector", "dimension": 512, "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss", "parameters": {"ef_construction": 512, "m": 16}}},
			"box": {"type": "float"}, "confidence": {"type": "float"}, "quality_score": {"type": "float"},
			"landmarks": {"type": "object", "properties": {"left_eye": {"type": "float"}, "right_eye": {"type": "float"}, "nose": {"type": "float"}, "left_mouth": {"type": "float"}, "right_mouth": {"type": "float"}}},
			"person_id": {"type": "keyword"}, "person_name": {"type": "keyword"}, "indexed_at": {"type": "date"}
		}}
	}`

	req, _ := http.NewRequest("PUT", osURL+"/visual_search_global", strings.NewReader(globalMapping))
	req.Header.Set("Content-Type", "application/json")
	resp, _ := client.Do(req)
	if resp != nil {
		resp.Body.Close()
	}
	logger.Println("[SETUP] Created visual_search_global with k-NN")

	req, _ = http.NewRequest("PUT", osURL+"/visual_search_faces", strings.NewReader(facesMapping))
	req.Header.Set("Content-Type", "application/json")
	resp, _ = client.Do(req)
	if resp != nil {
		resp.Body.Close()
	}
	logger.Println("[SETUP] Created visual_search_faces with k-NN")

	time.Sleep(1 * time.Second)
	return nil
}

func formatDuration(seconds float64) string {
	if seconds < 60 {
		return fmt.Sprintf("%.0fs", seconds)
	} else if seconds < 3600 {
		return fmt.Sprintf("%.0fm %.0fs", seconds/60, float64(int(seconds)%60))
	}
	hours := int(seconds / 3600)
	mins := int(seconds/60) % 60
	secs := int(seconds) % 60
	return fmt.Sprintf("%dh %dm %ds", hours, mins, secs)
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
			batches := atomic.LoadInt64(&stats.BatchesSent)

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

			if currentRate > 0 {
				stats.IntervalRatesMu.Lock()
				stats.IntervalRates = append(stats.IntervalRates, currentRate)
				stats.IntervalRatesMu.Unlock()
			}

			logger.Printf("[METRIC] %s | Progress: %d/%d (%.1f%%) | Rate: %.1f/s (cur) %.1f/s (avg) | Batches: %d | Success: %d | Failed: %d | ETA: %s",
				now.Format("15:04:05"),
				processed, stats.TotalImages, progressPct,
				currentRate, overallRate,
				batches, successful, failed,
				formatDuration(etaSeconds),
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
	logger.Println("║         OPENPROCESSOR BATCH INGEST BENCHMARK                 ║")
	logger.Println("╚══════════════════════════════════════════════════════════════╝")
	logger.Println()
	logger.Printf("[CONFIG] Directory: %s\n", config.Directory)
	logger.Printf("[CONFIG] Workers: %d (batch senders)\n", config.Workers)
	logger.Printf("[CONFIG] Batch Size: %d images/batch\n", config.BatchSize)
	logger.Printf("[CONFIG] Queue Size: %d images pre-loaded\n", config.QueueSize)
	logger.Printf("[CONFIG] API: http://%s:%d/ingest/batch\n", config.APIHost, config.APIPort)
	logger.Println()
	logger.Println("[PIPELINE] Enabled:")
	logger.Printf("[PIPELINE]   Detection: %t | CLIP: %t | Faces: %t | OCR: %t\n",
		config.EnableDetection, config.EnableClip, config.EnableFaces, config.EnableOCR)
	logger.Println()

	// Find images
	logger.Println("[SCAN] Scanning for images...")
	images, err := findAllImages(config.Directory)
	if err != nil {
		return nil, err
	}
	logger.Printf("[SCAN] Found %d images\n", len(images))

	if config.ClearIndex {
		clearIndexes(4607)
	}

	stats := &Stats{
		TotalImages:   int64(len(images)),
		StartTime:     time.Now(),
		Latencies:     make([]float64, 0, len(images)/config.BatchSize+1),
		IntervalRates: make([]float64, 0, 1000),
	}

	// Create pipeline: images -> loader -> queue -> batcher -> batches -> workers
	imageQueue := make(chan ImageData, config.QueueSize)
	batchQueue := make(chan []ImageData, config.Workers*2)

	var loaderWg sync.WaitGroup
	var batcherWg sync.WaitGroup
	var workerWg sync.WaitGroup

	// Start image loader (producer)
	loaderWg.Add(1)
	go imageLoader(images, imageQueue, &loaderWg)

	// Start batch collector
	batcherWg.Add(1)
	go batchCollector(imageQueue, batchQueue, config.BatchSize, &batcherWg)

	// Start batch workers (consumers)
	for i := 0; i < config.Workers; i++ {
		workerWg.Add(1)
		go batchWorker(i, batchQueue, stats, config, &workerWg)
	}

	// Start progress reporter
	done := make(chan bool)
	go progressReporter(stats, config, done)

	logger.Println()
	logger.Printf("[START] Beginning batch ingest: %d images, %d workers, batch size %d\n",
		len(images), config.Workers, config.BatchSize)
	logger.Println()

	// Wait for completion
	loaderWg.Wait()
	batcherWg.Wait()
	workerWg.Wait()
	done <- true

	totalTime := time.Since(stats.StartTime).Seconds()

	// Calculate stats
	stats.LatenciesMutex.Lock()
	latencies := make([]float64, len(stats.Latencies))
	copy(latencies, stats.Latencies)
	stats.LatenciesMutex.Unlock()

	var meanLatency, p50, p95, p99 float64
	if len(latencies) > 0 {
		sort.Float64s(latencies)
		sum := 0.0
		for _, l := range latencies {
			sum += l
		}
		meanLatency = sum / float64(len(latencies))
		p50 = calculatePercentile(latencies, 0.50)
		p95 = calculatePercentile(latencies, 0.95)
		p99 = calculatePercentile(latencies, 0.99)
	}

	stats.IntervalRatesMu.Lock()
	var peakRate float64
	for _, r := range stats.IntervalRates {
		if r > peakRate {
			peakRate = r
		}
	}
	stats.IntervalRatesMu.Unlock()

	successRate := 0.0
	if stats.Processed > 0 {
		successRate = float64(stats.Successful) / float64(stats.Processed) * 100
	}

	result := &Result{
		TotalImages:         stats.TotalImages,
		Processed:           stats.Processed,
		Successful:          stats.Successful,
		Failed:              stats.Failed,
		BatchesSent:         stats.BatchesSent,
		SuccessRate:         successRate,
		TotalTimeSeconds:    totalTime,
		TotalTimeHuman:      formatDuration(totalTime),
		ImagesPerSecond:     float64(stats.Successful) / totalTime,
		ImagesPerMinute:     float64(stats.Successful) / totalTime * 60,
		ImagesPerHour:       float64(stats.Successful) / totalTime * 3600,
		PeakImagesPerSecond: peakRate,
		LatencyMeanMs:       meanLatency,
		LatencyP50Ms:        p50,
		LatencyP95Ms:        p95,
		LatencyP99Ms:        p99,
		Workers:             config.Workers,
		BatchSize:           config.BatchSize,
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
	logger.Printf("│ Total Images:      %d\n", result.TotalImages)
	logger.Printf("│ Processed:         %d\n", result.Processed)
	logger.Printf("│ Successful:        %d (%.1f%%)\n", result.Successful, result.SuccessRate)
	logger.Printf("│ Failed:            %d\n", result.Failed)
	logger.Printf("│ Batches Sent:      %d\n", result.BatchesSent)
	logger.Println("├──────────────────────────────────────────────────────────────")
	logger.Printf("│ Total Time:        %.2fs (%s)\n", result.TotalTimeSeconds, result.TotalTimeHuman)
	logger.Printf("│ Throughput:        %.2f images/second\n", result.ImagesPerSecond)
	logger.Printf("│ Per Minute:        %.0f images/minute\n", result.ImagesPerMinute)
	logger.Printf("│ Per Hour:          %.0f images/hour\n", result.ImagesPerHour)
	logger.Printf("│ Peak Rate:         %.2f images/second\n", result.PeakImagesPerSecond)
	logger.Println("├──────────────────────────────────────────────────────────────")
	logger.Printf("│ Batch Latency Mean: %.2f ms\n", result.LatencyMeanMs)
	logger.Printf("│ Batch Latency P50:  %.2f ms\n", result.LatencyP50Ms)
	logger.Printf("│ Batch Latency P95:  %.2f ms\n", result.LatencyP95Ms)
	logger.Printf("│ Batch Latency P99:  %.2f ms\n", result.LatencyP99Ms)
	logger.Println("└──────────────────────────────────────────────────────────────")
}

func main() {
	config := &Config{}

	flag.StringVar(&config.Directory, "dir", "", "Directory containing images (required)")
	flag.StringVar(&config.APIHost, "host", "localhost", "API host")
	flag.IntVar(&config.APIPort, "port", 4603, "API port")
	flag.IntVar(&config.Workers, "workers", 8, "Number of batch sender workers")
	flag.IntVar(&config.BatchSize, "batch", 32, "Images per batch (max 64)")
	flag.IntVar(&config.QueueSize, "queue", 256, "Pre-load queue size")
	flag.DurationVar(&config.Timeout, "timeout", 300*time.Second, "Request timeout")
	flag.BoolVar(&config.ClearIndex, "clear", false, "Clear indexes before starting")
	flag.IntVar(&config.ReportEvery, "report", 10, "Report interval in seconds")
	flag.StringVar(&config.LogFile, "log", "", "Log file path")

	flag.BoolVar(&config.EnableDetection, "detection", true, "Enable YOLO detection")
	flag.BoolVar(&config.EnableFaces, "faces", true, "Enable face detection")
	flag.BoolVar(&config.EnableClip, "clip", true, "Enable CLIP embedding")
	flag.BoolVar(&config.EnableOCR, "ocr", false, "Enable OCR")

	flag.Parse()

	if config.Directory == "" {
		fmt.Println("Error: -dir is required")
		flag.Usage()
		os.Exit(1)
	}

	if config.BatchSize > 64 {
		config.BatchSize = 64
	}

	timestamp := time.Now().Format("20060102_150405")
	if config.LogFile == "" {
		config.LogFile = fmt.Sprintf("benchmarks/results/batch_ingest_%s.log", timestamp)
	}
	setupLogging(config.LogFile)
	defer closeLogging()

	result, err := runBenchmark(config)
	if err != nil {
		logger.Printf("[ERROR] Benchmark failed: %v\n", err)
		os.Exit(1)
	}

	printResult(result)

	resultFile := fmt.Sprintf("benchmarks/results/batch_ingest_%s.json", timestamp)
	jsonBytes, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile(resultFile, jsonBytes, 0644)
	logger.Printf("\n[SAVED] %s\n", resultFile)
}
