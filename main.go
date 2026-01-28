package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
)

func main() {
	// Target Python FastAPI server (running locally in the container)
	pythonTarget := "http://127.0.0.1:8001"
	pythonURL, err := url.Parse(pythonTarget)
	if err != nil {
		log.Fatalf("Invalid Python target URL: %v", err)
	}

	// Create Reverse Proxy
	proxy := httputil.NewSingleHostReverseProxy(pythonURL)

	// FIXED: Point to the new location safe from Docker Volumes
	staticPath := "/usr/share/vchat/static"
	fs := http.FileServer(http.Dir(staticPath))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Check if request is for the API
		// We filter requests by prefix to forward to FastAPI
		if strings.HasPrefix(r.URL.Path, "/process") ||
			strings.HasPrefix(r.URL.Path, "/label_video") ||
			strings.HasPrefix(r.URL.Path, "/batch_label") ||
			strings.HasPrefix(r.URL.Path, "/model-architecture") ||
			strings.HasPrefix(r.URL.Path, "/download-dataset") ||
			strings.HasPrefix(r.URL.Path, "/extension") ||
			strings.HasPrefix(r.URL.Path, "/manage") ||
			strings.HasPrefix(r.URL.Path, "/workflow") ||
			strings.HasPrefix(r.URL.Path, "/queue") ||
			// NEW ROUTES ADDED HERE
			strings.HasPrefix(r.URL.Path, "/profiles") ||
			strings.HasPrefix(r.URL.Path, "/community") ||
			strings.HasPrefix(r.URL.Path, "/analyze") ||
			strings.HasPrefix(r.URL.Path, "/analytics") ||
			strings.HasPrefix(r.URL.Path, "/dataset") ||
			// FIXED: Added missing routes for Manual Labeling and Config
			strings.HasPrefix(r.URL.Path, "/manual") ||
			strings.HasPrefix(r.URL.Path, "/benchmarks") ||
			strings.HasPrefix(r.URL.Path, "/config") {

			log.Printf("Proxying %s to Python Backend...", r.URL.Path)
			proxy.ServeHTTP(w, r)
			return
		}

		// Check if file exists in static dir, otherwise serve index.html (SPA Routing)
		path := staticPath + r.URL.Path
		if _, err := os.Stat(path); os.IsNotExist(err) {
			http.ServeFile(w, r, staticPath+"/index.html")
			return
		}

		fs.ServeHTTP(w, r)
	})

	port := "8000"
	log.Printf("vChat Modern Server listening on port %s", port)
	log.Printf("Serving static files from %s", staticPath)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}
