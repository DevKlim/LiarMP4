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

	// HF Spaces: Files are copied to /app/static in Dockerfile
	staticPath := "/app/static"
	fs := http.FileServer(http.Dir(staticPath))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Proxy API requests to Python
		if strings.HasPrefix(r.URL.Path, "/process") ||
			strings.HasPrefix(r.URL.Path, "/label_video") ||
			strings.HasPrefix(r.URL.Path, "/batch_label") ||
			strings.HasPrefix(r.URL.Path, "/model-architecture") ||
			strings.HasPrefix(r.URL.Path, "/download-dataset") ||
			strings.HasPrefix(r.URL.Path, "/extension") ||
			strings.HasPrefix(r.URL.Path, "/manage") ||
			strings.HasPrefix(r.URL.Path, "/queue") {

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

	// HF Spaces requires listening on port 7860
	port := "7860"
	log.Printf("vChat HF Server listening on port %s", port)
	log.Printf("Serving static files from %s", staticPath)
	if err := http.ListenAndServe(":"+port, nil); err != nil {
		log.Fatal(err)
	}
}