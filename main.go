package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

// TradingData represents a single trading data point
type TradingData struct {
	Timestamp string  `json:"timestamp"`
	Price     float64 `json:"price"`
	Close     float64 `json:"close"`
	High      float64 `json:"high"`
	Low       float64 `json:"low"`
	Open      float64 `json:"open"`
	Volume    int64   `json:"volume"`
	Ticker    string  `json:"ticker"`
}

// Client represents a WebSocket client connection
type Client struct {
	conn *websocket.Conn
	send chan []byte
	hub  *Hub
	mu   sync.Mutex
}

// Hub manages all WebSocket connections
type Hub struct {
	clients    map[*Client]bool
	broadcast  chan []byte
	register   chan *Client
	unregister chan *Client
	mu         sync.RWMutex
}

// DataSimulator handles CSV data loading and simulation
type DataSimulator struct {
	data       []TradingData
	currentIdx int
	mu         sync.RWMutex
	speed      time.Duration
}

var (
	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true // Allow all origins for development
		},
	}
	simulator *DataSimulator
)

// NewHub creates a new WebSocket hub
func NewHub() *Hub {
	return &Hub{
		clients:    make(map[*Client]bool),
		broadcast:  make(chan []byte, 1000), // Buffered channel for high throughput
		register:   make(chan *Client, 100),
		unregister: make(chan *Client, 100),
	}
}

// NewDataSimulator creates a new data simulator
func NewDataSimulator(csvFile string, speed time.Duration) (*DataSimulator, error) {
	ds := &DataSimulator{
		speed: speed,
	}

	if err := ds.loadCSVData(csvFile); err != nil {
		return nil, err
	}

	return ds, nil
}

// loadCSVData loads and parses the CSV file
func (ds *DataSimulator) loadCSVData(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open CSV file: %v", err)
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	records, err := reader.ReadAll()
	if err != nil {
		return fmt.Errorf("failed to read CSV: %v", err)
	}

	// Skip header rows (first 3 rows)
	if len(records) < 4 {
		return fmt.Errorf("CSV file too short")
	}

	ds.mu.Lock()
	defer ds.mu.Unlock()

	// Parse data starting from row 4 (index 3)
	log.Printf("Total CSV records: %d", len(records))
	for i := 3; i < len(records); i++ {
		record := records[i]
		if len(record) < 6 {
			log.Printf("Skipping record %d: insufficient columns (%d)", i, len(record))
			continue
		}

		// Parse timestamp
		timestamp := strings.TrimSpace(record[0])
		if timestamp == "" {
			log.Printf("Skipping record %d: empty timestamp", i)
			continue
		}

		// Parse numeric values - CSV columns: Timestamp,Price,Close,High,Low,Volume
		price, _ := strconv.ParseFloat(strings.TrimSpace(record[1]), 64)
		close, _ := strconv.ParseFloat(strings.TrimSpace(record[2]), 64)
		high, _ := strconv.ParseFloat(strings.TrimSpace(record[3]), 64)
		low, _ := strconv.ParseFloat(strings.TrimSpace(record[4]), 64)
		open := price                                                       // Use price as open since there's no separate open column
		volume, _ := strconv.ParseInt(strings.TrimSpace(record[5]), 10, 64) // Volume is in the last column

		data := TradingData{
			Timestamp: timestamp,
			Price:     price,
			Close:     close,
			High:      high,
			Low:       low,
			Open:      open,
			Volume:    volume,
			Ticker:    "INFY.NS",
		}

		ds.data = append(ds.data, data)
	}

	log.Printf("Loaded %d trading records", len(ds.data))
	return nil
}

// GetNextData returns the next data point and advances the index
func (ds *DataSimulator) GetNextData() TradingData {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	if len(ds.data) == 0 {
		return TradingData{}
	}

	data := ds.data[ds.currentIdx]
	ds.currentIdx = (ds.currentIdx + 1) % len(ds.data) // Loop back to start

	return data
}

// StartSimulation starts the data simulation loop
func (ds *DataSimulator) StartSimulation(hub *Hub) {
	ticker := time.NewTicker(ds.speed)
	defer ticker.Stop()

	for range ticker.C {
		data := ds.GetNextData()
		if data.Timestamp == "" {
			continue
		}

		jsonData, err := json.Marshal(data)
		if err != nil {
			log.Printf("Error marshaling data: %v", err)
			continue
		}

		// Send to all connected clients
		hub.broadcast <- jsonData
	}
}

// run starts the hub's main loop
func (h *Hub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
			log.Printf("Client connected. Total clients: %d", len(h.clients))

		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				close(client.send)
			}
			h.mu.Unlock()
			log.Printf("Client disconnected. Total clients: %d", len(h.clients))

		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				select {
				case client.send <- message:
				default:
					// Client buffer is full, close connection
					close(client.send)
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// readPump handles reading messages from WebSocket clients
func (c *Client) readPump() {
	defer func() {
		c.hub.unregister <- c
		c.conn.Close()
	}()

	c.conn.SetReadLimit(512)
	c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	c.conn.SetPongHandler(func(string) error {
		c.conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		_, _, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket read error: %v", err)
			}
			break
		}
	}
}

// writePump handles writing messages to WebSocket clients
func (c *Client) writePump() {
	ticker := time.NewTicker(54 * time.Second)
	defer func() {
		ticker.Stop()
		c.conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.send:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if !ok {
				c.conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			w, err := c.conn.NextWriter(websocket.TextMessage)
			if err != nil {
				return
			}
			w.Write(message)

			// Add queued messages to the current websocket message
			n := len(c.send)
			for i := 0; i < n; i++ {
				w.Write([]byte("\n"))
				w.Write(<-c.send)
			}

			if err := w.Close(); err != nil {
				return
			}

		case <-ticker.C:
			c.conn.SetWriteDeadline(time.Now().Add(10 * time.Second))
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return
			}
		}
	}
}

// serveWs handles WebSocket connections
func serveWs(hub *Hub, c *gin.Context) {
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return
	}

	client := &Client{
		hub:  hub,
		conn: conn,
		send: make(chan []byte, 256), // Buffered channel for high throughput
	}

	client.hub.register <- client

	// Start goroutines for reading and writing
	go client.writePump()
	go client.readPump()
}

// setupRouter configures the Gin router
func setupRouter(hub *Hub) *gin.Engine {
	router := gin.Default()

	// Configure CORS
	config := cors.DefaultConfig()
	config.AllowAllOrigins = true
	config.AllowHeaders = []string{"Origin", "Content-Length", "Content-Type", "Authorization"}
	router.Use(cors.New(config))

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status": "healthy",
			"time":   time.Now().UTC(),
		})
	})

	// WebSocket endpoint
	router.GET("/ws", func(c *gin.Context) {
		serveWs(hub, c)
	})

	// REST API endpoint for current data
	router.GET("/api/current", func(c *gin.Context) {
		if simulator == nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Simulator not initialized"})
			return
		}

		data := simulator.GetNextData()
		c.JSON(http.StatusOK, data)
	})

	// Serve static files
	router.StaticFile("/", "./client.html")
	router.StaticFile("/client.html", "./client.html")
	router.StaticFile("/test", "./test_websocket.html")

	return router
}

func main() {
	// Initialize data simulator
	var err error
	simulator, err = NewDataSimulator("infy_2min_60days.csv", 500*time.Millisecond) // 2-second intervals
	if err != nil {
		log.Fatalf("Failed to initialize data simulator: %v", err)
	}

	// Initialize WebSocket hub
	hub := NewHub()
	go hub.run()

	// Start data simulation
	go simulator.StartSimulation(hub)

	// Setup and start HTTP server
	router := setupRouter(hub)

	port := ":8080"
	log.Printf("Starting trading API server on port %s", port)
	log.Printf("WebSocket endpoint: ws://localhost%s/ws", port)
	log.Printf("Health check: http://localhost%s/health", port)
	log.Printf("Current data API: http://localhost%s/api/current", port)

	if err := router.Run(port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
