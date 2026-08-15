package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/event"
	"github.com/mostlygeek/llama-swap/internal/logmon"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/store"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// stubRouter is a minimal router.LocalRouter for Server dispatch tests.
type stubRouter struct {
	models        map[string]bool
	response      string
	serveHTTP     func(http.ResponseWriter, *http.Request)
	shutdownCalls atomic.Int32
	running       map[string]process.ProcessState
	unloadCalls   atomic.Int32
	unloadModels  []string
	unloadTimeout time.Duration
	loggers       map[string]*logmon.Monitor
	// statuses backs RunningModelStatus. It is separate from running so a
	// test only has to populate the listing the handler under test reads.
	statuses map[string]router.ModelStatus
	// serveCalls counts dispatches into the router. Listing tests assert it
	// stays at zero: a dispatch is the only seam through which a request can
	// make llama-swap load a model.
	serveCalls atomic.Int32
}

func newStubRouter(models []string, response string) *stubRouter {
	m := make(map[string]bool, len(models))
	for _, id := range models {
		m[id] = true
	}
	return &stubRouter{models: m, response: response}
}

func (s *stubRouter) Handles(model string) bool      { return s.models[model] }
func (s *stubRouter) Shutdown(_ time.Duration) error { s.shutdownCalls.Add(1); return nil }
func (s *stubRouter) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.serveCalls.Add(1)
	if s.serveHTTP != nil {
		s.serveHTTP(w, r)
		return
	}
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(s.response))
}

func (s *stubRouter) RunningModels() map[string]process.ProcessState { return s.running }
func (s *stubRouter) RunningModelStatus() map[string]router.ModelStatus {
	return s.statuses
}
func (s *stubRouter) Unload(timeout time.Duration, models ...string) {
	s.unloadCalls.Add(1)
	s.unloadTimeout = timeout
	s.unloadModels = append([]string(nil), models...)
}
func (s *stubRouter) ProcessLogger(modelID string) (*logmon.Monitor, bool) {
	if s.loggers != nil {
		if lg, ok := s.loggers[modelID]; ok {
			return lg, true
		}
	}
	return nil, false
}

// newTestServer wires a Server with stub routers and a built mux.
func newTestServer(local router.LocalRouter, peer router.Router) *Server {
	ctx, cancel := context.WithCancel(context.Background())
	proxylog := logmon.NewWriter(io.Discard)
	st, err := store.New("")
	if err != nil {
		panic(err)
	}
	s := &Server{
		cfg:         config.Config{},
		muxlog:      logmon.NewWriter(io.Discard),
		proxylog:    proxylog,
		upstreamlog: logmon.NewWriter(io.Discard),
		inflight:    newInflightTracker(),
		metrics:     newMetricsMonitor(proxylog, 0, 0, st),
		store:       st,
		local:       local,
		peer:        peer,
		shutdownCtx: ctx,
		shutdownFn:  cancel,
	}
	s.routes()
	return s
}

func newTestMetricsMonitor(t *testing.T, logger *logmon.Monitor, maxMetrics int, captureBufferMB int) *metricsMonitor {
	t.Helper()
	st, err := store.New("")
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	t.Cleanup(func() {
		if err := st.Close(); err != nil {
			t.Errorf("store.Close: %v", err)
		}
	})
	return newMetricsMonitor(logger, maxMetrics, captureBufferMB, st)
}

func metricsEntries(t *testing.T, mm *metricsMonitor) []ActivityLogEntry {
	t.Helper()
	page, err := mm.store.ListActivity(context.Background(), store.ActivityQuery{Limit: 1000, Page: 1})
	if err != nil {
		t.Fatalf("ListActivity: %v", err)
	}
	mm.overlayCaptureState(page.Data)
	return page.Data
}

func chatRequest(model string) *http.Request {
	body := strings.NewReader(`{"model":"` + model + `"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
	req.Header.Set("Content-Type", "application/json")
	return req
}

func TestServer_New_GroupConfig(t *testing.T) {
	discard := logmon.NewWriter(io.Discard)
	cfg := config.Config{HealthCheckTimeout: 15}
	cfg.Routing.Router.Use = "group"
	st, err := store.New("")
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	defer st.Close()
	s, err := New(cfg, discard, discard, discard, nil, st, BuildInfo{}, nil)
	if err != nil {
		t.Fatalf("New (group): %v", err)
	}
	if _, ok := s.local.(*router.Group); !ok {
		t.Fatalf("localRouter=%T want *router.Group", s.local)
	}
	if err := s.Shutdown(time.Second); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
}

func TestServer_New_MatrixConfig(t *testing.T) {
	discard := logmon.NewWriter(io.Discard)
	cfg := config.Config{HealthCheckTimeout: 15}
	cfg.Models = map[string]config.ModelConfig{
		"model": {
			Cmd:   "echo ready",
			Proxy: "http://localhost:8080",
		},
	}
	cfg.Routing.Router.Use = "matrix"
	cfg.Routing.Router.Settings.Matrix = &config.MatrixConfig{
		Sets: config.OrderedSets{{Name: "single", DSL: "model"}},
	}
	st, err := store.New("")
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	defer st.Close()
	s, err := New(cfg, discard, discard, discard, nil, st, BuildInfo{}, nil)
	if err != nil {
		t.Fatalf("New (matrix): %v", err)
	}
	if _, ok := s.local.(*router.Matrix); !ok {
		t.Fatalf("localRouter=%T want *router.Matrix", s.local)
	}
	if err := s.Shutdown(time.Second); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
}

func TestServer_RouteToLocalModel(t *testing.T) {
	s := newTestServer(
		newStubRouter([]string{"local-model"}, "local response"),
		newStubRouter(nil, ""),
	)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("local-model"))

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if w.Body.String() != "local response" {
		t.Errorf("body=%q want %q", w.Body.String(), "local response")
	}
}

func TestServer_RouteToPeerModel(t *testing.T) {
	s := newTestServer(
		newStubRouter(nil, ""),
		newStubRouter([]string{"peer-model"}, "peer response"),
	)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("peer-model"))

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if w.Body.String() != "peer response" {
		t.Errorf("body=%q want %q", w.Body.String(), "peer response")
	}
}

func TestServer_RouteToLocalModel_PrefersLocalCollision(t *testing.T) {
	s := newTestServer(
		newStubRouter([]string{"shared"}, "local response"),
		newStubRouter([]string{"shared"}, "peer response"),
	)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("shared"))

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if w.Body.String() != "local response" {
		t.Errorf("body=%q want local response", w.Body.String())
	}
}

func TestServer_UnknownModelReturns404(t *testing.T) {
	s := newTestServer(
		newStubRouter([]string{"local-model"}, ""),
		newStubRouter(nil, ""),
	)

	w := httptest.NewRecorder()
	s.ServeHTTP(w, chatRequest("unknown-model"))

	if w.Code != http.StatusNotFound {
		t.Errorf("status=%d want 404 body=%q", w.Code, w.Body.String())
	}
}

func TestServer_UnknownPathReturns404(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/does-not-exist", nil))

	if w.Code != http.StatusNotFound {
		t.Errorf("status=%d want 404", w.Code)
	}
}

func TestServer_Health(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	for _, path := range []string{"/health", "/wol-health"} {
		w := httptest.NewRecorder()
		s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
		if w.Code != http.StatusOK || w.Body.String() != "OK" {
			t.Errorf("%s: status=%d body=%q", path, w.Code, w.Body.String())
		}
	}
}

func TestServer_CORSPreflight(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	req := httptest.NewRequest(http.MethodOptions, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusNoContent {
		t.Fatalf("status=%d want 204", w.Code)
	}
	if got := w.Header().Get("Access-Control-Allow-Origin"); got != "*" {
		t.Errorf("Access-Control-Allow-Origin=%q want *", got)
	}
}

func TestServer_Unload(t *testing.T) {
	local := newStubRouter([]string{"m1"}, "")
	s := newTestServer(local, newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/unload", nil))

	if w.Code != http.StatusOK || w.Body.String() != "OK" {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if got := local.unloadCalls.Load(); got != 1 {
		t.Errorf("unloadCalls=%d want 1", got)
	}
	if len(local.unloadModels) != 0 {
		t.Errorf("unloadModels=%v want empty for unload all", local.unloadModels)
	}
	if local.unloadTimeout != 0 {
		t.Errorf("unloadTimeout=%v want 0 (use configured timeouts)", local.unloadTimeout)
	}
}

// getRunning asks the server for /running and decodes the listing.
func getRunning(t *testing.T, s *Server) []runningModel {
	t.Helper()
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/running", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	var resp struct {
		Running []runningModel `json:"running"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v body=%q", err, w.Body.String())
	}
	return resp.Running
}

// addInflight registers n in-flight requests against modelID and removes them
// again when the test ends.
func addInflight(t *testing.T, s *Server, modelID string, n int) {
	t.Helper()
	for range n {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req = req.WithContext(swaputil.SetContext(req.Context(),
			swaputil.ReqContextData{Model: modelID, ModelID: modelID}))
		id := s.inflight.Add(req, func() {})
		t.Cleanup(func() { s.inflight.Remove(id) })
	}
}

// completeInflight simulates one already-finished request against modelID:
// it registers with the tracker and immediately removes itself, so
// LastCompletionByModel records the completion time without leaving
// in_flight_requests non-zero.
func completeInflight(t *testing.T, s *Server, modelID string) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req = req.WithContext(swaputil.SetContext(req.Context(),
		swaputil.ReqContextData{Model: modelID, ModelID: modelID}))
	id := s.inflight.Add(req, func() {})
	s.inflight.Remove(id)
}

func TestServer_Running(t *testing.T) {
	local := newStubRouter([]string{"m1"}, "")
	since := time.Now().Add(-90 * time.Second)
	local.statuses = map[string]router.ModelStatus{
		"m1": {State: process.StateReady, Since: since},
	}
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{
		"m1": {
			Cmd:         "llama-server",
			Proxy:       "http://localhost:9999",
			UnloadAfter: 300,
			Name:        "Model One",
			Description: "the first model",
		},
	}}

	running := getRunning(t, s)
	if len(running) != 1 {
		t.Fatalf("running=%v want 1 entry", running)
	}
	got := running[0]

	// Compared field by field rather than with ==: Busy is a pointer, so struct
	// equality would compare pointer identity, and == on a time.Time compares
	// the monotonic reading and location as well as the instant.
	want := runningModel{
		Model:       "m1",
		State:       "ready",
		Cmd:         "llama-server",
		Proxy:       "http://localhost:9999",
		TTL:         300,
		Name:        "Model One",
		Description: "the first model",
	}
	if got.Model != want.Model || got.State != want.State || got.Cmd != want.Cmd ||
		got.Proxy != want.Proxy || got.TTL != want.TTL || got.Name != want.Name ||
		got.Description != want.Description {
		t.Errorf("got %+v want %+v", got, want)
	}
	if got.Busy == nil {
		t.Fatalf("busy = null for a ready model, want false")
	}
	if *got.Busy {
		t.Errorf("busy = true with nothing in flight, want false")
	}
	if got.InFlightRequests != 0 {
		t.Errorf("in_flight_requests = %d, want 0", got.InFlightRequests)
	}
	if !got.Since.Equal(since) {
		t.Errorf("since = %v, want the transition time %v", got.Since, since)
	}
	if loc := got.Since.Location(); loc != time.UTC {
		t.Errorf("since location = %v, want UTC", loc)
	}
	if n := local.serveCalls.Load(); n != 0 {
		t.Errorf("router dispatches = %d, want 0: listing a model must never load it", n)
	}
}

// TestServer_RunningBusyAndInFlight covers the fields /running reports beyond
// the model's config: busy is null unless the model is ready, in_flight_requests
// comes from llama-swap's own bookkeeping for that model alone, and since is
// the model's last transition. None of it dispatches into the router.
func TestServer_RunningBusyAndInFlight(t *testing.T) {
	tests := []struct {
		name         string
		state        process.ProcessState
		inFlight     int
		wantBusyNull bool
		wantBusy     bool
	}{
		{name: "starting is neither busy nor idle", state: process.StateStarting, wantBusyNull: true},
		{name: "stopping is neither busy nor idle", state: process.StateStopping, wantBusyNull: true},
		{name: "starting with requests queued is still not busy", state: process.StateStarting, inFlight: 1, wantBusyNull: true},
		{name: "ready with nothing in flight is idle", state: process.StateReady, wantBusy: false},
		{name: "ready with requests in flight is busy", state: process.StateReady, inFlight: 2, wantBusy: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			local := newStubRouter([]string{"m1"}, "")
			since := time.Now().Add(-time.Minute)
			local.statuses = map[string]router.ModelStatus{
				"m1": {State: tt.state, Since: since},
			}
			s := newTestServer(local, newStubRouter(nil, ""))
			s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}}}

			addInflight(t, s, "m1", tt.inFlight)
			// Another model's traffic must not be counted against m1.
			addInflight(t, s, "m2", 3)

			running := getRunning(t, s)
			if len(running) != 1 {
				t.Fatalf("running=%v want 1 entry", running)
			}
			got := running[0]

			if got.State != string(tt.state) {
				t.Errorf("state = %q, want %q", got.State, tt.state)
			}
			switch {
			case tt.wantBusyNull && got.Busy != nil:
				t.Errorf("busy = %v, want null: a %s model is neither busy nor idle", *got.Busy, tt.state)
			case !tt.wantBusyNull && got.Busy == nil:
				t.Fatalf("busy = null, want %v", tt.wantBusy)
			case !tt.wantBusyNull && *got.Busy != tt.wantBusy:
				t.Errorf("busy = %v, want %v", *got.Busy, tt.wantBusy)
			}
			if got.InFlightRequests != tt.inFlight {
				t.Errorf("in_flight_requests = %d, want %d", got.InFlightRequests, tt.inFlight)
			}
			if !got.Since.Equal(since) {
				t.Errorf("since = %v, want the transition time %v", got.Since, since)
			}
			if n := local.serveCalls.Load(); n != 0 {
				t.Errorf("router dispatches = %d, want 0: listing a model must never load it", n)
			}
		})
	}
}

// TestServer_RunningBusyGracePeriod covers the busyGracePeriod extension:
// busy stays true for the configured window after a model's last in-flight
// request completes, even with in_flight_requests back at 0, reverts once
// the window elapses, and a grace period of 0 reproduces the pre-extension
// behavior exactly. State-null precedence and per-model isolation are
// unconditional even with a live grace window.
func TestServer_RunningBusyGracePeriod(t *testing.T) {
	t.Run("holds busy true within the grace window after completion", func(t *testing.T) {
		local := newStubRouter([]string{"m1"}, "")
		local.statuses = map[string]router.ModelStatus{
			"m1": {State: process.StateReady, Since: time.Now()},
		}
		s := newTestServer(local, newStubRouter(nil, ""))
		s.cfg = config.Config{BusyGracePeriod: 1, Models: map[string]config.ModelConfig{"m1": {}}}

		completeInflight(t, s, "m1")

		running := getRunning(t, s)
		if len(running) != 1 {
			t.Fatalf("running=%v want 1 entry", running)
		}
		got := running[0]
		if got.InFlightRequests != 0 {
			t.Errorf("in_flight_requests = %d, want 0", got.InFlightRequests)
		}
		if got.Busy == nil || !*got.Busy {
			t.Fatalf("busy = %v, want true: within the grace window after completion", got.Busy)
		}
	})

	t.Run("reverts to false once the grace window has elapsed", func(t *testing.T) {
		local := newStubRouter([]string{"m1"}, "")
		local.statuses = map[string]router.ModelStatus{
			"m1": {State: process.StateReady, Since: time.Now()},
		}
		s := newTestServer(local, newStubRouter(nil, ""))
		s.cfg = config.Config{BusyGracePeriod: 1, Models: map[string]config.ModelConfig{"m1": {}}}

		completeInflight(t, s, "m1")
		time.Sleep(1100 * time.Millisecond)

		running := getRunning(t, s)
		if len(running) != 1 {
			t.Fatalf("running=%v want 1 entry", running)
		}
		got := running[0]
		if got.Busy == nil || *got.Busy {
			t.Fatalf("busy = %v, want false: the grace window has elapsed", got.Busy)
		}
	})

	t.Run("grace period of 0 reproduces the pre-extension behavior", func(t *testing.T) {
		local := newStubRouter([]string{"m1"}, "")
		local.statuses = map[string]router.ModelStatus{
			"m1": {State: process.StateReady, Since: time.Now()},
		}
		s := newTestServer(local, newStubRouter(nil, ""))
		s.cfg = config.Config{BusyGracePeriod: 0, Models: map[string]config.ModelConfig{"m1": {}}}

		completeInflight(t, s, "m1")

		running := getRunning(t, s)
		if len(running) != 1 {
			t.Fatalf("running=%v want 1 entry", running)
		}
		got := running[0]
		if got.Busy == nil || *got.Busy {
			t.Fatalf("busy = %v, want false: busyGracePeriod=0 disables the extension", got.Busy)
		}
	})

	t.Run("state != ready forces busy null regardless of a live grace window", func(t *testing.T) {
		local := newStubRouter([]string{"m1"}, "")
		local.statuses = map[string]router.ModelStatus{
			"m1": {State: process.StateStopping, Since: time.Now()},
		}
		s := newTestServer(local, newStubRouter(nil, ""))
		s.cfg = config.Config{BusyGracePeriod: 30, Models: map[string]config.ModelConfig{"m1": {}}}

		completeInflight(t, s, "m1")

		running := getRunning(t, s)
		if len(running) != 1 {
			t.Fatalf("running=%v want 1 entry", running)
		}
		got := running[0]
		if got.Busy != nil {
			t.Errorf("busy = %v, want null: state is not ready", *got.Busy)
		}
	})

	t.Run("another model's completion does not extend this model's grace window", func(t *testing.T) {
		local := newStubRouter([]string{"m1", "m2"}, "")
		local.statuses = map[string]router.ModelStatus{
			"m1": {State: process.StateReady, Since: time.Now()},
			"m2": {State: process.StateReady, Since: time.Now()},
		}
		s := newTestServer(local, newStubRouter(nil, ""))
		s.cfg = config.Config{
			BusyGracePeriod: 30,
			Models:          map[string]config.ModelConfig{"m1": {}, "m2": {}},
		}

		completeInflight(t, s, "m2")

		running := getRunning(t, s)
		var m1 *runningModel
		for i := range running {
			if running[i].Model == "m1" {
				m1 = &running[i]
			}
		}
		if m1 == nil {
			t.Fatalf("m1 missing from running=%v", running)
		}
		if m1.Busy == nil || *m1.Busy {
			t.Fatalf("m1 busy = %v, want false: m2's completion must not extend m1's grace window", m1.Busy)
		}
	})
}

// TestServer_RunningExcludesStoppedModels pins the endpoint's existing
// contract: /running lists what the router reports as running and nothing
// else, so a configured but unloaded model stays absent rather than gaining
// an entry from the new fields.
func TestServer_RunningExcludesStoppedModels(t *testing.T) {
	local := newStubRouter([]string{"m1", "m2"}, "")
	local.statuses = map[string]router.ModelStatus{
		"m1": {State: process.StateReady, Since: time.Now()},
	}
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"m1": {}, "m2": {}}}

	running := getRunning(t, s)
	if len(running) != 1 || running[0].Model != "m1" {
		t.Fatalf("running=%+v want only m1", running)
	}
	if n := local.serveCalls.Load(); n != 0 {
		t.Errorf("router dispatches = %d, want 0", n)
	}
}

func TestServer_Preload(t *testing.T) {
	local := newStubRouter([]string{"m1"}, "ok")
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Hooks: config.HooksConfig{
		OnStartup: config.HookOnStartup{Preload: []string{"m1"}},
	}}

	got := make(chan swaputil.ModelPreloadedEvent, 1)
	cancel := event.On(func(e swaputil.ModelPreloadedEvent) { got <- e })
	defer cancel()

	s.startPreload()

	select {
	case e := <-got:
		if e.ModelName != "m1" || !e.Success {
			t.Errorf("event=%+v want {ModelName:m1 Success:true}", e)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("preload event not received")
	}
}

func TestServer_Shutdown_StopsRoutersAndIsIdempotent(t *testing.T) {
	local := newStubRouter([]string{"local-model"}, "")
	peer := newStubRouter(nil, "")
	s := newTestServer(local, peer)

	if err := s.Shutdown(time.Second); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
	if err := s.Shutdown(time.Second); err != nil {
		t.Fatalf("second Shutdown: %v", err)
	}
	if got := local.shutdownCalls.Load(); got != 1 {
		t.Errorf("local shutdownCalls=%d want 1", got)
	}
	if got := peer.shutdownCalls.Load(); got != 1 {
		t.Errorf("peer shutdownCalls=%d want 1", got)
	}
}

func TestServer_LogStream_ModelID(t *testing.T) {
	buf := logmon.NewWriter(io.Discard)
	buf.Write([]byte("hello from model"))

	local := newStubRouter([]string{"mymodel"}, "")
	local.loggers = map[string]*logmon.Monitor{"mymodel": buf}

	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: map[string]config.ModelConfig{"mymodel": {}}}

	// Pre-cancel the context so the streaming loop exits immediately after
	// flushing history.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := httptest.NewRequest(http.MethodGet, "/logs/stream/mymodel", nil).WithContext(ctx)
	w := httptest.NewRecorder()
	s.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status=%d body=%q", w.Code, w.Body.String())
	}
	if got := w.Body.String(); got != "hello from model" {
		t.Errorf("body=%q want %q", got, "hello from model")
	}
}

func TestServer_LogStream_UnknownID_Returns400(t *testing.T) {
	s := newTestServer(newStubRouter(nil, ""), newStubRouter(nil, ""))

	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/logs/stream/no-such-model", nil))

	if w.Code != http.StatusBadRequest {
		t.Errorf("status=%d want 400", w.Code)
	}
}
