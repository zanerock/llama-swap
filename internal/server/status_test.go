package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/internal/config"
	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// statusFields is the complete, closed field set of a status object. Clients
// pin an allow-list against it, so an added or renamed key is a breaking
// change and this list is what makes that break visible.
var statusFields = []string{"busy", "in_flight_requests", "model", "since", "state"}

// newStatusServer builds a Server whose local router reports the given model
// statuses and whose config declares the same models.
func newStatusServer(statuses map[string]router.ModelStatus) (*Server, *stubRouter) {
	local := newStubRouter(nil, "")
	local.statuses = statuses

	models := make(map[string]config.ModelConfig, len(statuses))
	for id := range statuses {
		models[id] = config.ModelConfig{}
	}

	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = config.Config{Models: models}
	return s, local
}

// addInflight registers n in-flight requests dispatched to modelID, the way
// the in-flight middleware does around a proxied request.
func addInflight(s *Server, modelID string, n int) {
	for i := 0; i < n; i++ {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		req = req.WithContext(swaputil.SetContext(req.Context(), swaputil.ReqContextData{
			Model:    modelID,
			ModelID:  modelID,
			Metadata: map[string]string{},
		}))
		s.inflight.Add(req, func() {})
	}
}

func getStatus(t *testing.T, s *Server, path string) (int, []byte) {
	t.Helper()
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w.Code, w.Body.Bytes()
}

// decodeStatusObject decodes one status object without losing unexpected keys.
func decodeStatusObject(t *testing.T, body []byte) map[string]json.RawMessage {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		t.Fatalf("decode status object: %v (body=%q)", err, body)
	}
	return obj
}

func assertStatusFields(t *testing.T, obj map[string]json.RawMessage) {
	t.Helper()
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if strings.Join(keys, ",") != strings.Join(statusFields, ",") {
		t.Errorf("fields = %v, want exactly %v", keys, statusFields)
	}
}

func TestServer_APIStatus_SingleModelShape(t *testing.T) {
	since := time.Date(2026, 8, 13, 21, 13, 1, 0, time.UTC)
	s, _ := newStatusServer(map[string]router.ModelStatus{
		"m1": {State: process.StateReady, Since: since},
	})
	addInflight(s, "m1", 2)

	code, body := getStatus(t, s, "/api/status/m1")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", code, body)
	}

	obj := decodeStatusObject(t, body)
	assertStatusFields(t, obj)

	var entry statusEntry
	if err := json.Unmarshal(body, &entry); err != nil {
		t.Fatalf("decode entry: %v", err)
	}
	if entry.Model != "m1" {
		t.Errorf("model = %q, want m1", entry.Model)
	}
	if entry.State != statusStateReady {
		t.Errorf("state = %q, want %q", entry.State, statusStateReady)
	}
	if entry.Busy == nil || !*entry.Busy {
		t.Errorf("busy = %v, want true", entry.Busy)
	}
	if entry.InFlightRequests != 2 {
		t.Errorf("in_flight_requests = %d, want 2", entry.InFlightRequests)
	}
	if !entry.Since.Equal(since) {
		t.Errorf("since = %v, want %v", entry.Since, since)
	}
	if string(obj["since"]) != `"2026-08-13T21:13:01Z"` {
		t.Errorf("since encoding = %s, want an ISO-8601 UTC timestamp", obj["since"])
	}
}

func TestServer_APIStatus_AllModels(t *testing.T) {
	s, _ := newStatusServer(map[string]router.ModelStatus{
		"m2": {State: process.StateReady, Since: time.Now()},
		"m1": {State: process.StateStopped, Since: time.Now()},
	})
	// A peer model is configured but has no local process, so it has no
	// lifecycle state to report and must not appear.
	s.cfg.Peers = config.PeerDictionaryConfig{"peer1": {Models: []string{"remote"}}}

	code, body := getStatus(t, s, "/api/status")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", code, body)
	}

	var container map[string]json.RawMessage
	if err := json.Unmarshal(body, &container); err != nil {
		t.Fatalf("decode container: %v", err)
	}
	if len(container) != 1 || container["models"] == nil {
		t.Fatalf("container keys = %v, want exactly [models]", container)
	}

	var raw []map[string]json.RawMessage
	if err := json.Unmarshal(container["models"], &raw); err != nil {
		t.Fatalf("decode models: %v", err)
	}
	if len(raw) != 2 {
		t.Fatalf("models = %d entries, want 2 (body=%q)", len(raw), body)
	}
	for _, obj := range raw {
		assertStatusFields(t, obj)
	}

	var resp statusResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Models[0].Model != "m1" || resp.Models[1].Model != "m2" {
		t.Errorf("models = %q, %q, want m1, m2 in sorted order", resp.Models[0].Model, resp.Models[1].Model)
	}
	if resp.Models[0].State != statusStateUnloaded || resp.Models[0].Busy != nil {
		t.Errorf("m1 = %+v, want state unloaded and busy null", resp.Models[0])
	}
	if resp.Models[1].State != statusStateReady || resp.Models[1].Busy == nil || *resp.Models[1].Busy {
		t.Errorf("m2 = %+v, want state ready and busy false", resp.Models[1])
	}
}

// TestServer_APIStatus_StateMapping pins the internal-state to API-state
// mapping, including that busy is null for every state except ready.
func TestServer_APIStatus_StateMapping(t *testing.T) {
	tests := []struct {
		name      string
		state     process.ProcessState
		wantState string
		wantBusy  bool // whether busy is non-null
	}{
		{"stopped is unloaded", process.StateStopped, statusStateUnloaded, false},
		{"shutdown is unloaded", process.StateShutdown, statusStateUnloaded, false},
		{"starting", process.StateStarting, statusStateStarting, false},
		{"ready", process.StateReady, statusStateReady, true},
		{"stopping is unloading", process.StateStopping, statusStateUnloading, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			s, _ := newStatusServer(map[string]router.ModelStatus{
				"m1": {State: tc.state, Since: time.Now()},
			})
			// In-flight requests exist in every state: a request queued behind
			// a load is counted, but busy stays null until the model is ready.
			addInflight(s, "m1", 1)

			code, body := getStatus(t, s, "/api/status/m1")
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200", code)
			}
			var entry statusEntry
			if err := json.Unmarshal(body, &entry); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if entry.State != tc.wantState {
				t.Errorf("state = %q, want %q", entry.State, tc.wantState)
			}
			if (entry.Busy != nil) != tc.wantBusy {
				t.Errorf("busy = %v, want non-null=%v", entry.Busy, tc.wantBusy)
			}
			if entry.InFlightRequests != 1 {
				t.Errorf("in_flight_requests = %d, want 1", entry.InFlightRequests)
			}
		})
	}
}

func TestServer_APIStatus_BusyFollowsInflightCount(t *testing.T) {
	s, _ := newStatusServer(map[string]router.ModelStatus{
		"idle": {State: process.StateReady, Since: time.Now()},
		"busy": {State: process.StateReady, Since: time.Now()},
	})
	addInflight(s, "busy", 3)

	code, body := getStatus(t, s, "/api/status")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200", code)
	}
	var resp statusResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	byModel := map[string]statusEntry{}
	for _, entry := range resp.Models {
		byModel[entry.Model] = entry
	}

	if got := byModel["busy"]; got.Busy == nil || !*got.Busy || got.InFlightRequests != 3 {
		t.Errorf("busy model = %+v, want busy=true in_flight_requests=3", got)
	}
	if got := byModel["idle"]; got.Busy == nil || *got.Busy || got.InFlightRequests != 0 {
		t.Errorf("idle model = %+v, want busy=false in_flight_requests=0", got)
	}
}

func TestServer_APIStatus_UnknownModelIs404(t *testing.T) {
	s, local := newStatusServer(map[string]router.ModelStatus{
		"m1": {State: process.StateStopped, Since: time.Now()},
	})
	// A peer model: known to the config, but with no local process state.
	s.cfg.Peers = config.PeerDictionaryConfig{"peer1": {Models: []string{"remote"}}}

	for _, path := range []string{"/api/status/nope", "/api/status/peer1/remote", "/api/status/"} {
		code, body := getStatus(t, s, path)
		if code != http.StatusNotFound {
			t.Errorf("GET %s = %d, want 404 (body=%q)", path, code, body)
		}
		if strings.Contains(string(body), "nope") || strings.Contains(string(body), "remote") {
			t.Errorf("GET %s body echoes the requested key: %q", path, body)
		}
	}
	if got := local.serveCalls.Load(); got != 0 {
		t.Errorf("router dispatches = %d, want 0", got)
	}
}

func TestServer_APIStatus_ResolvesAliases(t *testing.T) {
	cfg, err := config.LoadConfigFromReader(strings.NewReader(`
models:
  real:
    cmd: echo ${PORT}
    aliases: [nick]
`))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}

	local := newStubRouter(nil, "")
	local.statuses = map[string]router.ModelStatus{
		"real": {State: process.StateReady, Since: time.Now()},
	}
	s := newTestServer(local, newStubRouter(nil, ""))
	s.cfg = cfg

	code, body := getStatus(t, s, "/api/status/nick")
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body=%q)", code, body)
	}
	var entry statusEntry
	if err := json.Unmarshal(body, &entry); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// The real model ID is reported so the response lines up with the
	// GET /api/status listing.
	if entry.Model != "real" {
		t.Errorf("model = %q, want real", entry.Model)
	}
}

// TestServer_APIStatus_HasNoSideEffects is the regression test for the defect
// this endpoint exists to avoid: reading status must never load, unload, or
// otherwise touch a model. Dispatching into the router is the only way a
// request can start a model, and Unload is the only way it can stop one.
func TestServer_APIStatus_HasNoSideEffects(t *testing.T) {
	before := router.ModelStatus{State: process.StateStopped, Since: time.Now()}
	s, local := newStatusServer(map[string]router.ModelStatus{"m1": before})

	for i := 0; i < 20; i++ {
		if code, body := getStatus(t, s, "/api/status"); code != http.StatusOK {
			t.Fatalf("GET /api/status = %d (body=%q)", code, body)
		}
		if code, body := getStatus(t, s, "/api/status/m1"); code != http.StatusOK {
			t.Fatalf("GET /api/status/m1 = %d (body=%q)", code, body)
		}
	}

	if got := local.serveCalls.Load(); got != 0 {
		t.Errorf("router dispatches = %d, want 0: a dispatch can trigger a model load", got)
	}
	if got := local.unloadCalls.Load(); got != 0 {
		t.Errorf("unload calls = %d, want 0", got)
	}
	if after, _ := local.ModelStatus("m1"); after != before {
		t.Errorf("model status changed from %+v to %+v", before, after)
	}
}
