package server

import (
	"encoding/json"
	"net/http"
	"sort"
	"time"

	"github.com/mostlygeek/llama-swap/internal/process"
	"github.com/mostlygeek/llama-swap/internal/router"
	"github.com/mostlygeek/llama-swap/internal/swaputil"
)

// The four states GET /api/status reports. They are deliberately a small,
// closed set: clients poll this endpoint to decide whether a model can take
// work, so the vocabulary must stay stable even if the internal process state
// machine grows more states.
const (
	statusStateUnloaded  = "unloaded"
	statusStateStarting  = "starting"
	statusStateReady     = "ready"
	statusStateUnloading = "unloading"
)

// statusEntry is the per-model status object, and is the complete response of
// GET /api/status/{model}. Every field is explicit: clients pin an allow-list
// against this shape and treat an unexpected key as a breaking change, so it
// must never be widened by accident (no maps, no embedded config structs, no
// request or response payload data of any kind).
type statusEntry struct {
	Model string `json:"model"`
	State string `json:"state"`
	// Busy is null unless State is "ready": a model that is not loaded is
	// neither busy nor idle.
	Busy             *bool     `json:"busy"`
	InFlightRequests int       `json:"in_flight_requests"`
	Since            time.Time `json:"since"`
}

// statusResponse is the container returned by GET /api/status. Entries are the
// same objects the single-model route returns, sorted by model ID, so one
// client-side parser covers both routes.
type statusResponse struct {
	Models []statusEntry `json:"models"`
}

// statusState maps an internal process state onto the four states this API
// exposes. StateShutdown maps to "unloaded" like StateStopped: the process is
// not resident either way, and the distinction (it will not be restarted
// because llama-swap itself is shutting down) is not something a polling
// client can act on differently.
func statusState(state process.ProcessState) string {
	switch state {
	case process.StateStarting:
		return statusStateStarting
	case process.StateReady:
		return statusStateReady
	case process.StateStopping:
		return statusStateUnloading
	default: // process.StateStopped, process.StateShutdown
		return statusStateUnloaded
	}
}

// newStatusEntry builds the response object for one model. inFlight comes from
// llama-swap's own in-flight bookkeeping (see inflightTracker); nothing here
// asks the upstream process anything.
func newStatusEntry(modelID string, status router.ModelStatus, inFlight int) statusEntry {
	entry := statusEntry{
		Model:            modelID,
		State:            statusState(status.State),
		InFlightRequests: inFlight,
		Since:            status.Since.UTC(),
	}
	if entry.State == statusStateReady {
		busy := inFlight > 0
		entry.Busy = &busy
	}
	return entry
}

// handleAPIStatus reports the status of every configured local model in one
// response, so polling a full fleet costs one round trip.
//
// Like handleAPIStatusModel it is strictly a read of already-published state:
// no model is loaded, unloaded, health-checked, or otherwise touched.
func (s *Server) handleAPIStatus(w http.ResponseWriter, r *http.Request) {
	ids := make([]string, 0, len(s.cfg.Models))
	for id := range s.cfg.Models {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	inFlight := s.inflight.CountByModel()
	resp := statusResponse{Models: make([]statusEntry, 0, len(ids))}
	for _, id := range ids {
		status, ok := s.local.ModelStatus(id)
		if !ok {
			// Configured but not backed by a local process; there is no
			// lifecycle state to report.
			continue
		}
		resp.Models = append(resp.Models, newStatusEntry(id, status, inFlight[id]))
	}

	s.writeStatusJSON(w, resp)
}

// handleAPIStatusModel reports the status of a single model. The model may be
// named by its config key or by one of its aliases; the response always names
// the real model ID so a client can line it up with the GET /api/status
// listing. Models with no local process (peer models, selectors, profile pins)
// have no lifecycle state and return 404.
//
// The handler never triggers a state transition. It reads the process's own
// published state snapshot and the in-flight counters, both of which are
// non-blocking reads, so the response time does not depend on what any model is
// currently doing.
func (s *Server) handleAPIStatusModel(w http.ResponseWriter, r *http.Request) {
	modelID, found := s.cfg.RealModelName(r.PathValue("model"))
	if !found {
		sendStatusNotFound(w, r)
		return
	}
	status, ok := s.local.ModelStatus(modelID)
	if !ok {
		sendStatusNotFound(w, r)
		return
	}

	s.writeStatusJSON(w, newStatusEntry(modelID, status, s.inflight.CountByModel()[modelID]))
}

// sendStatusNotFound answers with a fixed message. The requested key is never
// echoed: this endpoint returns status metadata only, and reflecting caller
// input would be the start of a payload-carrying error body.
func sendStatusNotFound(w http.ResponseWriter, r *http.Request) {
	swaputil.SendResponse(w, r, http.StatusNotFound, "model not found")
}

func (s *Server) writeStatusJSON(w http.ResponseWriter, payload any) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		// Encoding a fixed struct only fails when the client has gone away,
		// which is routine for a polling client with a short timeout.
		s.proxylog.Debugf("status: writing response failed: %v", err)
	}
}
