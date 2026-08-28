package gateway

import (
	"context"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/llmfast/gateway/internal/agent"
	"github.com/llmfast/gateway/internal/config"
	"github.com/llmfast/gateway/internal/modelspec"
)

// InspectResult is everything the admin UI needs to offer a one-click install:
// what the model is, whether it fits on each node, and what to charge for it.
type InspectResult struct {
	Info           *modelspec.Info           `json:"info"`
	SuggestedID    string                    `json:"suggested_id"`
	SuggestedName  string                    `json:"suggested_name"`
	Pricing        map[string]string         `json:"pricing"`
	Plans          []NodePlan                `json:"plans"`
	GGUFCandidates []modelspec.GGUFCandidate `json:"gguf_candidates,omitempty"`
	GGUFError      string                    `json:"gguf_error,omitempty"`

	// QuantCandidates is populated when the model does not fit at its native
	// precision on some node. Quantization cannot be applied at launch, so the
	// only way forward is a different repository, and the operator should not
	// have to go and find one.
	QuantCandidates  []modelspec.QuantCandidate `json:"quant_candidates,omitempty"`
	QuantError       string                     `json:"quant_error,omitempty"`
	AlreadyInstalled bool                       `json:"already_installed"`
}

// NodePlan pairs a node with the plan for running this model on it.
type NodePlan struct {
	Node      string          `json:"node"`
	Reachable bool            `json:"reachable"`
	Reason    string          `json:"reason,omitempty"`
	Hardware  *modelspec.Node `json:"hardware,omitempty"`
	Engines   []string        `json:"engines_available,omitempty"`
	Plan      *modelspec.Plan `json:"plan,omitempty"`
}

func (s *Server) adminInspect(w http.ResponseWriter, r *http.Request) {
	var body struct {
		HFID    string `json:"hf_id"`
		Context int    `json:"context"`
	}
	_ = decodeJSON(r, &body)
	body.HFID = strings.TrimSpace(body.HFID)
	if body.HFID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error",
			"hf_id is required, for example \"Qwen/Qwen3-8B\"")
		return
	}

	// HuggingFace lookups are two network round trips; the browser is waiting.
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	info, err := s.hf.Fetch(ctx, body.HFID)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	prompt, completion, cached := modelspec.SuggestPricing(info)
	suggestedID := SuggestModelID(info.ID)
	_, installed := s.catalog.Get(suggestedID)

	res := InspectResult{
		Info:          info,
		SuggestedID:   suggestedID,
		SuggestedName: SuggestDisplayName(info.ID),
		Pricing: map[string]string{
			"prompt": prompt, "completion": completion, "cached_prompt": cached,
		},
		AlreadyInstalled: installed,
	}

	// Plan against every configured node so the operator can see at a glance
	// where this model can actually run.
	needGGUF, needQuant := false, false
	runnable := map[string]bool{}
	for _, st := range s.nodes.Statuses() {
		np := NodePlan{Node: st.Name, Reachable: st.Reachable}
		if !st.Reachable || st.Info == nil {
			np.Reason = st.LastError
			if np.Reason == "" {
				np.Reason = "not yet contacted"
			}
			res.Plans = append(res.Plans, np)
			continue
		}
		hw := st.Info.Node
		np.Hardware = &hw
		np.Engines = st.Info.EnginesAvailable
		np.Plan = modelspec.PlanFor(info, &hw, body.Context)
		if np.Plan.Engine == "llamacpp" && !info.Local {
			needGGUF = true
		}
		if np.Plan.NeedsQuantized {
			needQuant = true
			for _, q := range np.Plan.RunnableQuants {
				runnable[q] = true
			}
		}
		// A plan is useless if the engine it needs is not installed there.
		if !contains(st.Info.EnginesAvailable, np.Plan.Engine) {
			np.Plan.Fits, np.Plan.Viable = false, false
			np.Plan.Blockers = append(np.Plan.Blockers,
				np.Plan.Engine+" is not installed on this node")
		}
		// Nor is it any use if the engine has never heard of this architecture.
		// The checkpoint names it in config.json; whether vLLM implements it
		// depends on the exact release, and a version too old to know it fails
		// after the weights have downloaded, with a traceback whose real cause
		// is dozens of frames above the line that gets reported.
		//
		// An empty list means the question could not be asked, not that the
		// answer is no, so it is only acted on when the node answered.
		if np.Plan.Engine == "vllm" && len(st.Info.SupportedArchs) > 0 &&
			info.Architecture != "" && !contains(st.Info.SupportedArchs, info.Architecture) {
			np.Plan.Fits, np.Plan.Viable = false, false
			np.Plan.Blockers = append(np.Plan.Blockers, fmt.Sprintf(
				"the vLLM installed on this node does not implement %s. That is a version "+
					"question, not a hardware one: upgrade vLLM to a release that supports it, "+
					"keeping torch within what this host's driver can run",
				info.Architecture))
		}
		res.Plans = append(res.Plans, np)
	}

	// llama.cpp cannot read the original safetensors weights, so a CPU node
	// needs a converted GGUF repository. Resolving it here means the operator
	// never has to go and find one.
	if needGGUF {
		if cands, err := s.hf.ResolveGGUF(ctx, info.ID); err != nil {
			res.GGUFError = err.Error()
		} else {
			res.GGUFCandidates = cands
		}
	}

	// Same idea for GPU nodes where the model overflows VRAM at full precision.
	if needQuant {
		if cands, err := s.hf.ResolveQuantized(ctx, info.ID); err != nil {
			res.QuantError = err.Error()
		} else {
			// Only formats some node can actually execute. An NVFP4 repository
			// is the top search result for many models and no Ampere card can
			// run one, so offering it would just cost the operator a round trip.
			for _, c := range cands {
				if runnable[c.Quant] {
					res.QuantCandidates = append(res.QuantCandidates, c)
				}
			}
			if len(res.QuantCandidates) == 0 {
				res.QuantError = "found pre-quantized publications of this model, but none in a " +
					"format your GPUs can run"
			}
		}
	}

	writeJSON(w, http.StatusOK, res)
}

// InstallRequest is the confirmed plan the operator chose.
type InstallRequest struct {
	Node    string `json:"node"`
	HFID    string `json:"hf_id"`
	ModelID string `json:"model_id"`
	Name    string `json:"name"`

	Engine         string `json:"engine"`
	Quantization   string `json:"quantization"`
	Hybrid         bool   `json:"hybrid"`
	QuantFromCkpt  bool   `json:"quant_from_checkpoint"`
	KVCacheDType   string `json:"kv_cache_dtype"`
	TensorParallel int    `json:"tensor_parallel"`
	MaxModelLen    int    `json:"max_model_len"`
	MaxNumSeqs     int    `json:"max_num_seqs"`
	GGUFRepo       string `json:"gguf_repo,omitempty"`
	LocalGGUF      string `json:"local_gguf,omitempty"`

	PromptUSD     string `json:"prompt_usd"`
	CompletionUSD string `json:"completion_usd"`
	CachedUSD     string `json:"cached_usd"`

	Tools     bool `json:"tools"`
	Reasoning bool `json:"reasoning"`
	// ZDR publishes compliance.zdr on the model. It must match what your
	// privacy policy says: declaring zero retention while logging prompts is a
	// misrepresentation, and declaring false while operating ZDR costs you
	// enterprise customers who filter on it.
	ZDR   bool `json:"zdr"`
	HIPAA bool `json:"hipaa"`
	// StageHidden writes is_ready:false so OpenRouter keeps the model hidden
	// until it has been verified.
	StageHidden bool `json:"stage_hidden"`
}

func (s *Server) adminInstall(w http.ResponseWriter, r *http.Request) {
	var req InstallRequest
	if err := decodeJSON(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid request body")
		return
	}
	if !s.nodes.Has(req.Node) {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "unknown node "+req.Node)
		return
	}
	if req.ModelID == "" {
		req.ModelID = SuggestModelID(req.HFID)
	}
	if req.MaxModelLen <= 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "max_model_len is required")
		return
	}
	if s.modelDir == "" {
		writeError(w, http.StatusPreconditionFailed, "config_error",
			"server.model_dir is not set, so models cannot be installed from the admin UI")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	// Whether the engine can take a flag is a property of the model, not a
	// preference the browser gets to express. Reading it from the checkpoint
	// here rather than trusting the request means a stale admin page -- one
	// cached before these fields existed -- cannot launch an engine with flags
	// that make it exit, which is exactly what happened: the plan card showed
	// the model was hybrid while the install it produced still asked for
	// prefix caching.
	if info, err := s.hf.Fetch(ctx, req.HFID); err == nil {
		if info.LocalGGUF != "" && req.LocalGGUF == "" {
			req.LocalGGUF = info.LocalGGUF
		}
		req.Hybrid = info.IsHybrid
		req.QuantFromCkpt = info.PublishedQuant != ""
		if info.IsHybrid {
			// Neither has an implementation over a recurrent state, and vLLM
			// rejects both at startup rather than ignoring them.
			req.KVCacheDType = "auto"
		}
	} else {
		s.log.Warn("could not re-read model config before install; "+
			"relying on the values the admin page sent", "model", req.HFID, "err", err)
	}

	// Start the engine first. If the node rejects it there is nothing to undo,
	// whereas writing the catalog entry first would advertise a model that does
	// not exist and earn us 404s, which count against our uptime.
	spec := agent.Spec{
		HFID: req.HFID, ServedName: req.ModelID, Engine: req.Engine,
		Quantization: req.Quantization, KVCacheDType: req.KVCacheDType,
		Hybrid: req.Hybrid, QuantFromCheckpoint: req.QuantFromCkpt,
		TensorParallel: req.TensorParallel,
		MaxModelLen:    req.MaxModelLen, MaxNumSeqs: req.MaxNumSeqs,
		GGUFRepo: req.GGUFRepo, LocalGGUF: req.LocalGGUF,
	}
	out, err := s.nodes.Install(ctx, req.Node, spec)
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", err.Error())
		return
	}

	// The catalog entry is staged hidden by default. The engine is still
	// downloading weights at this point, so advertising it immediately would
	// route real traffic at a model that cannot answer yet.
	ready := false
	m := config.Model{
		ID:              req.ModelID,
		Name:            orDefault(req.Name, SuggestDisplayName(req.HFID)),
		UpstreamModel:   req.ModelID, // the engine was told to serve under this name
		Backends:        []string{req.Node},
		HuggingFaceID:   req.HFID,
		Quantization:    normalizeQuant(req.Quantization),
		ContextLength:   req.MaxModelLen,
		MaxOutputTokens: minPositive(req.MaxModelLen, 32768),
		Pricing: config.Pricing{
			Prompt: req.PromptUSD, Completion: req.CompletionUSD, CachedPrompt: req.CachedUSD,
		},
		Capacity:   config.Capacity{Concurrency: req.MaxNumSeqs},
		Features:   config.Features{Tools: req.Tools, Reasoning: req.Reasoning},
		Compliance: config.Compliance{ZDR: req.ZDR, HIPAA: req.HIPAA},
		IsReady:    &ready,
	}
	path, err := config.WriteModel(s.modelDir, m)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error",
			"the engine started but the catalog entry could not be written: "+err.Error())
		return
	}
	if err := s.ReloadFromDisk(); err != nil {
		s.log.Error("reload after install", "err", err)
	}

	writeJSON(w, http.StatusAccepted, map[string]any{
		"model_id":    req.ModelID,
		"node":        req.Node,
		"config_file": path,
		"agent":       out,
		"note": "The engine is downloading weights and starting. The model is staged hidden " +
			"(is_ready: false) so OpenRouter will not route to it. Publish it once it reports ready.",
	})
}

// adminPublish flips a staged model to visible once it is confirmed working.
func (s *Server) adminPublish(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ModelID string `json:"model_id"`
		Ready   bool   `json:"ready"`
	}
	if err := decodeJSON(r, &body); err != nil || body.ModelID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model_id is required")
		return
	}
	m, ok := s.cfg.Model(body.ModelID)
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "unknown model "+body.ModelID)
		return
	}
	updated := *m
	ready := body.Ready
	updated.IsReady = &ready
	if _, err := config.WriteModel(s.modelDir, updated); err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	if err := s.ReloadFromDisk(); err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"model_id": body.ModelID, "ready": ready})
}

// adminUninstall stops the engine and removes the catalog entry.
func (s *Server) adminUninstall(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ModelID string `json:"model_id"`
		Node    string `json:"node"`
	}
	if err := decodeJSON(r, &body); err != nil || body.ModelID == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model_id is required")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	// Remove from the catalog first so no new traffic is routed at a model we
	// are about to stop.
	if err := config.RemoveModel(s.modelDir, body.ModelID); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if err := s.ReloadFromDisk(); err != nil {
		s.log.Error("reload after uninstall", "err", err)
	}
	if body.Node != "" {
		if err := s.nodes.RemoveModel(ctx, body.Node, body.ModelID); err != nil {
			// The catalog entry is already gone, so this is not fatal; the
			// operator just needs to know the engine is still holding memory.
			writeJSON(w, http.StatusOK, map[string]any{
				"model_id": body.ModelID, "removed_from_catalog": true,
				"warning": "the engine could not be stopped: " + err.Error(),
			})
			return
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"model_id": body.ModelID, "removed": true})
}

func (s *Server) adminNodes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"nodes": s.nodes.Statuses()})
}

func (s *Server) adminNodeLogs(w http.ResponseWriter, r *http.Request) {
	node := r.PathValue("node")
	served := r.URL.Query().Get("served_name")
	n, _ := strconv.Atoi(r.URL.Query().Get("n"))
	if n <= 0 {
		n = 150
	}
	ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
	defer cancel()

	out, err := s.nodes.Logs(ctx, node, served, n)
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) adminNodeStop(w http.ResponseWriter, r *http.Request) {
	node := r.PathValue("node")
	var body struct {
		ServedName string `json:"served_name"`
	}
	if err := decodeJSON(r, &body); err != nil || body.ServedName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "served_name is required")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	if err := s.nodes.StopModel(ctx, node, body.ServedName); err != nil {
		writeError(w, http.StatusBadGateway, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]bool{"ok": true})
}

// SuggestModelID converts a HuggingFace id into the lowercase owner/name form
// OpenRouter uses for model slugs.
func SuggestModelID(hfID string) string {
	id := strings.TrimSpace(hfID)
	// A checkpoint loaded from disk has a filesystem path where a repository id
	// would be. Publishing that as a model id would leak the directory layout
	// of the machine and read as nonsense to anyone calling the API, so only
	// the last path element is used.
	if modelspec.IsLocalPath(id) {
		return "local/" + strings.ToLower(filepath.Base(strings.TrimRight(id, "/")))
	}
	return strings.ToLower(id)
}

// SuggestDisplayName renders "Qwen/Qwen3-32B" as "Qwen: Qwen3 32B", matching
// how models are labelled on OpenRouter.
func SuggestDisplayName(hfID string) string {
	if modelspec.IsLocalPath(hfID) {
		return filepath.Base(strings.TrimRight(hfID, "/"))
	}
	owner, name, ok := strings.Cut(hfID, "/")
	if !ok {
		return hfID
	}
	return fmt.Sprintf("%s: %s", owner, strings.ReplaceAll(name, "-", " "))
}

// normalizeQuant maps engine-specific quantization names onto the vocabulary
// OpenRouter's schema accepts. A GGUF Q4_K_M is a 4-bit format, and declaring
// it as "q4_k_m" would fail their validation.
func normalizeQuant(q string) string {
	switch strings.ToLower(q) {
	case "fp8":
		return "fp8"
	case "awq", "gptq", "int4", "q4_k_m", "q4_0", "iq4_xs":
		return "int4"
	case "q8_0", "int8":
		return "int8"
	case "nvfp4", "fp4", "mxfp4":
		return "fp4"
	case "bf16", "fp16", "float16", "bfloat16":
		return "bf16"
	case "fp32", "float32":
		return "fp32"
	case "compressed-tensors":
		// The wrapper does not name the scheme inside it without reading the
		// per-layer config, and guessing would publish a claim about our own
		// serving precision that we cannot stand behind.
		return "unknown"
	}
	return ""
}

func orDefault(v, def string) string {
	if strings.TrimSpace(v) == "" {
		return def
	}
	return v
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}

// adminNodeCache lists the weights a node has downloaded.
func (s *Server) adminNodeCache(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()
	out, err := s.nodes.Cache(ctx, r.PathValue("node"))
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", err.Error())
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// adminNodeCacheDelete reclaims the disk one model's weights are using.
func (s *Server) adminNodeCacheDelete(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Repo string `json:"repo"`
	}
	_ = decodeJSON(r, &body)
	ctx, cancel := context.WithTimeout(r.Context(), 120*time.Second)
	defer cancel()
	out, err := s.nodes.DeleteCache(ctx, r.PathValue("node"), body.Repo)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	s.log.Info("deleted cached weights", "node", r.PathValue("node"), "repo", body.Repo)
	writeJSON(w, http.StatusOK, out)
}
