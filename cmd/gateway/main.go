// Command gateway runs the LLMFast inference edge.
//
// Two listeners are started: a public one carrying the OpenAI-compatible API
// that OpenRouter calls, and a private one carrying the admin UI and its API.
// Keep the admin listener bound to localhost or a private interface -- it
// exposes API keys and the full request history.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/llmfast/gateway/internal/config"
	"github.com/llmfast/gateway/internal/gateway"
	"github.com/llmfast/gateway/internal/store"
	"github.com/llmfast/gateway/internal/upstream"
)

func main() {
	configPath := flag.String("config", "config/config.yaml", "path to config file")
	addAdmin := flag.String("add-admin", "", "create or reset a dashboard account for this email, then exit")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error("load config", "err", err)
		os.Exit(1)
	}

	st, err := store.Open(cfg.Server.DBPath)
	if err != nil {
		log.Error("open database", "err", err)
		os.Exit(1)
	}
	defer st.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if *addAdmin != "" {
		if err := manageAdmin(ctx, st, *addAdmin); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			os.Exit(1)
		}
		return
	}

	bootstrapKey(ctx, st, log)

	pool := upstream.NewPool(cfg)
	pool.StartHealthChecks(10 * time.Second)
	defer pool.Stop()

	srv := gateway.New(cfg, st, pool, log)
	srv.SetConfigPath(*configPath)
	srv.StartBackground(ctx)

	// Poll node agents for hardware and running engines. Engines they report as
	// ready become routable backends automatically, so installing a model needs
	// no gateway restart.
	if n := srv.Nodes().Count(); n > 0 {
		srv.Nodes().Start(ctx, 10*time.Second)
		log.Info("polling node agents", "nodes", n)
	}

	public := &http.Server{
		Addr:    cfg.Server.Listen,
		Handler: srv.PublicHandler(),
		// ReadHeaderTimeout guards against slowloris. There is deliberately no
		// WriteTimeout: it would kill long streams mid-generation, and per
		// request deadlines are already applied through the request context.
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       cfg.Server.ReadTimeout,
		IdleTimeout:       120 * time.Second,
	}
	admin := &http.Server{
		Addr:              cfg.Server.AdminListen,
		Handler:           srv.AdminHandler(),
		ReadHeaderTimeout: 10 * time.Second,
		// ReadTimeout is generous rather than absent because the playground
		// posts prompts that can be large. There is deliberately no
		// WriteTimeout: the playground streams completions, and a deadline
		// here would truncate a long generation mid-answer.
		ReadTimeout: 5 * time.Minute,
		IdleTimeout: 120 * time.Second,
	}

	// Only the public listener is fatal. The admin UI failing to bind is
	// annoying; taking the inference API down with it would turn a port clash
	// into an outage, and OpenRouter scores those against us. So the admin
	// error is reported loudly and the gateway keeps serving traffic.
	errCh := make(chan error, 1)
	go func() {
		log.Info("public API listening", "addr", cfg.Server.Listen,
			"models", len(cfg.Models), "backends", len(cfg.Backends))
		if err := public.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- fmt.Errorf("public listener on %s: %w", cfg.Server.Listen, err)
		}
	}()
	go func() {
		log.Info("admin UI listening", "addr", cfg.Server.AdminListen)
		if err := admin.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("admin UI unavailable; the inference API is unaffected",
				"addr", cfg.Server.AdminListen, "err", err,
				"fix", "change server.admin_listen in your config to a free port, then restart")
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)

	for {
		select {
		case err := <-errCh:
			log.Error("listener failed", "err", err)
			shutdown(public, admin, log)
			return

		case s := <-sig:
			// SIGHUP reloads models and pricing in place. Backends are not
			// re-read: swapping them means rebuilding connection pools and
			// admission counters underneath live streams, so that needs a
			// restart.
			if s == syscall.SIGHUP {
				if err := srv.ReloadFromDisk(); err != nil {
					log.Error("reload failed, keeping current config", "err", err)
				}
				continue
			}
			log.Info("shutting down", "signal", s.String())
			cancel()
			shutdown(public, admin, log)
			return
		}
	}
}

// shutdown drains both listeners. The grace period is long enough for an
// in-flight generation to finish rather than being cut off mid-stream.
func shutdown(public, admin *http.Server, log *slog.Logger) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := public.Shutdown(ctx); err != nil {
		log.Warn("public shutdown", "err", err)
	}
	if err := admin.Shutdown(ctx); err != nil {
		log.Warn("admin shutdown", "err", err)
	}
}

// bootstrapKey mints a first API key on an empty database and prints it once,
// so a fresh install is usable without reaching into SQLite by hand.
func bootstrapKey(ctx context.Context, st *store.Store, log *slog.Logger) {
	n, err := st.CountKeys(ctx)
	if err != nil {
		log.Error("count api keys", "err", err)
		return
	}
	if n > 0 {
		return
	}
	key, secret, err := st.CreateKey(ctx, "bootstrap", 0)
	if err != nil {
		log.Error("create bootstrap key", "err", err)
		return
	}
	fmt.Printf("\n  No API keys found. Created one for you:\n\n    %s\n\n"+
		"  This is shown once and only its hash is stored. Save it now.\n"+
		"  Manage keys at the admin UI (key id %d).\n\n", secret, key.ID)
}

// manageAdmin creates a dashboard account, or resets the password of one that
// already exists.
//
// The password is read from the terminal rather than taken as a flag, because
// a flag would put it in the shell history and in the process list, where
// anyone else on the machine can read it.
func manageAdmin(ctx context.Context, st *store.Store, email string) error {
	fmt.Printf("Password for %s: ", email)
	pw, err := readPassword()
	if err != nil {
		return err
	}
	fmt.Print("Repeat it: ")
	again, err := readPassword()
	if err != nil {
		return err
	}
	if pw != again {
		return errors.New("the two passwords do not match")
	}

	_, err = st.CreateAdminUser(ctx, email, pw)
	switch {
	case err == nil:
		fmt.Printf("\nCreated %s. Sign in at the admin URL with that email and password.\n", email)
		return nil
	case errors.Is(err, store.ErrUserExists):
		if err := st.SetAdminPassword(ctx, email, pw); err != nil {
			return err
		}
		fmt.Printf("\nPassword reset for %s. Any existing sessions were signed out.\n", email)
		return nil
	default:
		return err
	}
}

// stdin is read through one shared buffered reader. A fresh bufio.Reader per
// call would buffer everything available and then hand the next call EOF,
// which broke the confirmation prompt whenever input was piped.
var stdin = bufio.NewReader(os.Stdin)

// readPassword reads a line without echoing it, falling back to a plain read
// when stdin is not a terminal -- which is what makes the command scriptable
// as `echo "$PW" | llmfast -add-admin you@example.com`.
func readPassword() (string, error) {
	fd := int(os.Stdin.Fd())
	restore, err := disableEcho(fd)
	if err == nil {
		defer restore()
	}
	line, err := stdin.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	if restore != nil {
		fmt.Println()
	}
	return strings.TrimRight(line, "\r\n"), nil
}
