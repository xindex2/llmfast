package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"
)

// TestUsageIsScopedToTheAccount is the isolation guarantee the customer
// dashboard rests on. Requests record the API key they arrived on, not the
// account, so every per-account query joins through api_keys. Getting that
// wrong would show one customer another's traffic.
func TestUsageIsScopedToTheAccount(t *testing.T) {
	st, err := Open(filepath.Join(t.TempDir(), "u.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	ctx := context.Background()

	alice, err := st.CreateUser(ctx, "alice@example.com", "correcthorsebattery")
	if err != nil {
		t.Fatal(err)
	}
	bob, err := st.CreateUser(ctx, "bob@example.com", "correcthorsebattery")
	if err != nil {
		t.Fatal(err)
	}
	if alice.Role != RoleUser {
		t.Errorf("role = %q, want user", alice.Role)
	}

	ak, _, err := st.CreateKeyFor(ctx, alice.ID, "alice-key", 0)
	if err != nil {
		t.Fatal(err)
	}
	bk, _, err := st.CreateKeyFor(ctx, bob.ID, "bob-key", 0)
	if err != nil {
		t.Fatal(err)
	}

	now := time.Now()
	write := func(keyID int64, model string, comp int, cost float64) {
		st.Log(Record{TS: now, RequestID: "r", Model: model, APIKeyID: keyID,
			Status: 200, PromptTokens: 10, CompletionTokens: comp, TPS: 50, CostUSD: cost})
	}
	write(ak.ID, "qwen/a", 100, 0.01)
	write(ak.ID, "qwen/a", 100, 0.01)
	write(ak.ID, "qwen/b", 50, 0.02)
	write(bk.ID, "qwen/a", 999, 9.99)
	flush(t, st, 4)

	from, to := now.Add(-time.Hour), now.Add(time.Hour)

	au, err := st.UsageFor(ctx, alice.ID, from, to)
	if err != nil {
		t.Fatal(err)
	}
	if au.Requests != 3 {
		t.Errorf("alice requests = %d, want 3 (bob's must not appear)", au.Requests)
	}
	if au.CompTok != 250 {
		t.Errorf("alice completion tokens = %d, want 250", au.CompTok)
	}

	bu, _ := st.UsageFor(ctx, bob.ID, from, to)
	if bu.Requests != 1 || bu.CompTok != 999 {
		t.Errorf("bob = %+v, want exactly his own single request", bu)
	}

	byModel, err := st.UsageByModelFor(ctx, alice.ID, from, to)
	if err != nil {
		t.Fatal(err)
	}
	if len(byModel) != 2 || byModel[0].Model != "qwen/a" || byModel[0].Requests != 2 {
		t.Errorf("by-model = %+v, want qwen/a first with 2 requests", byModel)
	}

	recent, err := st.RecentFor(ctx, alice.ID, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(recent) != 3 {
		t.Errorf("alice's log has %d rows, want 3", len(recent))
	}

	// Keys are scoped the same way.
	ks, err := st.ListKeysFor(ctx, alice.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(ks) != 1 || ks[0].Name != "alice-key" {
		t.Errorf("alice sees %+v, want only her own key", ks)
	}
	if all, _ := st.ListKeys(ctx); len(all) != 2 {
		t.Errorf("admin sees %d keys, want both", len(all))
	}
}
