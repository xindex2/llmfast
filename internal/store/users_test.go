package store

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
)

func testStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestPasswordHashingIsSaltedAndVerifiable(t *testing.T) {
	a, err := hashPassword("correcthorsebattery")
	if err != nil {
		t.Fatal(err)
	}
	b, _ := hashPassword("correcthorsebattery")
	if a == b {
		t.Error("the same password hashed twice produced the same digest, so it is unsalted")
	}
	if strings.Contains(a, "correcthorsebattery") {
		t.Error("the password appears in its own hash")
	}
	if !verifyPassword(a, "correcthorsebattery") {
		t.Error("the correct password did not verify")
	}
	if verifyPassword(a, "correcthorsebatterx") {
		t.Error("a wrong password verified")
	}
	// A malformed record must fail closed rather than panic or pass.
	for _, bad := range []string{"", "garbage", "pbkdf2-sha256$notanumber$a$b", "md5$1$a$b"} {
		if verifyPassword(bad, "anything") {
			t.Errorf("malformed hash %q verified", bad)
		}
	}
}

func TestAdminAccountLifecycle(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()

	if n, _ := s.CountAdminUsers(ctx); n != 0 {
		t.Fatalf("a fresh database reported %d accounts", n)
	}
	if _, err := s.CreateAdminUser(ctx, "you@example.com", "short"); !errors.Is(err, ErrWeakPassword) {
		t.Errorf("short password gave %v, want ErrWeakPassword", err)
	}
	if _, err := s.CreateAdminUser(ctx, "not-an-email", "longenoughpassword"); err == nil {
		t.Error("accepted an address with no @ in it")
	}

	u, err := s.CreateAdminUser(ctx, "  YOU@Example.com  ", "correcthorsebattery")
	if err != nil {
		t.Fatal(err)
	}
	if u.Email != "you@example.com" {
		t.Errorf("email stored as %q; it must be normalised so case does not lock people out", u.Email)
	}
	if _, err := s.CreateAdminUser(ctx, "you@example.com", "anotherpassword"); !errors.Is(err, ErrUserExists) {
		t.Errorf("duplicate email gave %v, want ErrUserExists", err)
	}

	// Case-insensitive sign-in, since that is how people retype their address.
	if _, err := s.VerifyAdminLogin(ctx, "YOU@EXAMPLE.COM", "correcthorsebattery"); err != nil {
		t.Errorf("sign-in with different capitalisation failed: %v", err)
	}
	if _, err := s.VerifyAdminLogin(ctx, "you@example.com", "wrong"); !errors.Is(err, ErrBadCredential) {
		t.Errorf("wrong password gave %v, want ErrBadCredential", err)
	}
	// An unknown address must report the same error, so the endpoint cannot be
	// used to find out which addresses have accounts.
	if _, err := s.VerifyAdminLogin(ctx, "nobody@example.com", "correcthorsebattery"); !errors.Is(err, ErrBadCredential) {
		t.Errorf("unknown email gave %v, want the same ErrBadCredential", err)
	}

	if err := s.DeleteAdminUser(ctx, u.ID); err == nil {
		t.Error("deleted the only account, locking everyone out")
	}
}

func TestSessionsExpireAndRevoke(t *testing.T) {
	s := testStore(t)
	ctx := context.Background()
	u, err := s.CreateAdminUser(ctx, "you@example.com", "correcthorsebattery")
	if err != nil {
		t.Fatal(err)
	}

	token, err := s.CreateSession(ctx, u.ID)
	if err != nil {
		t.Fatal(err)
	}
	got, err := s.LookupSession(ctx, token)
	if err != nil || got.ID != u.ID {
		t.Fatalf("lookup returned (%v, %v), want the session's owner", got, err)
	}

	// The raw token must not be recoverable from the database.
	var stored string
	if err := s.r.QueryRow(`SELECT token_hash FROM admin_sessions`).Scan(&stored); err != nil {
		t.Fatal(err)
	}
	if stored == token {
		t.Error("the session token is stored verbatim, so a database leak yields live sessions")
	}

	if _, err := s.LookupSession(ctx, "not-a-real-token"); !errors.Is(err, ErrUserNotFound) {
		t.Errorf("unknown token gave %v, want ErrUserNotFound", err)
	}
	if _, err := s.LookupSession(ctx, ""); !errors.Is(err, ErrUserNotFound) {
		t.Error("an empty cookie must not resolve to a session")
	}

	// Changing a password must evict every session for that account; otherwise
	// changing it does not actually remove whoever else was signed in.
	if err := s.SetAdminPassword(ctx, "you@example.com", "abrandnewpassword"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.LookupSession(ctx, token); !errors.Is(err, ErrUserNotFound) {
		t.Error("a session survived a password change")
	}
	if _, err := s.VerifyAdminLogin(ctx, "you@example.com", "abrandnewpassword"); err != nil {
		t.Errorf("the new password does not work: %v", err)
	}

	// Explicit sign-out.
	t2, _ := s.CreateSession(ctx, u.ID)
	if err := s.DeleteSession(ctx, t2); err != nil {
		t.Fatal(err)
	}
	if _, err := s.LookupSession(ctx, t2); !errors.Is(err, ErrUserNotFound) {
		t.Error("a session survived sign-out")
	}

	// Expired sessions must not resolve, even before they are purged.
	t3, _ := s.CreateSession(ctx, u.ID)
	if _, err := s.w.Exec(`UPDATE admin_sessions SET expires_at = 1 WHERE token_hash = ?`, HashKey(t3)); err != nil {
		t.Fatal(err)
	}
	if _, err := s.LookupSession(ctx, t3); !errors.Is(err, ErrUserNotFound) {
		t.Error("an expired session still resolved")
	}
	if err := s.PurgeExpiredSessions(ctx); err != nil {
		t.Fatal(err)
	}
}
