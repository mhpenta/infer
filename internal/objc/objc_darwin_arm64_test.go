package objc

import "testing"

func TestString(t *testing.T) {
	pool := PushPool()
	defer pool.Pop()
	for _, s := range []string{"", "hello", "héllo, 世界 🙂", "a\x00b"} {
		str := NewString(s)
		got := str.String()
		str.Release()
		want := s
		if s == "a\x00b" {
			want = "a" // UTF8String is NUL-terminated
		}
		if got != want {
			t.Errorf("NewString(%q).String() = %q", s, got)
		}
	}
}

func TestClass(t *testing.T) {
	if GetClass("NSString") == 0 || GetClass("MTLCompileOptions") == 0 {
		t.Error("Foundation or Metal class not found")
	}
	if c := GetClass("NoSuchClassInfer"); c != 0 {
		t.Errorf("GetClass of a missing class = %#x", c)
	}
	if RegisterName("release") != selRelease {
		t.Error("selectors are not interned")
	}
}

// TestSendArgs sends a message with the most arguments Send takes and a
// result, through an NSString method: -[NSString rangeOfString:options:range:]
// returns a 16-byte NSRange in two registers, of which Send returns the
// first (location).
func TestSendArgs(t *testing.T) {
	pool := PushPool()
	defer pool.Pop()
	hay, needle := NewString("abcabc"), NewString("c")
	defer hay.Release()
	defer needle.Release()
	sel := RegisterName("rangeOfString:options:range:")
	// Search from 3 for 3 characters: an NSRange argument is two words.
	if loc := hay.Send(sel, uintptr(needle), 0, 3, 3); loc != 5 {
		t.Errorf("location = %d, want 5", loc)
	}
	if n := hay.Send(RegisterName("length")); n != 6 {
		t.Errorf("length = %d, want 6", n)
	}
}
