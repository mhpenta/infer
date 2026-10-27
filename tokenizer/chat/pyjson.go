package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// pythonJSON writes a JSON value as transformers' tojson filter does:
// Python's json.dumps(value, ensure_ascii=False) applied to the value
// json.loads would parse. Keys keep their order; separators are ", " and
// ": "; strings escape only quotes, backslashes and control characters;
// integers are written exactly and other numbers as Python floats (1e-05,
// 100.0).
func pythonJSON(raw json.RawMessage) (string, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	var b strings.Builder
	if err := writePython(&b, d); err != nil {
		return "", fmt.Errorf("chat: %w", err)
	}
	if _, err := d.Token(); err != io.EOF {
		return "", fmt.Errorf("chat: %s holds more than one JSON value", raw)
	}
	return b.String(), nil
}

// writePython writes the next value from d.
func writePython(b *strings.Builder, d *json.Decoder) error {
	tok, err := d.Token()
	if err != nil {
		return err
	}
	switch v := tok.(type) {
	case json.Delim:
		closing, sep := "]", ", "
		if v == '{' {
			closing = "}"
		}
		b.WriteString(v.String())
		for first := true; d.More(); first = false {
			if !first {
				b.WriteString(sep)
			}
			if v == '{' {
				key, err := d.Token()
				if err != nil {
					return err
				}
				writePythonString(b, key.(string))
				b.WriteString(": ")
			}
			if err := writePython(b, d); err != nil {
				return err
			}
		}
		if _, err := d.Token(); err != nil { // the closing delimiter
			return err
		}
		b.WriteString(closing)
	case string:
		writePythonString(b, v)
	case json.Number:
		s, err := pythonNumber(string(v))
		if err != nil {
			return err
		}
		b.WriteString(s)
	case bool:
		b.WriteString(strconv.FormatBool(v))
	case nil:
		b.WriteString("null")
	}
	return nil
}

// writePythonString writes s as json.dumps(s, ensure_ascii=False) does.
func writePythonString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if r < 0x20 {
				fmt.Fprintf(b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// pythonNumber writes a JSON number as Python does after json.loads: one
// without a fraction or exponent is an int, written exactly (-0 becomes 0);
// any other is a float, written as repr writes it: the shortest digits
// that round-trip, in positional notation with at least one decimal when the
// decimal exponent is from -4 to 15, in scientific notation with a signed
// two-digit exponent otherwise.
func pythonNumber(s string) (string, error) {
	if !strings.ContainsAny(s, ".eE") {
		if strings.TrimLeft(s, "-0") == "" {
			return "0", nil
		}
		return s, nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return "", fmt.Errorf("number %s: %w", s, err)
	}
	if f == 0 {
		if strings.HasPrefix(s, "-") {
			return "-0.0", nil
		}
		return "0.0", nil
	}
	sci := strconv.FormatFloat(f, 'e', -1, 64) // d.ddde±XX
	exp, err := strconv.Atoi(sci[strings.IndexByte(sci, 'e')+1:])
	if err != nil {
		return "", err
	}
	if exp < -4 || exp >= 16 {
		return sci, nil
	}
	fixed := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(fixed, ".") {
		fixed += ".0"
	}
	return fixed, nil
}
