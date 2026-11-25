package chat

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Inkling2 is Inkling 2's template. Every turn is a run of messages, each a
// role token, the message's name if it has one, a content token, the text
// and <|end_message|>; an assistant turn's reasoning is a
// <|content_thinking|> message and each of its tool calls a
// <|content_invoke_tool_json|> one holding {"name":…,"args":{…}}, and the
// turn ends with <|content_model_end_sampling|>. Tools are declared in a
// first system message. Before the first message that is not a system one
// (or at the end, when there is none), a system message gives the thinking
// effort, Options.ReasoningEffort.
//
// There is no generation prompt: the model writes its own role token, so
// Options.AddGenerationPrompt and Options.Thinking change nothing. A tool
// message's header carries its Message.Name (the template can also look it
// up from a tool_call_id, which Message does not have).
var Inkling2 = Template{name: "inkling2", render: renderInkling2, parseCalls: parseInkling2Calls}

// inkling2Efforts are the named reasoning efforts and their levels.
var inkling2Efforts = map[string]float64{
	"none": 0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99,
}

var inkling2Roles = map[string]string{
	"user": "<|message_user|>", "assistant": "<|message_model|>", "system": "<|message_system|>", "tool": "<|message_tool|>",
}

// inkling2Effort writes the thinking effort message: the level of effort, a
// name from inkling2Efforts or a number from 0 to 0.99, or 0.9 when empty.
func inkling2Effort(b *strings.Builder, effort string) error {
	level := 0.9
	if effort != "" {
		if v, ok := inkling2Efforts[strings.TrimSpace(effort)]; ok {
			level = v
		} else if v, err := strconv.ParseFloat(effort, 64); err == nil {
			level = v
		} else {
			return fmt.Errorf("chat: unknown reasoning effort %q", effort)
		}
	}
	if !(level >= 0 && level <= 0.99) {
		return errors.New("chat: reasoning effort must be in [0.0, 0.99]")
	}
	b.WriteString("<|message_system|><|content_text|>Thinking effort level: ")
	if level == 0 {
		b.WriteString("0")
	} else {
		s, err := pythonNumber(strconv.FormatFloat(level, 'e', -1, 64))
		if err != nil {
			return err
		}
		b.WriteString(s)
	}
	b.WriteString("<|end_message|>")
	return nil
}

func renderInkling2(messages []Message, opts Options) (string, error) {
	var b strings.Builder
	if len(opts.Tools) > 0 {
		var specs []string
		for i, raw := range opts.Tools {
			spec, err := inkling2ToolSpec(raw)
			if err != nil {
				return "", fmt.Errorf("chat: tool %d: %w", i, err)
			}
			specs = append(specs, spec)
		}
		list, err := pythonJSONStyle(json.RawMessage("["+strings.Join(specs, ",")+"]"), jsonStyle{compact: true, sortKeys: true})
		if err != nil {
			return "", err
		}
		b.WriteString("<|message_system|>tool_declare<|content_xml|>" + list + "<|end_message|>")
	}

	effortDone := false
	for i, m := range messages {
		role, ok := inkling2Roles[m.Role]
		if !ok {
			return "", fmt.Errorf("chat: message %d has role %q", i, m.Role)
		}
		if !effortDone && m.Role != "system" {
			if err := inkling2Effort(&b, opts.ReasoningEffort); err != nil {
				return "", err
			}
			effortDone = true
		}
		header := role
		if (m.Role == "tool" || m.Role == "assistant") && m.Name != "" {
			header += m.Name
		}
		assistant := m.Role == "assistant"
		if assistant && m.Reasoning != "" {
			b.WriteString(header + "<|content_thinking|>" + m.Reasoning + "<|end_message|>")
		}
		if m.Content != "" || !(assistant && (len(m.ToolCalls) > 0 || m.Reasoning != "")) {
			b.WriteString(header + "<|content_text|>" + m.Content + "<|end_message|>")
		}
		if assistant {
			for k, call := range m.ToolCalls {
				if m.Name != "" && m.Name != call.Name {
					return "", fmt.Errorf("chat: message %d: assistant name %s does not match its tool call %s", i, m.Name, call.Name)
				}
				args, err := inkling2Arguments(call.Arguments)
				if err != nil {
					return "", fmt.Errorf("chat: message %d, tool call %d: %w", i, k, err)
				}
				name, _ := json.Marshal(call.Name)
				nameJSON, err := pythonJSONStyle(name, jsonStyle{compact: true})
				if err != nil {
					return "", err
				}
				b.WriteString("<|message_model|>" + call.Name + "<|content_invoke_tool_json|>")
				b.WriteString(`{"name":` + nameJSON + `,"args":{` + args + "}}<|end_message|>")
			}
			b.WriteString("<|content_model_end_sampling|>")
		}
	}
	if !effortDone {
		if err := inkling2Effort(&b, opts.ReasoningEffort); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

// inkling2ToolSpec returns a tool as the template declares it, a JSON object
// {"description", "name", "parameters", "type"} from the tool or its
// "function".
func inkling2ToolSpec(raw json.RawMessage) (string, error) {
	var tool map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tool); err != nil {
		return "", errors.New("a tool must be a JSON object")
	}
	fn := tool
	if f, ok := tool["function"]; ok {
		if err := json.Unmarshal(f, &fn); err != nil {
			return "", errors.New("a tool's function must be a JSON object")
		}
	}
	name, ok := fn["name"]
	if !ok {
		return "", errors.New("a tool has no name")
	}
	or := func(v json.RawMessage, fallback string) string {
		if !truthy(v) {
			return fallback
		}
		return string(v)
	}
	return `{"description":` + or(fn["description"], `""`) + `,"name":` + string(name) +
		`,"parameters":` + or(fn["parameters"], "{}") + `,"type":` + or(tool["type"], `"function"`) + "}", nil
}

// truthy reports whether a JSON value is true in Jinja: present, and not
// null, false, 0, "" or empty.
func truthy(v json.RawMessage) bool {
	switch s := strings.TrimSpace(string(v)); s {
	case "", "null", "false", "0", `""`, "[]", "{}":
		return false
	}
	return true
}

// inkling2Arguments writes a tool call's arguments as the template does:
// the object's entries in key order, each key as tojson writes it and each
// value as tojson with compact separators. Arguments must be an object (or
// empty).
func inkling2Arguments(raw json.RawMessage) (string, error) {
	if !truthy(raw) {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return "", errors.New("tool call arguments must be a parsed object, not a JSON string")
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return "", errors.New("tool call arguments must be an object")
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	for i, k := range keys {
		if i > 0 {
			b.WriteString(",")
		}
		writePythonString(&b, k)
		v, err := pythonJSONStyle(obj[k], jsonStyle{compact: true})
		if err != nil {
			return "", err
		}
		b.WriteString(":" + v)
	}
	return b.String(), nil
}

// parseInkling2Calls reads a reply's messages, each ending <|end_message|>
// and starting, after an optional <|message_model|> and name, with a content
// token: text (and tool errors) make the content, joined; each
// <|content_invoke_tool_json|> message is a call {"name":…,"args":{…}};
// reasoning is skipped. <|content_model_end_sampling|> ends the reply.
func parseInkling2Calls(reply string) (string, []ToolCall, error) {
	reply, _, _ = strings.Cut(reply, "<|content_model_end_sampling|>")
	var content strings.Builder
	var calls []ToolCall
	for _, msg := range strings.Split(reply, "<|end_message|>") {
		if rest, ok := strings.CutPrefix(msg, "<|message_model|>"); ok {
			if i := strings.Index(rest, "<|"); i >= 0 {
				msg = rest[i:]
			}
		}
		switch {
		case strings.HasPrefix(msg, "<|content_thinking|>"):
		case strings.HasPrefix(msg, "<|content_invoke_tool_json|>"):
			var call struct {
				Name *string         `json:"name"`
				Args json.RawMessage `json:"args"`
			}
			if err := json.Unmarshal([]byte(strings.TrimPrefix(msg, "<|content_invoke_tool_json|>")), &call); err != nil {
				return "", nil, fmt.Errorf("chat: tool call %d: %w", len(calls), err)
			}
			if call.Name == nil {
				return "", nil, fmt.Errorf("chat: tool call %d has no name", len(calls))
			}
			if call.Args == nil {
				call.Args = json.RawMessage("{}")
			}
			calls = append(calls, ToolCall{Name: *call.Name, Arguments: call.Args})
		default:
			for _, tok := range []string{"<|content_text|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>"} {
				if text, ok := strings.CutPrefix(msg, tok); ok {
					content.WriteString(text)
					break
				}
			}
		}
	}
	return content.String(), calls, nil
}
