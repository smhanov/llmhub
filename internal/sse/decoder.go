package sse

import (
	"bufio"
	"errors"
	"io"
	"strings"
)

// Event represents a single Server-Sent Event frame.
type Event struct {
	Name string
	Data string
}

// Decoder reads SSE streams from an io.Reader.
type Decoder struct {
	reader *bufio.Reader
}

// NewDecoder constructs a Decoder from any readable stream.
func NewDecoder(r io.Reader) *Decoder {
	return &Decoder{reader: bufio.NewReader(r)}
}

// Next returns the next SSE event, blocking until one is available or an error occurs.
func (d *Decoder) Next() (Event, error) {
	if d == nil || d.reader == nil {
		return Event{}, io.EOF
	}
	var evt Event
	for {
		line, err := d.reader.ReadString('\n')
		if err != nil {
			if errors.Is(err, io.EOF) {
				line = strings.TrimRight(line, "\r\n")
				if line != "" {
					d.consumeLine(&evt, line)
				}
				if evt.Data != "" || evt.Name != "" {
					out := evt
					evt = Event{}
					return out, nil
				}
				return Event{}, io.EOF
			}
			return Event{}, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if evt.Data == "" && evt.Name == "" {
				continue
			}
			out := evt
			evt = Event{}
			return out, nil
		}
		d.consumeLine(&evt, line)
	}
}

func (d *Decoder) consumeLine(evt *Event, line string) {
	if strings.HasPrefix(line, ":") {
		return
	}
	switch {
	case strings.HasPrefix(line, "event:"):
		evt.Name = strings.TrimSpace(line[len("event:"):])
	case strings.HasPrefix(line, "data:"):
		if evt.Data != "" {
			evt.Data += "\n"
		}
		evt.Data += strings.TrimSpace(line[len("data:"):])
	default:
		if evt.Data != "" {
			evt.Data += "\n"
		}
		evt.Data += line
	}
}
