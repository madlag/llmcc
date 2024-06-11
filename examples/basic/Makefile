all: book.jschema

%.jschema: %.jinja
	llmcc -m gpt-4 -o $@ $<
