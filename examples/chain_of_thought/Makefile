all: story.txt

story_outline.txt : story_outline.prompt
	llmcc -m gpt-4 -o $@ $<

story.txt : story_outline.txt
	llmcc -m gpt-4 -o $@ full_story.prompt $<
