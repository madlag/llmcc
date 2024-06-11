import asyncio

import click
from jinja2 import Environment, FunctionLoader, select_autoescape

from . import llms


class LLMCC:
    def __init__(self, model, templates_path, fields={}, max_tokens=4096):
        self.model = model
        self.templates_path = templates_path
        self.fields = fields
        self.max_tokens = max_tokens
        self.env = Environment(
            loader=FunctionLoader(lambda template_path: open(template_path).read()),
            autoescape=select_autoescape(["html", "xml", "jinja"]),
        )

    def render_template(self):
        # Convert tuple of fields into a dictionary
        fields_dict = dict(field.split("=") for field in self.fields if "=" in field)
        # Load and render the template
        ret = ""
        for template_path in self.templates_path:
            template = self.env.get_template(template_path)
            ret += template.render(**fields_dict)
        return ret

    async def generate_text(self, prompt):
        llm = llms.LLM.llm_by_name(self.model)
        return await llm.get_full_message(prompt=prompt, max_tokens=self.max_tokens)

    def run(self):
        template = self.render_template()
        # run async function generate_text
        loop = asyncio.new_event_loop()
        ret = loop.run_until_complete(self.generate_text(template))
        return ret


@click.command()
@click.option("--model", "-m", default="gpt-4", help="Model name")
@click.option("--output", "-o", default="output.txt", help="Output text file.")
@click.option("--field", "-f", multiple=True)
@click.argument("templates", nargs=-1)
def main(model, output, field, templates):
    llmcc = LLMCC(model, templates, field)
    result_text = llmcc.run()
    with open(output, "w") as file:
        file.write(result_text)
